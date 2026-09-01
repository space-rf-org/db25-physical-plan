#include "db25/physical/memo.hpp"

#include "db25/physical/cost.hpp"
#include "db25/physical/properties.hpp"
#include "db25/physical/structural_key.hpp"

#include <limits>
#include <utility>

namespace db25::physical {

namespace {
std::uint64_t mix64(std::uint64_t h, std::uint64_t v) noexcept {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}
}  // namespace

void GroupKey::finish() noexcept {
    comparable = expr_is_comparable(predicate);
    for (const Expr* e : residual) comparable = comparable && expr_is_comparable(e);
    for (const Expr* e : op_exprs) comparable = comparable && expr_is_comparable(e);

    std::uint64_t h = mix64(13, static_cast<std::uint64_t>(logical_op));
    if (table_name != nullptr) {
        for (const char c : *table_name) h = mix64(h, static_cast<unsigned char>(c));
    }
    if (output != nullptr) h = mix64(h, schema_hash(*output));
    for (const GroupId g : inputs) h = mix64(h, g);
    h = mix64(h, expr_structural_hash(predicate));
    for (const Expr* e : residual) h = mix64(h, expr_structural_hash(e));
    for (const Expr* e : op_exprs) h = mix64(h, expr_structural_hash(e));
    for (const HashKey& k : hash_keys) { h = mix64(h, k.left_index); h = mix64(h, k.right_index); }
    h = mix64(h, static_cast<std::uint64_t>(join_kind));
    for (const SortKey& k : sort_keys) {
        h = mix64(h, k.column);
        h = mix64(h, static_cast<std::uint64_t>(k.descending));
        h = mix64(h, static_cast<std::uint64_t>(k.nulls_specified));
        h = mix64(h, static_cast<std::uint64_t>(k.nulls_first));
    }
    h = mix64(h, static_cast<std::uint64_t>(limits.has_limit));
    h = mix64(h, static_cast<std::uint64_t>(limits.limit));
    h = mix64(h, static_cast<std::uint64_t>(limits.has_offset));
    h = mix64(h, static_cast<std::uint64_t>(limits.offset));
    h = mix64(h, group_key_count);
    hash = h;
}

bool GroupKey::equals(const GroupKey& o) const noexcept {
    if (!comparable || !o.comparable) return false;  // never share an unknown payload
    if (logical_op != o.logical_op) return false;
    if (join_kind != o.join_kind) return false;
    if (sort_keys.size() != o.sort_keys.size()) return false;
    for (std::size_t i = 0; i < sort_keys.size(); ++i) {
        if (!(sort_keys[i] == o.sort_keys[i])) return false;
    }
    if (limits.has_limit != o.limits.has_limit ||
        limits.has_offset != o.limits.has_offset ||
        (limits.has_limit && limits.limit != o.limits.limit) ||
        (limits.has_offset && limits.offset != o.limits.offset)) {
        return false;
    }
    const std::string empty;
    if ((table_name ? *table_name : empty) != (o.table_name ? *o.table_name : empty)) return false;
    if (inputs != o.inputs) return false;
    if (hash_keys.size() != o.hash_keys.size()) return false;
    for (std::size_t i = 0; i < hash_keys.size(); ++i) {
        if (hash_keys[i].left_index != o.hash_keys[i].left_index ||
            hash_keys[i].right_index != o.hash_keys[i].right_index) {
            return false;
        }
    }
    if ((output == nullptr) != (o.output == nullptr)) return false;
    if (output != nullptr && !schema_equal(*output, *o.output)) return false;
    if (!expr_structurally_equal(predicate, o.predicate)) return false;
    if (residual.size() != o.residual.size()) return false;
    for (std::size_t i = 0; i < residual.size(); ++i) {
        if (!expr_structurally_equal(residual[i], o.residual[i])) return false;
    }
    if (group_key_count != o.group_key_count) return false;
    if (op_exprs.size() != o.op_exprs.size()) return false;
    for (std::size_t i = 0; i < op_exprs.size(); ++i) {
        if (!expr_structurally_equal(op_exprs[i], o.op_exprs[i])) return false;
    }
    return true;
}

std::optional<GroupId> Memo::find_group(const GroupKey& key) const {
    if (!key.comparable) return std::nullopt;
    const auto it = index_.find(key.hash);
    if (it == index_.end()) return std::nullopt;
    for (const auto& [k, id] : it->second) {
        if (k.equals(key)) return id;  // verified, not merely hash-equal
    }
    return std::nullopt;
}

void Memo::index_group(const GroupKey& key, GroupId id) {
    if (!key.comparable) return;
    index_[key.hash].emplace_back(key, id);
}

GroupId Memo::add_group(const Schema& output) {
    const auto id = static_cast<GroupId>(groups_.size());
    Group g;
    g.id = id;
    g.output = &output;  // borrowed; must outlive the memo
    groups_.push_back(std::move(g));
    return id;
}

std::uint32_t Memo::add_expr(GroupId group, GroupExpr expr) {
    Group& g = groups_[group];
    const auto index = static_cast<std::uint32_t>(g.exprs.size());
    g.exprs.push_back(std::move(expr));
    return index;
}

void Memo::set_winner(GroupId group, const PhysicalProperties& required,
                      std::uint32_t expr_index, double cost, PhysicalProperties provided,
                      ArityVec<PhysicalProperties> input_required) {
    Group& g = groups_[group];
    for (WinnerEntry& w : g.winners) {
        if (w.required == required) {  // replace in place: one winner per key
            w.expr_index = expr_index;
            w.cost = cost;
            w.provided = std::move(provided);
            w.input_required = std::move(input_required);
            return;
        }
    }
    // A group is asked for a handful of goals at most, and a WinnerEntry is not
    // small, so this reserve trades a little unused capacity for not reallocating
    // (and copying every entry) as goals arrive. Two, not four: it covers the
    // common case without over-allocating for the many groups asked only once.
    if (g.winners.capacity() == 0) g.winners.reserve(2);
    g.winners.push_back(WinnerEntry{required, expr_index, cost, std::move(provided),
                                    std::move(input_required)});
}

void Memo::set_rows(GroupId group, double rows) {
    groups_[group].rows = rows;
}

bool Memo::select_cheapest(GroupId group, const CalibrationProfile& cal) {
    return select_cheapest(group, Group::unconstrained(), cal);
}

bool Memo::select_cheapest(GroupId group, const PhysicalProperties& required,
                           const CalibrationProfile& cal) {
    Group& g = groups_[group];
    if (g.exprs.empty()) return false;

    // Each candidate is charged the enforcement ITS OWN output would need to meet
    // the requirement. That is what makes the answer requirement-specific: a
    // candidate that already provides the property pays nothing and can win here
    // while losing the unconstrained question, and vice versa.
    std::uint32_t best = g.exprs.size();  // = "none"
    double best_cost = std::numeric_limits<double>::infinity();
    for (std::uint32_t i = 0; i < g.exprs.size(); ++i) {
        const double c =
            g.exprs[i].cost + enforcement_cost(g.exprs[i].provided, required, g.rows, cal);
        if (c < best_cost) { best_cost = c; best = i; }
    }
    // Infinite means nothing here can satisfy the requirement at any price - an
    // unmet Fresh requirement, which no enforcer establishes. That is a failure to
    // report, not a winner to record.
    if (best == g.exprs.size() || best_cost == std::numeric_limits<double>::infinity()) {
        return false;
    }

    // What the winner will provide once its enforcers are in place: the properties
    // it already had, upgraded by whatever the requirement demanded.
    PhysicalProperties provided = g.exprs[best].provided;
    if (required.format != StorageFormat::Any) provided.format = required.format;
    if (!required.sort.empty() && !satisfies(g.exprs[best].provided, required)) {
        provided.sort = required.sort;
    }
    // Single-level chooser: it does not optimize inputs, so the child goals it
    // records are the operator's own input requirements.
    set_winner(group, required, best, best_cost, std::move(provided),
               g.exprs[best].input_reqs);
    return true;
}

PhysicalNodePtr Memo::extract_winner(GroupId id) const {
    if (id == kInvalidGroup) {
        id = root_;
    }
    return extract_winner_for(id, Group::unconstrained());
}

PhysicalNodePtr Memo::extract_winner_for(GroupId id, const PhysicalProperties& required) const {
    if (id == kInvalidGroup || id >= groups_.size()) {
        return nullptr;
    }
    const Group& g = groups_[id];
    const WinnerEntry* won = g.winner_for(required);
    if (won == nullptr) {
        return nullptr;  // this group was never optimized for that requirement
    }
    const GroupExpr& ge = g.exprs[won->expr_index];

    auto node = std::make_unique<PhysicalNode>(ge.op);
    if (g.output != nullptr) node->output = *g.output;
    node->table_name = g.table_name;
    node->scan_format = ge.scan_format;
    node->scan_freshness = ge.scan_freshness;
    node->predicate = g.predicate;
    node->residual = g.residual;
    // A Project's expressions and an Aggregate's payload share one vector on the
    // group; they are separate fields on the extracted NODE, which is not in a
    // deque and so is not size-critical.
    if (ge.op == PhysicalOp::HashAggregate || ge.op == PhysicalOp::StreamingAggregate) {
        node->group_keys.assign(g.op_exprs.begin(), g.op_exprs.begin() + g.group_key_count);
        node->aggregates.assign(g.op_exprs.begin() + g.group_key_count, g.op_exprs.end());
    } else {
        node->projections = g.op_exprs;
    }
    node->hash_keys = g.hash_keys;
    node->join_kind = g.join_kind;
    node->sort_keys = g.sort_keys;
    node->limits = g.limits;

    // Recurse on the SAME goals the winner was costed against, not on each
    // child's unconstrained winner: a child optimized for "sorted on k" is a
    // different plan from the same child optimized for nothing, and extracting
    // the wrong one would build a plan nobody costed.
    const ArityVec<PhysicalProperties>& reqs = won->input_required;
    for (std::size_t i = 0; i < g.inputs.size(); ++i) {
        const PhysicalProperties& child_req =
            i < reqs.size() ? reqs[i] : Group::unconstrained();
        auto child = extract_winner_for(g.inputs[i], child_req);
        if (!child) {
            return nullptr;  // an input group had no winner: extraction fails
        }
        // Establish anything this operator requires but the input does not give.
        // enforce() returns null when the requirement is not enforceable at all;
        // that must fail extraction, not silently emit the unenforced input.
        child = enforce(std::move(child), child_req);
        if (!child) return nullptr;
        node->children.push_back(std::move(child));
    }
    return node;
}

}  // namespace db25::physical
