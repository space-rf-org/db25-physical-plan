#include "db25/physical/memo.hpp"

#include "db25/physical/cost.hpp"
#include "db25/physical/properties.hpp"

#include <limits>
#include <utility>

namespace db25::physical {

GroupId Memo::add_group(Schema output) {
    const auto id = static_cast<GroupId>(groups_.size());
    Group g;
    g.id = id;
    g.output = std::move(output);
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
                      std::vector<PhysicalProperties> input_required) {
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
    node->output = g.output;
    node->table_name = ge.table_name;
    node->scan_format = ge.scan_format;
    node->scan_freshness = ge.scan_freshness;
    node->predicate = ge.predicate;
    node->residual = ge.residual;
    node->projections = ge.projections;
    node->hash_keys = ge.hash_keys;

    // Recurse on the SAME goals the winner was costed against, not on each
    // child's unconstrained winner: a child optimized for "sorted on k" is a
    // different plan from the same child optimized for nothing, and extracting
    // the wrong one would build a plan nobody costed.
    const std::vector<PhysicalProperties>& reqs = won->input_required;
    for (std::size_t i = 0; i < ge.inputs.size(); ++i) {
        const PhysicalProperties& child_req =
            i < reqs.size() ? reqs[i] : Group::unconstrained();
        auto child = extract_winner_for(ge.inputs[i], child_req);
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
