#include "db25/physical/memo.hpp"

#include "db25/physical/properties.hpp"

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

void Memo::set_winner(GroupId group, std::uint32_t expr_index) {
    groups_[group].winner = expr_index;
}

void Memo::set_rows(GroupId group, double rows) {
    groups_[group].rows = rows;
}

void Memo::set_provided(GroupId group, PhysicalProperties provided) {
    groups_[group].provided = std::move(provided);
}

bool Memo::select_cheapest(GroupId group) {
    Group& g = groups_[group];
    if (g.exprs.empty()) return false;
    std::uint32_t best = 0;
    for (std::uint32_t i = 1; i < g.exprs.size(); ++i) {
        if (g.exprs[i].cost < g.exprs[best].cost) best = i;
    }
    g.winner = best;
    return true;
}

PhysicalNodePtr Memo::extract_winner(GroupId id) const {
    if (id == kInvalidGroup) {
        id = root_;
    }
    if (id == kInvalidGroup || id >= groups_.size()) {
        return nullptr;
    }
    const Group& g = groups_[id];
    if (!g.winner) {
        return nullptr;
    }
    const GroupExpr& ge = g.exprs[*g.winner];

    auto node = std::make_unique<PhysicalNode>(ge.op);
    node->output = g.output;
    node->table_name = ge.table_name;
    node->scan_format = ge.scan_format;
    node->scan_freshness = ge.scan_freshness;
    node->predicate = ge.predicate;
    node->projections = ge.projections;
    node->hash_keys = ge.hash_keys;

    const std::vector<PhysicalProperties> reqs =
        required_input_properties(ge.op, ge.hash_keys);
    for (std::size_t i = 0; i < ge.inputs.size(); ++i) {
        auto child = extract_winner(ge.inputs[i]);
        if (!child) {
            return nullptr;  // an input group had no winner: extraction fails
        }
        // Establish anything this operator requires but the input does not give.
        if (i < reqs.size()) child = enforce(std::move(child), reqs[i]);
        node->children.push_back(std::move(child));
    }
    return node;
}

}  // namespace db25::physical
