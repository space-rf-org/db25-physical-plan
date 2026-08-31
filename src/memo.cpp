#include "db25/physical/memo.hpp"

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
    node->predicate = ge.predicate;
    node->projections = ge.projections;
    node->hash_keys = ge.hash_keys;

    for (const GroupId input : ge.inputs) {
        auto child = extract_winner(input);
        if (!child) {
            return nullptr;  // an input group had no winner: extraction fails
        }
        node->children.push_back(std::move(child));
    }
    return node;
}

}  // namespace db25::physical
