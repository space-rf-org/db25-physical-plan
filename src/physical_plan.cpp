#include "db25/physical/physical_plan.hpp"

#include <utility>

namespace db25::physical {

const char* physical_op_to_string(PhysicalOp op) noexcept {
    switch (op) {
        case PhysicalOp::SeqScan:       return "SeqScan";
        case PhysicalOp::Filter:        return "Filter";
        case PhysicalOp::Project:       return "Project";
        case PhysicalOp::HashJoin:      return "HashJoin";
        case PhysicalOp::MergeJoin:     return "MergeJoin";
        case PhysicalOp::NestedLoopJoin: return "NestedLoopJoin";
        case PhysicalOp::Sort:          return "Sort";
        case PhysicalOp::FormatConvert: return "FormatConvert";
    }
    return "?";
}

std::optional<PhysicalOp> physical_op_from_name(const std::string& name) noexcept {
    for (const PhysicalOp op : kAllPhysicalOps) {
        if (name == physical_op_to_string(op)) return op;
    }
    return std::nullopt;
}

const char* join_kind_to_string(ast::JoinType k) noexcept {
    switch (k) {
        case ast::JoinType::Inner:       return "inner";
        case ast::JoinType::Left:        return "left";
        case ast::JoinType::Right:       return "right";
        case ast::JoinType::Full:        return "full";
        case ast::JoinType::Cross:       return "cross";
        case ast::JoinType::Lateral:     return "lateral";
        case ast::JoinType::LeftLateral: return "leftlateral";
    }
    return "?";
}

bool join_null_extends_left(ast::JoinType k) noexcept {
    // The LEFT input is null-extended when unmatched RIGHT rows must be kept.
    return k == ast::JoinType::Right || k == ast::JoinType::Full;
}

bool join_null_extends_right(ast::JoinType k) noexcept {
    return k == ast::JoinType::Left || k == ast::JoinType::Full ||
           k == ast::JoinType::LeftLateral;
}

bool join_is_lateral(ast::JoinType k) noexcept {
    return k == ast::JoinType::Lateral || k == ast::JoinType::LeftLateral;
}

const char* freshness_to_string(Freshness f) noexcept {
    switch (f) {
        case Freshness::Any:   return "any";
        case Freshness::Fresh: return "fresh";
        case Freshness::Stale: return "stale";
    }
    return "?";
}

const char* storage_format_to_string(StorageFormat f) noexcept {
    switch (f) {
        case StorageFormat::Any:    return "any";
        case StorageFormat::Row:    return "row";
        case StorageFormat::Column: return "column";
    }
    return "?";
}

std::size_t expected_arity(PhysicalOp op) noexcept {
    switch (op) {
        case PhysicalOp::SeqScan:       return 0;
        case PhysicalOp::Filter:        return 1;
        case PhysicalOp::Project:       return 1;
        case PhysicalOp::HashJoin:      return 2;
        case PhysicalOp::MergeJoin:     return 2;
        case PhysicalOp::NestedLoopJoin: return 2;
        case PhysicalOp::Sort:          return 1;
        case PhysicalOp::FormatConvert: return 1;
    }
    return 0;
}

PhysicalNodePtr make_seq_scan(std::string table, Schema output) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::SeqScan);
    n->table_name = std::move(table);
    n->output = std::move(output);
    return n;
}

PhysicalNodePtr make_filter(PhysicalNodePtr input, const Expr* predicate) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::Filter);
    n->predicate = predicate;
    // A filter does not change the row shape: its output is its input's schema.
    n->output = input->output;
    n->children.push_back(std::move(input));
    return n;
}

PhysicalNodePtr make_project(PhysicalNodePtr input, Schema output,
                             std::vector<const Expr*> projections) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::Project);
    n->output = std::move(output);
    n->projections = std::move(projections);
    n->children.push_back(std::move(input));
    return n;
}

PhysicalNodePtr make_hash_join(PhysicalNodePtr left, PhysicalNodePtr right,
                               std::vector<HashKey> keys, Schema output,
                               std::vector<const Expr*> residual) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::HashJoin);
    n->hash_keys = std::move(keys);
    n->residual = std::move(residual);
    n->output = std::move(output);
    n->children.push_back(std::move(left));
    n->children.push_back(std::move(right));
    return n;
}

PhysicalNodePtr make_merge_join(PhysicalNodePtr left, PhysicalNodePtr right,
                                std::vector<HashKey> keys, Schema output,
                                std::vector<const Expr*> residual) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::MergeJoin);
    n->hash_keys = std::move(keys);
    n->residual = std::move(residual);
    n->output = std::move(output);
    n->children.push_back(std::move(left));
    n->children.push_back(std::move(right));
    return n;
}

PhysicalNodePtr make_nested_loop_join(PhysicalNodePtr left, PhysicalNodePtr right,
                                     Schema output, std::vector<const Expr*> residual) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::NestedLoopJoin);
    n->residual = std::move(residual);   // no hash_keys: that is the whole point
    n->output = std::move(output);
    n->children.push_back(std::move(left));
    n->children.push_back(std::move(right));
    return n;
}

PhysicalNodePtr make_sort(PhysicalNodePtr input, std::vector<SortKey> keys) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::Sort);
    n->sort_keys = std::move(keys);
    n->output = input->output;  // a sort reorders rows, it does not reshape them
    n->children.push_back(std::move(input));
    return n;
}

PhysicalNodePtr make_format_convert(PhysicalNodePtr input, StorageFormat target) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::FormatConvert);
    n->target_format = target;
    n->output = input->output;  // a format change does not reshape the rows
    n->children.push_back(std::move(input));
    return n;
}

}  // namespace db25::physical
