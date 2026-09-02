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
        case PhysicalOp::Limit:         return "Limit";
        case PhysicalOp::Window:        return "Window";
        case PhysicalOp::HashDistinct:  return "HashDistinct";
        case PhysicalOp::StreamingDistinct: return "StreamingDistinct";
        case PhysicalOp::UnionAll:      return "UnionAll";
        case PhysicalOp::HashSetOp:     return "HashSetOp";
        case PhysicalOp::ValuesScan:    return "ValuesScan";
        case PhysicalOp::HashSemiJoin:  return "HashSemiJoin";
        case PhysicalOp::HashAntiJoin:  return "HashAntiJoin";
        case PhysicalOp::NestedLoopSemiJoin: return "NestedLoopSemiJoin";
        case PhysicalOp::NestedLoopAntiJoin: return "NestedLoopAntiJoin";
        case PhysicalOp::HashGroupingSets: return "HashGroupingSets";
        case PhysicalOp::HashAggregate: return "HashAggregate";
        case PhysicalOp::StreamingAggregate: return "StreamingAggregate";
        case PhysicalOp::RecursiveFixpoint: return "RecursiveFixpoint";
        case PhysicalOp::WorkingTableScan: return "WorkingTableScan";
        case PhysicalOp::CreateTableAs: return "CreateTableAs";
        case PhysicalOp::Insert:        return "Insert";
        case PhysicalOp::Update:        return "Update";
        case PhysicalOp::Delete:        return "Delete";
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

bool is_nested_loop(PhysicalOp op) noexcept {
    return op == PhysicalOp::NestedLoopJoin || op == PhysicalOp::NestedLoopSemiJoin ||
           op == PhysicalOp::NestedLoopAntiJoin;
}

bool computes_grouping_sets(PhysicalOp op) noexcept {
    return op == PhysicalOp::HashGroupingSets;
}

bool is_aggregate_family(PhysicalOp op) noexcept {
    return op == PhysicalOp::HashAggregate || op == PhysicalOp::StreamingAggregate ||
           op == PhysicalOp::HashGroupingSets;
}

bool is_build_side_choosable(PhysicalOp op, ast::JoinType join_kind) noexcept {
    if (op != PhysicalOp::HashJoin) return false;
    return join_kind == ast::JoinType::Inner || join_kind == ast::JoinType::Cross;
}

bool needs_equi_key(PhysicalOp op) noexcept {
    // A merge join has nothing to merge on without a key; the three hash-building
    // operators have nothing to hash on. Grouped here rather than listed at each
    // call site so a new hash operator cannot be added without meeting this.
    return op == PhysicalOp::MergeJoin || op == PhysicalOp::HashJoin ||
           op == PhysicalOp::HashSemiJoin || op == PhysicalOp::HashAntiJoin;
}

const char* set_op_to_string(ast::SetOp op) noexcept {
    switch (op) {
        case ast::SetOp::Union:        return "union";
        case ast::SetOp::UnionAll:     return "union-all";
        case ast::SetOp::Intersect:    return "intersect";
        case ast::SetOp::Except:       return "except";
        case ast::SetOp::IntersectAll: return "intersect-all";
        case ast::SetOp::ExceptAll:    return "except-all";
    }
    return "?";
}

bool set_op_deduplicates(ast::SetOp op) noexcept {
    return op == ast::SetOp::Union || op == ast::SetOp::Intersect ||
           op == ast::SetOp::Except;
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
        case PhysicalOp::Limit:         return 1;
        case PhysicalOp::Window:        return 1;
        case PhysicalOp::HashDistinct:  return 1;
        case PhysicalOp::StreamingDistinct: return 1;
        case PhysicalOp::UnionAll:      return 2;
        case PhysicalOp::HashSetOp:     return 2;
        case PhysicalOp::ValuesScan:    return 0;
        case PhysicalOp::HashSemiJoin:  return 2;
        case PhysicalOp::HashAntiJoin:  return 2;
        case PhysicalOp::NestedLoopSemiJoin: return 2;
        case PhysicalOp::NestedLoopAntiJoin: return 2;
        case PhysicalOp::HashGroupingSets: return 1;
        case PhysicalOp::HashAggregate: return 1;
        case PhysicalOp::StreamingAggregate: return 1;
        // Binary: the anchor and the recursive term, in that order. Two children
        // rather than one with the recursion hidden inside, because the anchor is
        // evaluated ONCE and the term repeatedly - a distinction the plan has to
        // make for an executor to be able to run it.
        case PhysicalOp::RecursiveFixpoint: return 2;
        case PhysicalOp::WorkingTableScan: return 0;
        case PhysicalOp::CreateTableAs: return 1;
        // One input each: the source query for an Insert, the target rows for an
        // Update or a Delete. The TARGET RELATION is a name on the node, not a
        // child - it is written, not read, and a scan of it as an input would say
        // the wrong thing.
        case PhysicalOp::Insert:        return 1;
        case PhysicalOp::Update:        return 1;
        case PhysicalOp::Delete:        return 1;
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

PhysicalNodePtr make_limit(PhysicalNodePtr input, LimitSpec limits) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::Limit);
    n->limits = limits;
    n->output = input->output;  // a limit drops rows, it does not reshape them
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
