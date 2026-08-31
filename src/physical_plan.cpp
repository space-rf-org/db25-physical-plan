#include "db25/physical/physical_plan.hpp"

#include <utility>

namespace db25::physical {

const char* physical_op_to_string(PhysicalOp op) noexcept {
    switch (op) {
        case PhysicalOp::SeqScan:  return "SeqScan";
        case PhysicalOp::Filter:   return "Filter";
        case PhysicalOp::Project:  return "Project";
        case PhysicalOp::HashJoin: return "HashJoin";
    }
    return "?";
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
                               const Expr* residual) {
    auto n = std::make_unique<PhysicalNode>(PhysicalOp::HashJoin);
    n->hash_keys = std::move(keys);
    n->predicate = residual;
    n->output = std::move(output);
    n->children.push_back(std::move(left));
    n->children.push_back(std::move(right));
    return n;
}

}  // namespace db25::physical
