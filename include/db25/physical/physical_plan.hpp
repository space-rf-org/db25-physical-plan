#pragma once
// The DB25 physical plan IR - the ONE physical operator vocabulary (design D3).
//
// A physical plan is an executable operator DAG: it fixes the access path, the
// join algorithm, the data layout, and (later) the parallelism and storage
// substrate that a logical plan left open. Every planner tier and every rule
// emits nodes of THIS IR, costed by the one cost model; the IR grows by editing
// the spec (the IDL, Unit 0.3), never by forking a second node type.
//
// Increment 0 admits four operators (scan / filter / project / one join). The
// Cascades memo (memo.hpp) is where the many transient candidates live during
// search; a PhysicalNode is the concrete plan the memo's winner extracts to.
#include "db25/plan/expr_ir.hpp"       // complete db25::plan::Expr (borrowed payloads)
#include "db25/plan/logical_plan.hpp"  // Schema, ColumnSchema

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace db25::physical {

using plan::ColumnSchema;
using plan::Expr;
using plan::Schema;

// The physical operators of Increment 0.
enum class PhysicalOp : std::uint8_t {
    SeqScan,   // base-table sequential scan (an access path)
    Filter,    // predicate filter
    Project,   // projection to named output columns
    HashJoin,  // hash join (the one join algorithm of Increment 0)
};

[[nodiscard]] const char* physical_op_to_string(PhysicalOp op) noexcept;

struct PhysicalNode;
using PhysicalNodePtr = std::unique_ptr<PhysicalNode>;

// One equi-join key pair for a HashJoin: positional column indices into the left
// and right input schemas respectively.
struct HashKey {
    std::uint32_t left_index = 0;
    std::uint32_t right_index = 0;
};

// A physical plan node. Ownership is a plain owning tree (unique_ptr children),
// mirroring the logical plan: the extracted plan is small and short-lived.
//
// Expression payloads are BORROWED (const Expr*): physical planning selects
// algorithms and access paths, it does not rewrite expressions, so the physical
// plan points at the expressions the logical plan already owns. The logical plan
// (and the AST beneath it) must therefore outlive the physical plan - the same
// borrowing contract the analyzer and logical plan already rely on.
struct PhysicalNode {
    PhysicalOp op;
    std::vector<PhysicalNodePtr> children;
    Schema output;  // the columns this node produces

    // --- op-specific payload (union-by-convention on `op`) ---
    std::string table_name;                // SeqScan: the base table
    const Expr* predicate = nullptr;       // Filter: predicate; HashJoin: optional residual
    std::vector<const Expr*> projections;  // Project: one borrowed expr per output column
    std::vector<HashKey> hash_keys;        // HashJoin: equi-join key pairs

    explicit PhysicalNode(PhysicalOp o) : op(o) {}
};

// Builders - keep the tests and the future lowering (Unit 0.6) terse.
[[nodiscard]] PhysicalNodePtr make_seq_scan(std::string table, Schema output);
[[nodiscard]] PhysicalNodePtr make_filter(PhysicalNodePtr input, const Expr* predicate);
[[nodiscard]] PhysicalNodePtr make_project(PhysicalNodePtr input, Schema output,
                                           std::vector<const Expr*> projections);
[[nodiscard]] PhysicalNodePtr make_hash_join(PhysicalNodePtr left, PhysicalNodePtr right,
                                             std::vector<HashKey> keys, Schema output,
                                             const Expr* residual = nullptr);

}  // namespace db25::physical
