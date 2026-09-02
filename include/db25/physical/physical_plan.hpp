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
#include "db25/ast/node_types.hpp"    // ast::JoinType
#include "db25/physical/small_vec.hpp"
#include "db25/plan/expr_ir.hpp"       // complete db25::plan::Expr (borrowed payloads)
#include "db25/plan/logical_plan.hpp"  // Schema, ColumnSchema

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace db25::physical {

using plan::ColumnSchema;
using plan::Expr;
using plan::Schema;

// The physical operators of Increment 0. Sort and FormatConvert are ENFORCERS:
// operators inserted purely to establish a required physical property (a sort
// order, a storage format) that a subplan does not already provide.
enum class PhysicalOp : std::uint8_t {
    SeqScan,        // base-table sequential scan (an access path)
    Filter,         // predicate filter
    Project,        // projection to named output columns
    HashJoin,       // hash join: builds a hash table, indifferent to input order
    MergeJoin,      // merge join: needs both inputs sorted on the join keys, and
                    // keeps that order on its output
    NestedLoopJoin, // nested-loop join: the only join that needs no equi-key, so
                    // the only legal implementation of a cross product or a
                    // purely non-equi join. Quadratic, and priced accordingly.
    Sort,           // establishes a sort order - both the implementation of a
                    // logical Sort (ORDER BY) and the ENFORCER inserted to
                    // satisfy a required order. One operator, because they are
                    // the same operation; what differs is only who asked for it.
    FormatConvert,  // enforcer: converts the storage format (row <-> column)
    HashAggregate,  // GROUP BY via a hash table on the grouping keys: indifferent
                    // to input order, and produces none.
    StreamingAggregate, // GROUP BY over an input already sorted on the grouping
                    // keys: one pass, no hash table, and the group order survives
                    // on the output. Cheaper per row than hashing - and dearer
                    // once a Sort has to be enforced to feed it, which is exactly
                    // the trade the search exists to make.
    // Semi and anti joins, from EXISTS / IN and NOT EXISTS. Both emit the LEFT
    // schema only, and emit each qualifying left row ONCE however many right rows
    // match - which is why they are their own operators rather than a join
    // followed by a distinct. Four of them rather than two with a mode flag: an
    // operator that does not say which of semi and anti it is reads the same as
    // the other, and they are exact complements.
    HashSemiJoin,       // left rows that HAVE a match; hash table on the right
    HashAntiJoin,       // left rows with NO match; hash table on the right
    NestedLoopSemiJoin, // the keyless fallback, as NestedLoopJoin is for a join
    NestedLoopAntiJoin,
    HashGroupingSets, // GROUP BY GROUPING SETS / ROLLUP / CUBE: one pass computing
                    // SEVERAL grouping-key combinations at once, hashing on each
                    // set's active subset. There is no streaming counterpart -
                    // different sets want different input orderings, and no single
                    // sort serves them all.
    HashDistinct,   // SELECT DISTINCT via a hash table on every output column:
                    // indifferent to input order, and produces none.
    StreamingDistinct, // SELECT DISTINCT over an input already sorted on every
                    // output column: adjacent duplicates collapse in one pass, and
                    // the order survives. The same trade as the two aggregates.
    UnionAll,       // concatenate two inputs, keeping duplicates. No comparison
                    // at all, so it is the one set operation that needs neither a
                    // hash table nor sorted input.
    HashSetOp,      // UNION / INTERSECT / EXCEPT (and the ALL variants of the
                    // latter two): builds a hash table on the RIGHT input and
                    // probes it with the left. `set_op` says which.
    ValuesScan,     // a literal row set - VALUES (..), (..) - and the synthetic
                    // single empty row a FROM-less SELECT is evaluated over.
    Window,         // window functions (RANK / SUM(..) OVER (...)): consumes its
                    // input sorted by (PARTITION BY ++ ORDER BY), appends one
                    // column per window function, and emits in that same order.
    Limit,          // LIMIT / OFFSET: pass at most `limit` rows through, after
                    // discarding the first `offset`. Order-SENSITIVE - which
                    // rows survive depends entirely on the order of its input.
    // The two operators a WITH RECURSIVE needs. They are a PAIR: neither means
    // anything without the other, which is why they arrive together.
    RecursiveFixpoint,  // iterate to a fixpoint: evaluate the anchor once, then
                    // the recursive term repeatedly over the rows the previous
                    // iteration produced, until an iteration produces none.
                    // `set_op` says whether UNION (dedup across iterations) or
                    // UNION ALL. Named for what it DOES rather than for the SQL
                    // that produces it - the operator is a fixpoint, and a plan
                    // that said "RecursiveCTE" would be naming its own syntax.
    WorkingTableScan, // read the working table: the rows the previous iteration
                    // produced. A leaf, and the only operator whose input comes
                    // from elsewhere in the same plan - which is exactly what
                    // makes the fixpoint above a fixpoint rather than a join.
    CreateTableAs,  // materialize the input stream into a NEW table. Named for
                    // the statement because that is precisely what it does; a
                    // synonym would only be a mapping that can drift.
    // The write path. Each takes its rows from one input - the source query for
    // an Insert, the (scanned and filtered) target rows for an Update or a
    // Delete - and emits the AFFECTED rows, which is what a RETURNING clause
    // then projects. Three operators rather than one with a mode, for the reason
    // semi and anti joins are four: an operator that does not say which
    // modification it performs reads the same as the other two.
    Insert,
    Update,
    Delete,
};

[[nodiscard]] const char* physical_op_to_string(PhysicalOp op) noexcept;

// Inverse of physical_op_to_string: map a name (as it appears in the spec) back
// to its operator, or nullopt if unknown. Used to resolve spec-declared
// implementation rules to concrete operators.
[[nodiscard]] std::optional<PhysicalOp> physical_op_from_name(const std::string& name) noexcept;

// Every physical operator, for exhaustive iteration (the conformance check walks
// this against the spec so a newly-added op that the spec has not declared is
// caught, not silently emittable). Keep in sync with PhysicalOp.
inline constexpr std::array<PhysicalOp, 28> kAllPhysicalOps = {
    PhysicalOp::SeqScan,        PhysicalOp::Filter,        PhysicalOp::Project,
    PhysicalOp::HashJoin,       PhysicalOp::MergeJoin,     PhysicalOp::NestedLoopJoin,
    PhysicalOp::Sort,           PhysicalOp::FormatConvert, PhysicalOp::Limit,
    PhysicalOp::HashAggregate,  PhysicalOp::StreamingAggregate, PhysicalOp::Window,
    PhysicalOp::HashDistinct,   PhysicalOp::StreamingDistinct, PhysicalOp::UnionAll,
    PhysicalOp::HashSetOp,      PhysicalOp::ValuesScan,
    PhysicalOp::HashSemiJoin,   PhysicalOp::HashAntiJoin,
    PhysicalOp::NestedLoopSemiJoin, PhysicalOp::NestedLoopAntiJoin,
    PhysicalOp::HashGroupingSets, PhysicalOp::RecursiveFixpoint,
    PhysicalOp::WorkingTableScan, PhysicalOp::CreateTableAs,
    PhysicalOp::Insert, PhysicalOp::Update, PhysicalOp::Delete};

// Is `op` a nested-loop family member - the implementations that re-evaluate
// their right input per left row, and so are the only ones that can serve a
// correlated (LATERAL) right side or a join with no equi-key?
[[nodiscard]] bool is_nested_loop(PhysicalOp op) noexcept;
// Does `op` build a hash table on its right input, and therefore need at least
// one equi-key to hash on?
[[nodiscard]] bool needs_equi_key(PhysicalOp op) noexcept;

// May the planner choose WHICH input a hash-family operator builds its table
// from? Only for an INNER or CROSS HashJoin, where the two inputs are
// interchangeable to the algorithm and only the output column order - fixed by
// the group's schema, not by the build side - distinguishes them.
//
// Everything else has a semantically FIXED build side and would be wrong with a
// chosen one. An outer join must keep unmatched rows of a particular input, so it
// has to track matches on that side; a semi or anti join probes the left stream
// against a set built from the right, and swapping them computes a different
// relation; EXCEPT is not symmetric at all. The choice is offered exactly where
// it is sound.
[[nodiscard]] bool is_build_side_choosable(PhysicalOp op, ast::JoinType join_kind) noexcept;

// Does `op` compute several grouping-key combinations at once? The plain
// aggregates compute exactly one, so offering them for a GROUPING SETS node - or
// this one for a plain GROUP BY - would return the wrong ROWS while reporting
// success. Applicability, not cost: the plain aggregates are the cheaper.
[[nodiscard]] bool computes_grouping_sets(PhysicalOp op) noexcept;
// The three aggregate implementations. The grouping-sets question only applies to
// them: asking whether a Filter computes several key combinations is meaningless,
// and answering "no, so it is inapplicable" would make every operator in the
// planner inapplicable to an aggregate group.
[[nodiscard]] bool is_aggregate_family(PhysicalOp op) noexcept;

// The storage format of a relation's rows - the HTAP substrate a subplan reads
// from or produces in. `Any` means "no requirement" (a required property) or
// "unconstrained" (never a derived property).
enum class StorageFormat : std::uint8_t { Any, Row, Column };
[[nodiscard]] const char* storage_format_to_string(StorageFormat f) noexcept;

// Whether data reflects all committed writes. Unlike a sort order or a storage
// format, freshness is a CORRECTNESS constraint, not a preference: a columnar
// replica that lags cannot answer a query that must see the latest writes, at any
// price. And no operator can manufacture it - a Sort fixes order and a
// FormatConvert fixes layout, but nothing turns stale rows into fresh ones. So a
// subplan that cannot provide the required freshness is DISCARDED, never enforced.
enum class Freshness : std::uint8_t {
    Any,    // as a requirement: don't care. Never a derived value.
    Fresh,  // reflects all committed writes
    Stale,  // may lag (a replica); can never satisfy a Fresh requirement
};
[[nodiscard]] const char* freshness_to_string(Freshness f) noexcept;

// LIMIT / OFFSET, as a value. The two are independently optional, so a sentinel
// row count would be indistinguishable from a real one - hence the flags.
struct LimitSpec {
    bool has_limit = false;
    bool has_offset = false;
    std::int64_t limit = -1;   // meaningful when has_limit
    std::int64_t offset = 0;   // meaningful when has_offset

    // How many of `in` rows survive. OFFSET discards from the front first, then
    // LIMIT caps what remains - the order matters, and doing it the other way
    // round over-counts whenever both are present.
    [[nodiscard]] double rows_out(double in) const noexcept {
        double n = in;
        if (has_offset && offset > 0) n = n > static_cast<double>(offset)
                                            ? n - static_cast<double>(offset) : 0.0;
        if (has_limit && limit >= 0) n = n < static_cast<double>(limit)
                                            ? n : static_cast<double>(limit);
        return n;
    }
};

// What the planner needs to know about an Aggregate's grouping beyond the key
// expressions themselves.
//
// `orderable` is the interesting one. A streaming aggregate asks for its input
// pre-sorted on the grouping keys, and a sort requirement is positional - so a
// grouping key that is not a plain column reference (`GROUP BY a + b`) cannot be
// expressed as one. Rather than fail the query, that makes STREAMING inapplicable
// and leaves hashing, which needs no order and can hash any expression. A
// capability the planner lacks costs a plan alternative here, not an answer.
struct GroupingSpec {
    std::uint32_t key_count = 0;  // 0 is a scalar aggregate: exactly one output row
    bool orderable = false;       // every grouping key is a plain column reference
    // GROUPING SETS / ROLLUP / CUBE. `set_count` is how many combinations are
    // computed - an aggregate over N sets emits roughly N times the rows of one,
    // and estimating it as one would understate every operator above it.
    bool has_grouping_sets = false;
    std::uint32_t set_count = 0;
};

// One key of a sort order: a positional column index and its direction, plus
// where NULLs go.
//
// The nulls ordering is carried rather than dropped. SQL's NULLS FIRST / NULLS
// LAST changes the result, so a physical Sort that forgot it would produce rows
// in a different order than the query asked for, silently - the same shape of
// defect as a physical join that forgot whether it was an outer join.
// `nulls_specified` distinguishes "the query said where NULLs go" from "the
// engine's default applies", so an enforcer-created key (which has no opinion)
// is not confused with an explicit request.
struct SortKey {
    std::uint32_t column = 0;
    bool descending = false;
    bool nulls_specified = false;  // the query wrote NULLS FIRST / NULLS LAST
    bool nulls_first = false;      // meaningful only when nulls_specified
};

// The input arity the IR expects of each operator (a SeqScan is a leaf; a Filter
// / Project has one input; a HashJoin has two). The spec declares its own arity;
// conformance asserts the two agree.
[[nodiscard]] std::size_t expected_arity(PhysicalOp op) noexcept;

struct PhysicalNode;
using PhysicalNodePtr = std::unique_ptr<PhysicalNode>;

// Render a join kind for the s-expr writer and for diagnostics. The physical IR
// reuses the LOGICAL join kind rather than defining a parallel enum: a physical
// join implements exactly the relational operator the logical join named, and a
// second vocabulary would only create a mapping that can drift.
[[nodiscard]] const char* join_kind_to_string(ast::JoinType k) noexcept;

// Whether a join kind null-extends unmatched rows of an input. A physical join
// that does not record its kind is indistinguishable from an inner join, which
// is why `PhysicalNode::join_kind` exists.
[[nodiscard]] bool join_null_extends_left(ast::JoinType k) noexcept;
[[nodiscard]] bool join_null_extends_right(ast::JoinType k) noexcept;

// Whether the right input is CORRELATED with the left - a LATERAL derived table,
// whose subtree may reference the left row through an OuterRef. Such a right
// input cannot be evaluated independently of the left, so the only admissible
// implementation is a nested-loop join that re-evaluates it per left row. A hash
// or merge join would build or scan the right side once, standalone, which for a
// correlated right input is not merely slower - it is not executable.
[[nodiscard]] bool join_is_lateral(ast::JoinType k) noexcept;

// Render a set operation, and say whether it de-duplicates. UNION removes
// duplicates; UNION ALL keeps them; INTERSECT ALL and EXCEPT ALL are the multiset
// forms of their un-suffixed counterparts. Carried and rendered rather than
// inferred, for the reason the join kind is: an operator that does not say which
// set operation it computes is indistinguishable from one that computes another.
[[nodiscard]] const char* set_op_to_string(ast::SetOp op) noexcept;
[[nodiscard]] bool set_op_deduplicates(ast::SetOp op) noexcept;

// One equi-join key pair for a HashJoin: positional column indices into the left
// and right input schemas respectively.
struct HashKey {
    std::uint32_t left_index = 0;
    std::uint32_t right_index = 0;

    [[nodiscard]] bool operator==(const HashKey& o) const noexcept {
        return left_index == o.left_index && right_index == o.right_index;
    }
};

// A join's key list. Two columns INLINE, because that is what almost every
// equi-join has and a std::vector reached the heap for the first one. Since
// Increment 3.8a a group-expression OWNS its keys - two candidates in one group
// may join on different keys - so the vector cost one allocation per candidate
// per join: four on the five-group budget query, and multiplied by every
// candidate a larger query enumerates. Longer key lists still work; they spill.
using HashKeyVec = SmallVec<HashKey, 2>;

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
    const Expr* predicate = nullptr;       // Filter: the predicate
    // Join: the conjuncts that are NOT equi-keys and must be re-checked on each
    // candidate row. A LIST, not one expression, because key extraction is
    // per-conjunct: `a.k = b.k AND a.d < b.d` yields one key and one residual,
    // and there is no owning place to synthesize a combined expression (payloads
    // are borrowed from the logical plan).
    std::vector<const Expr*> residual;
    std::vector<const Expr*> projections;  // Project: one borrowed expr per output column
    HashKeyVec hash_keys;        // HashJoin / MergeJoin: equi-join key pairs
    // Join: WHICH relational join this physical operator implements. Without it a
    // physical LEFT JOIN and a physical INNER JOIN are the same node - the plan
    // says "HashJoin" and nothing more, and an executor reading it has no way to
    // know it must null-extend unmatched left rows. The nullability flags in
    // `output` are inherited from the logical schema and describe the RESULT, not
    // the operator's obligation, so they cannot stand in for this.
    ast::JoinType join_kind = ast::JoinType::Inner;
    ast::SetOp set_op = ast::SetOp::Union;               // HashSetOp: which one
    // HashJoin: which input is materialized into the hash table. Recorded on the
    // node because an executor cannot infer it - the plan has to say.
    bool build_right = true;
    std::vector<SortKey> sort_keys;        // Sort: the order it establishes
    StorageFormat scan_format = StorageFormat::Row;      // SeqScan: the table's stored format
    Freshness scan_freshness = Freshness::Fresh;         // SeqScan: does this copy lag?
    StorageFormat target_format = StorageFormat::Any;    // FormatConvert: the format it produces
    LimitSpec limits;                                    // Limit: LIMIT / OFFSET
    // Aggregate: the GROUP BY key expressions and the aggregate calls, borrowed
    // from the logical plan like every other expression payload. The output
    // schema is [keys..., aggregates...], which the logical node already computed.
    std::vector<const Expr*> group_keys;
    std::vector<const Expr*> aggregates;
    // Window: the window-function calls, one per appended output column. Each
    // carries its own OVER clause inside the borrowed expression.
    std::vector<const Expr*> window_functions;
    // ValuesScan: the literal rows, flattened row-major, `values_columns` wide.
    // A FROM-less SELECT is one row of zero columns.
    std::vector<const Expr*> values;
    std::uint32_t values_columns = 0;
    // HashGroupingSets: one bitmask per grouping set over `group_keys`.
    std::vector<std::uint64_t> grouping_sets;
    // DML payload, BORROWED from the logical node on the same contract as every
    // expression payload here: the logical plan outlives the physical plan.
    //
    // These are carried rather than reconstructed because an executor cannot
    // recover them from anything else in the plan. An Update whose SET list was
    // dropped is a physical operator that says "modify these rows" and not what
    // to modify them to - the same shape of defect as a join that did not say it
    // was an outer join.
    const std::vector<plan::Assignment>* assignments = nullptr;   // Update, DO UPDATE
    const std::vector<std::string>* target_columns = nullptr;     // Insert: the column list
    const std::vector<std::string>* conflict_columns = nullptr;   // ON CONFLICT (...)
    plan::ConflictAction conflict_action = plan::ConflictAction::None;

    explicit PhysicalNode(PhysicalOp o) : op(o) {}
};

// Builders - keep the tests and the future lowering (Unit 0.6) terse.
[[nodiscard]] PhysicalNodePtr make_seq_scan(std::string table, Schema output);
[[nodiscard]] PhysicalNodePtr make_filter(PhysicalNodePtr input, const Expr* predicate);
[[nodiscard]] PhysicalNodePtr make_project(PhysicalNodePtr input, Schema output,
                                           std::vector<const Expr*> projections);
[[nodiscard]] PhysicalNodePtr make_hash_join(PhysicalNodePtr left, PhysicalNodePtr right,
                                             HashKeyVec keys, Schema output,
                                             std::vector<const Expr*> residual = {});
[[nodiscard]] PhysicalNodePtr make_merge_join(PhysicalNodePtr left, PhysicalNodePtr right,
                                              HashKeyVec keys, Schema output,
                                              std::vector<const Expr*> residual = {});
[[nodiscard]] PhysicalNodePtr make_nested_loop_join(PhysicalNodePtr left, PhysicalNodePtr right,
                                                    Schema output,
                                                    std::vector<const Expr*> residual = {});
[[nodiscard]] PhysicalNodePtr make_limit(PhysicalNodePtr input, LimitSpec limits);
// Sort is both an operator and an enforcer; FormatConvert is only an enforcer.
[[nodiscard]] PhysicalNodePtr make_sort(PhysicalNodePtr input, std::vector<SortKey> keys);
[[nodiscard]] PhysicalNodePtr make_format_convert(PhysicalNodePtr input, StorageFormat target);

}  // namespace db25::physical
