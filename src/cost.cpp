#include "db25/physical/cost.hpp"

#include "db25/physical/sexpr_read.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <span>
#include <vector>

namespace db25::physical {
namespace {

double atom_double(const SNode& root, const char* key, double dflt) {
    const SNode* kv = root.child(key);
    if (kv == nullptr) return dflt;
    const std::string s = value_of(*kv);
    double v = dflt;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    return res.ec == std::errc{} ? v : dflt;
}

std::uint32_t atom_uint(const SNode& root, const char* key, std::uint32_t dflt) {
    const SNode* kv = root.child(key);
    if (kv == nullptr) return dflt;
    const std::string s = value_of(*kv);
    unsigned long v = dflt;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    return res.ec == std::errc{} ? static_cast<std::uint32_t>(v) : dflt;
}

}  // namespace

CalibrationProfile default_calibration() {
    return CalibrationProfile{};  // the in-struct defaults ARE the built-in profile
}

std::optional<CalibrationProfile> load_lab_calibration(const std::string& path,
                                                       std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open calibration file: " + path; return std::nullopt; }
    std::ostringstream ss;
    ss << in.rdbuf();

    SNode root;
    if (!read_sexpr(ss.str(), root, error)) return std::nullopt;
    if (!root.is_list || root.head() != "calibration") {
        error = "top-level form must be (calibration ...)";
        return std::nullopt;
    }

    CalibrationProfile cal;
    if (const SNode* n = root.child("name")) cal.name = value_of(*n);
    cal.scan_row = atom_double(root, "scan-row", cal.scan_row);
    cal.column_scan_row = atom_double(root, "column-scan-row", cal.column_scan_row);
    cal.filter_row = atom_double(root, "filter-row", cal.filter_row);
    cal.project_row = atom_double(root, "project-row", cal.project_row);
    cal.hash_build_row = atom_double(root, "hash-build-row", cal.hash_build_row);
    cal.hash_probe_row = atom_double(root, "hash-probe-row", cal.hash_probe_row);
    cal.merge_join_row = atom_double(root, "merge-join-row", cal.merge_join_row);
    cal.nested_loop_pair = atom_double(root, "nested-loop-pair", cal.nested_loop_pair);
    cal.sort_row = atom_double(root, "sort-row", cal.sort_row);
    cal.convert_row = atom_double(root, "convert-row", cal.convert_row);
    cal.simd_width = atom_uint(root, "simd-width", cal.simd_width);
    cal.cache_line = atom_uint(root, "cache-line", cal.cache_line);
    return cal;
}

CalibrationProfile calibration_from(CalibrationSource source, const std::string& lab_path,
                                    std::string& error) {
    switch (source) {
        case CalibrationSource::PinnedLab: {
            auto cal = load_lab_calibration(lab_path, error);
            return cal ? *cal : default_calibration();  // error already set on failure
        }
        case CalibrationSource::Live:
        case CalibrationSource::Cached:
            // Documented stub seam: the measuring / caching producers arrive with
            // the execution engine. Until then both resolve to the default so the
            // seam is exercised end-to-end.
            return default_calibration();
    }
    return default_calibration();
}

// ---- representation-independent per-operator formulas ---------------------
// Both the tree cost (cost_of, over a PhysicalNode) and the memo cost (over a
// group-expression, whose inputs are GROUPS not nodes) go through these, so a
// plan is costed identically however it is currently represented. This is the
// "one cost model" invariant (design D3) made structural.

double operator_rows(PhysicalOp op, std::span<const double> input_rows,
                     const std::string& table_name, const CardinalityModel& card,
                     LimitSpec limits, GroupingSpec grouping, ast::SetOp set_op,
                     double values_rows) {
    const auto in = [&](std::size_t i) { return i < input_rows.size() ? input_rows[i] : 0.0; };
    switch (op) {
        case PhysicalOp::SeqScan: {
            const auto it = card.base_rows.find(table_name);
            return it != card.base_rows.end() ? it->second : card.default_base;
        }
        case PhysicalOp::Filter:
            return in(0) * card.filter_selectivity;
        case PhysicalOp::Project:
            return in(0);
        case PhysicalOp::HashJoin:
        case PhysicalOp::MergeJoin:
        case PhysicalOp::NestedLoopJoin:
            // The join ALGORITHM does not change how many rows come out.
            return in(0) * in(1) * card.join_selectivity;
        case PhysicalOp::Sort:
        case PhysicalOp::FormatConvert:
            return in(0);  // both pass every input row through unchanged
        case PhysicalOp::HashGroupingSets: {
            // One group per (set, active-key-combination). Summed over the sets
            // rather than taken once: GROUP BY CUBE(a, b) emits four groups' worth
            // of rows, not one, and every operator above it inherits the estimate.
            // The empty set contributes exactly one row - the grand total.
            double rows = 0.0;
            for (std::uint32_t i = 0; i < grouping.set_count; ++i) rows += 1.0;
            return rows == 0.0 ? 1.0 : in(0) * card.group_selectivity * rows;
        }
        case PhysicalOp::HashAggregate:
        case PhysicalOp::StreamingAggregate:
            // One row per group. With no grouping keys that is exactly one row -
            // `SELECT COUNT(*) FROM t` returns a single row however large t is,
            // and estimating it as a fraction of the input would be wrong rather
            // than merely imprecise. The ALGORITHM does not change the count, so
            // both aggregates answer identically, as both joins do.
            return grouping.key_count == 0 ? 1.0 : in(0) * card.group_selectivity;
        case PhysicalOp::Window:
            return in(0);  // appends columns; never adds or drops a row
        case PhysicalOp::HashSemiJoin:
        case PhysicalOp::NestedLoopSemiJoin:
            // A SUBSET of the left input, and each qualifying left row exactly
            // once however many right rows match - so unlike a join this can never
            // exceed in(0), and multiplying the two cardinalities would be wrong
            // rather than merely pessimistic.
            return in(0) * card.join_selectivity;
        case PhysicalOp::HashAntiJoin:
        case PhysicalOp::NestedLoopAntiJoin:
            // The exact complement of the semi join over the same inputs.
            return in(0) * (1.0 - card.join_selectivity);
        case PhysicalOp::HashDistinct:
        case PhysicalOp::StreamingDistinct:
            return in(0) * card.distinct_selectivity;
        case PhysicalOp::UnionAll:
            return in(0) + in(1);  // every row of both, duplicates kept
        case PhysicalOp::HashSetOp:
            switch (set_op) {
                case ast::SetOp::Union:
                    return (in(0) + in(1)) * card.distinct_selectivity;
                case ast::SetOp::UnionAll:
                    return in(0) + in(1);
                case ast::SetOp::Intersect:
                    // At most the smaller side, then de-duplicated.
                    return (in(0) < in(1) ? in(0) : in(1)) * card.distinct_selectivity;
                case ast::SetOp::IntersectAll:
                    return in(0) < in(1) ? in(0) : in(1);
                case ast::SetOp::Except:
                    return in(0) * card.distinct_selectivity;
                case ast::SetOp::ExceptAll:
                    return in(0);
            }
            return in(0);
        case PhysicalOp::ValuesScan:
            return values_rows;
        case PhysicalOp::Limit:
            return limits.rows_out(in(0));
        case PhysicalOp::RecursiveFixpoint: {
            // The anchor once, then the recursive term once per iteration. The
            // iteration count is an assumption, not a derivation - see
            // CardinalityModel::recursive_iterations.
            const double all = in(0) + card.recursive_iterations * in(1);
            // UNION de-duplicates ACROSS iterations, which is also what makes it
            // terminate on cyclic data; UNION ALL keeps every row.
            return set_op_deduplicates(set_op) ? all * card.distinct_selectivity : all;
        }
        case PhysicalOp::WorkingTableScan:
            return card.working_table_rows;
        case PhysicalOp::CreateTableAs:
            return in(0);  // every row the query produced goes into the new table
        case PhysicalOp::Insert:
        case PhysicalOp::Update:
        case PhysicalOp::Delete:
            // The AFFECTED rows: one per row that arrived. The filtering that
            // decides WHICH rows an UPDATE or DELETE touches happened below, in
            // the Filter that produced this input.
            return in(0);
        case PhysicalOp::Exchange:
            // Moving rows does not create or destroy them. A BROADCAST puts a
            // copy on every node, which is more DATA MOVED but not more rows in
            // the relation - the relation is still its input's rows, and an
            // operator above it that counted them n times would over-estimate
            // everything above THAT. The copies are priced, not counted.
            return in(0);
    }
    return 0.0;
}

double input_evaluations(PhysicalOp op, std::size_t index,
                         const CardinalityModel& card) noexcept {
    // The recursive term runs once per iteration; the anchor runs once.
    if (op == PhysicalOp::RecursiveFixpoint && index == 1) return card.recursive_iterations;
    return 1.0;
}

double operator_cost(PhysicalOp op, std::span<const double> input_rows, double out_rows,
                     const CalibrationProfile& cal, StorageFormat scan_format,
                     bool build_right, std::uint32_t grouping_sets,
                     std::uint32_t exchange_fanout) {
    const auto in = [&](std::size_t i) { return i < input_rows.size() ? input_rows[i] : 0.0; };
    switch (op) {
        case PhysicalOp::SeqScan:
            // The substrate is the cost difference: a columnar read touches only
            // the columns the query needs.
            return out_rows * (scan_format == StorageFormat::Column ? cal.column_scan_row
                                                                    : cal.scan_row);
        case PhysicalOp::Filter:
            return in(0) * cal.filter_row;  // the predicate runs on every INPUT row
        case PhysicalOp::Project:
            return in(0) * cal.project_row;
        case PhysicalOp::HashJoin: {
            // One side is materialized into the hash table, the other streams past
            // it. WHICH is the candidate's own choice rather than a fixed rule:
            // building the smaller side is what makes a hash join cheap, and this
            // is where that shows up.
            const double build = build_right ? in(1) : in(0);
            const double probe = build_right ? in(0) : in(1);
            return build * cal.hash_build_row + probe * cal.hash_probe_row;
        }
        case PhysicalOp::MergeJoin:
            // One linear pass over each already-sorted input - cheaper per row
            // than hashing, which is why it wins when the order comes for free
            // and loses when a Sort has to be enforced to get it.
            return (in(0) + in(1)) * cal.merge_join_row;
        case PhysicalOp::NestedLoopJoin:
            // Every left row against every right row: the condition is evaluated
            // once per PAIR. Quadratic, and the cost model says so - which is why
            // a nested loop only ever wins where it is the only legal option.
            return in(0) * in(1) * cal.nested_loop_pair;
        case PhysicalOp::Sort:
            return in(0) * std::log2(std::max(in(0), 2.0)) * cal.sort_row;  // n*log2(n)
        case PhysicalOp::FormatConvert:
            return in(0) * cal.convert_row;
        case PhysicalOp::HashGroupingSets:
            return in(0) * cal.grouping_set_row * (grouping_sets == 0 ? 1.0 : grouping_sets);
        case PhysicalOp::HashAggregate:
            return in(0) * cal.hash_aggregate_row;   // every INPUT row is hashed
        case PhysicalOp::StreamingAggregate:
            return in(0) * cal.streaming_aggregate_row;
        case PhysicalOp::Window:
            return in(0) * cal.window_row;
        case PhysicalOp::HashSemiJoin:
        case PhysicalOp::HashAntiJoin:
            // Same shape as a hash join: build from the right, probe with the left.
            return in(1) * cal.hash_build_row + in(0) * cal.hash_probe_row;
        case PhysicalOp::NestedLoopSemiJoin:
        case PhysicalOp::NestedLoopAntiJoin:
            return in(0) * in(1) * cal.nested_loop_pair;
        case PhysicalOp::HashDistinct:
            return in(0) * cal.hash_distinct_row;
        case PhysicalOp::StreamingDistinct:
            return in(0) * cal.streaming_distinct_row;
        case PhysicalOp::UnionAll:
            return (in(0) + in(1)) * cal.union_all_row;
        case PhysicalOp::HashSetOp:
            // Build from the right, probe with the left - both sides are touched.
            return (in(0) + in(1)) * cal.set_op_row;
        case PhysicalOp::ValuesScan:
            return out_rows * cal.values_row;
        case PhysicalOp::Limit:
            return out_rows * cal.limit_row;
        case PhysicalOp::RecursiveFixpoint:
            // Its own work is the round trip through the working table, once per
            // row it passes on. The recursive TERM's repeated evaluation is not
            // here - it is charged where it belongs, by multiplying that input's
            // own cost; see input_evaluations.
            return out_rows * cal.recursive_row;
        case PhysicalOp::WorkingTableScan:
            return out_rows * cal.working_table_row;
        case PhysicalOp::CreateTableAs:
        case PhysicalOp::Insert:
        case PhysicalOp::Update:
        case PhysicalOp::Exchange:
            // Priced on the rows that MOVE, which for a broadcast is every row to
            // every node. This is the one cost in the model that is not about the
            // CPU, and the one an operator can pay without doing any work of its
            // own - which is exactly why it has to be charged, or a plan would
            // scatter rows for free.
            //
            // A gather and a repartition each move every row once. The node count
            // is one by default, so on a single-node database all three cost the
            // same and none of them is ever inserted.
            return out_rows * cal.exchange_row * static_cast<double>(exchange_fanout);
        case PhysicalOp::Delete:
            // One coefficient for all four, and that is a statement rather than
            // laziness: this model has no measurement that says a delete is
            // cheaper per row than an insert, and inventing a ratio between them
            // would be a number with nothing behind it. They are durable
            // per-row modifications, priced as such, and separating them is a
            // calibration question for when there is an engine to measure.
            return out_rows * cal.table_write_row;
    }
    return 0.0;
}

double enforcement_cost(const PhysicalProperties& provided, const PhysicalProperties& required,
                        double rows, const CalibrationProfile& cal) {
    if (satisfies(provided, required)) return 0.0;
    // Freshness is not enforceable: there is no operator that turns stale rows
    // into fresh ones. Price it as impossible so such a candidate can never win -
    // the lowering also discards these outright, this is the belt to that braces.
    if (required.freshness == Freshness::Fresh && provided.freshness != Freshness::Fresh) {
        return std::numeric_limits<double>::infinity();
    }
    double c = 0.0;
    PhysicalProperties after = provided;
    // The exchange first, matching the order enforce() inserts them in - so the
    // price a candidate is charged is the price of the plan that gets built.
    if (!distribution_satisfies(provided.distribution, required.distribution)) {
        const double one[1] = {rows};
        const std::uint32_t fanout =
            required.distribution.kind == DistributionKind::Broadcast ? cal.cluster_nodes : 1u;
        c += operator_cost(PhysicalOp::Exchange, one, rows, cal, StorageFormat::Row, true, 0,
                           fanout);
        after.distribution = required.distribution;
        // And the exchange destroys the order, so a sort that was already
        // satisfied has to be paid for now. Charging the sort against the
        // PRE-exchange order would under-price this plan - and the plan that gets
        // BUILT does pay it, so the two would disagree.
        after.sort.clear();
    }
    if (required.format != StorageFormat::Any && required.format != provided.format) {
        const double one[1] = {rows};
        c += operator_cost(PhysicalOp::FormatConvert, one, rows, cal);
        after.format = required.format;
    }
    if (!satisfies(after, required)) {  // order still unmet
        const double one[1] = {rows};
        c += operator_cost(PhysicalOp::Sort, one, rows, cal);
    }
    return c;
}

// ---- tree form ------------------------------------------------------------

double CardinalityModel::rows(const PhysicalNode& node) const {
    std::vector<double> input_rows;
    input_rows.reserve(node.children.size());
    for (const auto& c : node.children) input_rows.push_back(rows(*c));
    return operator_rows(node.op, input_rows, node.table_name, *this, node.limits,
                         GroupingSpec{static_cast<std::uint32_t>(node.group_keys.size()),
                                      node.group_keys.size() == node.sort_keys.size(),
                                      !node.grouping_sets.empty(),
                                      static_cast<std::uint32_t>(node.grouping_sets.size())},
                         node.set_op,
                         node.values_columns != 0
                             ? static_cast<double>(node.values.size() / node.values_columns)
                             : (node.op == PhysicalOp::ValuesScan ? 1.0 : 0.0));
}

double cost_of(const PhysicalNode& node, const CalibrationProfile& cal,
               const CardinalityModel& card) {
    double child_cost = 0.0;
    std::vector<double> input_rows;
    input_rows.reserve(node.children.size());
    for (std::size_t i = 0; i < node.children.size(); ++i) {
        const PhysicalNode& c = *node.children[i];
        // Times the number of times this operator EVALUATES that input - one for
        // everything but a recursive fixpoint's second child.
        child_cost += cost_of(c, cal, card) * input_evaluations(node.op, i, card);
        input_rows.push_back(card.rows(c));
    }
    // A broadcast sends every row to every node; everything else sends it once.
    const std::uint32_t fanout =
        (node.op == PhysicalOp::Exchange &&
         node.target_distribution.kind == DistributionKind::Broadcast)
            ? cal.cluster_nodes
            : 1u;
    return child_cost + operator_cost(node.op, input_rows, card.rows(node), cal,
                                      node.scan_format, node.build_right,
                                      static_cast<std::uint32_t>(node.grouping_sets.size()),
                                      fanout);
}

}  // namespace db25::physical
