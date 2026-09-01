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
    }
    return 0.0;
}

double operator_cost(PhysicalOp op, std::span<const double> input_rows, double out_rows,
                     const CalibrationProfile& cal, StorageFormat scan_format) {
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
        case PhysicalOp::HashJoin:
            // input 0 probes, input 1 builds.
            return in(1) * cal.hash_build_row + in(0) * cal.hash_probe_row;
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
        case PhysicalOp::HashAggregate:
            return in(0) * cal.hash_aggregate_row;   // every INPUT row is hashed
        case PhysicalOp::StreamingAggregate:
            return in(0) * cal.streaming_aggregate_row;
        case PhysicalOp::Window:
            return in(0) * cal.window_row;
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
                                      node.group_keys.size() == node.sort_keys.size()},
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
    for (const auto& c : node.children) {
        child_cost += cost_of(*c, cal, card);
        input_rows.push_back(card.rows(*c));
    }
    return child_cost + operator_cost(node.op, input_rows, card.rows(node), cal, node.scan_format);
}

}  // namespace db25::physical
