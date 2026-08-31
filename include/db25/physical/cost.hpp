#pragma once
// The hardware-parameterized cost model (design D3) and the CalibrationProfile it
// reads (design D10).
//
// Cost is a pure, deterministic function of the plan, the calibration
// coefficients, and a cardinality model. The coefficients are an INPUT, not baked
// into the code: goldens pin a lab profile and stay deterministic, while
// production resolves a live- or cache-sourced profile through the same seam.
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/properties.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace db25::physical {

// Hardware-calibrated cost coefficients. Costs are abstract units; only their
// ratios matter to plan choice. Calibrated per host (see CalibrationSource).
struct CalibrationProfile {
    std::string name = "default";
    // Per-row work.
    double scan_row = 1.0;         // read one row from a ROW-format base table
    double column_scan_row = 0.35; // read one row from a COLUMN-format base table:
                                   // a scan touches only the columns it needs, which
                                   // is the whole reason a columnar substrate exists
    double filter_row = 0.5;      // evaluate a predicate on one input row
    double project_row = 0.3;     // evaluate the projection list on one input row
    double hash_build_row = 1.2;  // insert one build-side row into the hash table
    double hash_probe_row = 0.8;  // probe the hash table with one probe-side row
    double merge_join_row = 0.4;  // merge one row from an already-sorted input
    double sort_row = 1.0;        // per-row coefficient of an n*log2(n) sort
    double convert_row = 0.6;     // convert one row between storage formats
    // Hardware facts (informational today; richer costing consumes them later).
    std::uint32_t simd_width = 8;
    std::uint32_t cache_line = 64;
};

// Where a CalibrationProfile comes from (design D10). Live measures the host;
// PinnedLab loads a checked-in profile (the deterministic source of truth for
// goldens and lab work); Cached reuses a prior host-fingerprinted profile.
// Live and Cached are stubs today that delegate to the default/pinned profile -
// the SEAM is real and every consumer goes through it; the measuring and caching
// producers arrive with the execution engine.
enum class CalibrationSource { Live, PinnedLab, Cached };

// A reasonable built-in default profile (the stub Live/Cached currently return,
// and the fallback when a lab file cannot be read).
[[nodiscard]] CalibrationProfile default_calibration();

// Load a lab profile from an s-expr file. Returns nullopt and writes `error` on
// failure (missing file, malformed s-expr, wrong head).
[[nodiscard]] std::optional<CalibrationProfile> load_lab_calibration(const std::string& path,
                                                                     std::string& error);

// Resolve a profile from a source. PinnedLab reads `lab_path` (falling back to
// default() with `error` set if it cannot be read); Live and Cached currently
// return default() - the documented stub seam.
[[nodiscard]] CalibrationProfile calibration_from(CalibrationSource source,
                                                  const std::string& lab_path,
                                                  std::string& error);

// A minimal cardinality model for Increment 0. Base-table row counts are seeded
// by the caller (catalog-driven statistics arrive later); simple default
// selectivities carry the estimate up the tree. Deterministic.
struct CardinalityModel {
    std::unordered_map<std::string, double> base_rows;  // table name -> row count
    double default_base = 1000.0;      // an unseeded base table
    double filter_selectivity = 0.1;   // default WHERE selectivity
    double join_selectivity = 0.1;     // default equi-join selectivity, per key

    // Estimated output rows of the plan rooted at `node`.
    [[nodiscard]] double rows(const PhysicalNode& node) const;
};

// ---- representation-independent per-operator formulas ---------------------
// The single place each operator's cardinality and local cost are defined. Both
// the tree form below and the memo form (a group-expression, whose inputs are
// GROUPS rather than nodes) go through these, so a plan is costed identically
// however it is currently represented - the "one cost model" invariant (D3) made
// structural rather than merely intended. `input_rows` are the estimated output
// rows of each input, in operand order.
[[nodiscard]] double operator_rows(PhysicalOp op, const std::vector<double>& input_rows,
                                   const std::string& table_name, const CardinalityModel& card);
[[nodiscard]] double operator_cost(PhysicalOp op, const std::vector<double>& input_rows,
                                   double out_rows, const CalibrationProfile& cal,
                                   StorageFormat scan_format = StorageFormat::Row);

// What it would cost to make `rows` rows satisfying `provided` also satisfy
// `required` - i.e. the enforcers that would have to be inserted. Zero when the
// requirement is already met. Charging this to a candidate is what keeps the
// choice honest: a MergeJoin that needs its inputs sorted must pay for the Sorts
// it will cause, or it would look cheap and then cost more than the plan chosen.
[[nodiscard]] double enforcement_cost(const PhysicalProperties& provided,
                                      const PhysicalProperties& required, double rows,
                                      const CalibrationProfile& cal);

// Estimated cost of the plan rooted at `node`, computed bottom-up from the
// calibration coefficients and the cardinality model. Deterministic given its
// inputs; monotonic in cardinality.
[[nodiscard]] double cost_of(const PhysicalNode& node, const CalibrationProfile& cal,
                             const CardinalityModel& card);

}  // namespace db25::physical
