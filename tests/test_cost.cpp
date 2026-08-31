// Unit 0.4: the hardware-parameterized cost model and the pinned-lab
// CalibrationProfile with its pluggable source.
//
// Cost is a pure function of (plan, calibration, cardinality). The tests pin the
// lab profile (so the number is reproducible), assert a concrete cost for the
// Increment-0 shape, and then check the properties that matter: determinism,
// monotonicity in cardinality and in the coefficients (falsifiability - a change
// to either input must move the cost), and that the source seam resolves
// PinnedLab from the file while Live/Cached return the default.
#include "db25/physical/cost.hpp"
#include "db25/physical/physical_plan.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace db25::physical;

static int g_failures = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("  FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

static bool approx(double a, double b) { return std::fabs(a - b) < 1e-6; }

#ifndef DB25_PHYSICAL_SPEC_DIR
#define DB25_PHYSICAL_SPEC_DIR "."
#endif
static std::string lab_path() {
    return std::string(DB25_PHYSICAL_SPEC_DIR) + "/calibration.lab.sexpr";
}

// The Increment-0 shape; cost ignores expression contents, so payloads are null.
static PhysicalNodePtr build_plan() {
    auto scan_a = make_seq_scan("a", {});
    auto scan_b = make_seq_scan("b", {});
    auto join = make_hash_join(std::move(scan_a), std::move(scan_b), {{0, 0}}, {});
    auto filter = make_filter(std::move(join), nullptr);
    return make_project(std::move(filter), {}, {});
}

static CardinalityModel base_card() {
    CardinalityModel c;
    c.base_rows["a"] = 1000.0;
    c.base_rows["b"] = 500.0;
    return c;
}

static void test_lab_profile_loads() {
    std::printf("test_lab_profile_loads\n");
    std::string error;
    auto cal = load_lab_calibration(lab_path(), error);
    CHECK(cal.has_value());
    if (!cal) { std::printf("  load error: %s\n", error.c_str()); return; }
    CHECK(cal->name == "lab");
    CHECK(approx(cal->scan_row, 1.0));
    CHECK(approx(cal->hash_build_row, 1.2));
    CHECK(cal->simd_width == 8);
    CHECK(cal->cache_line == 64);
}

static void test_source_seam() {
    std::printf("test_source_seam\n");
    std::string error;
    const CalibrationProfile pinned =
        calibration_from(CalibrationSource::PinnedLab, lab_path(), error);
    CHECK(pinned.name == "lab");  // read from the file

    const CalibrationProfile live =
        calibration_from(CalibrationSource::Live, lab_path(), error);
    CHECK(live.name == "default");  // documented stub -> default

    const CalibrationProfile cached =
        calibration_from(CalibrationSource::Cached, lab_path(), error);
    CHECK(cached.name == "default");
}

static void test_cardinality_estimates() {
    std::printf("test_cardinality_estimates\n");
    const CardinalityModel card = base_card();
    auto scan = make_seq_scan("a", {});
    CHECK(approx(card.rows(*scan), 1000.0));

    auto unknown = make_seq_scan("nope", {});
    CHECK(approx(card.rows(*unknown), card.default_base));

    // Filter shrinks; join multiplies (1000 * 500 * 0.1); project passes through.
    auto plan = build_plan();  // Project<-Filter<-HashJoin<-(a,b)
    CHECK(approx(card.rows(*plan), 1000.0 * 500.0 * 0.1 * 0.1));  // *join_sel *filter_sel
}

static void test_cost_is_deterministic_and_concrete() {
    std::printf("test_cost_is_deterministic_and_concrete\n");
    std::string error;
    const CalibrationProfile lab =
        calibration_from(CalibrationSource::PinnedLab, lab_path(), error);
    const CardinalityModel card = base_card();
    auto plan = build_plan();

    const double c1 = cost_of(*plan, lab, card);
    const double c2 = cost_of(*plan, lab, card);
    CHECK(approx(c1, c2));  // deterministic

    // Hand-computed against the lab coefficients:
    //   scans: 1000*1.0 + 500*1.0                    = 1500
    //   join:  build 500*1.2 + probe 1000*0.8        = 1400
    //   filter: input rows (join out 50000) * 0.5    = 25000
    //   project: input rows (filter out 5000) * 0.3  = 1500
    CHECK(approx(c1, 1500.0 + 1400.0 + 25000.0 + 1500.0));  // 29400
}

static void test_monotonic_in_cardinality_and_coefficients() {
    std::printf("test_monotonic_in_cardinality_and_coefficients\n");
    std::string error;
    const CalibrationProfile lab =
        calibration_from(CalibrationSource::PinnedLab, lab_path(), error);
    auto plan = build_plan();

    const double base = cost_of(*plan, lab, base_card());

    // More base rows -> higher cost.
    CardinalityModel bigger = base_card();
    bigger.base_rows["a"] = 2000.0;
    CHECK(cost_of(*plan, lab, bigger) > base);

    // A costlier scan coefficient -> higher cost (a coefficient change moves it).
    CalibrationProfile pricier = lab;
    pricier.scan_row = lab.scan_row * 2.0;
    CHECK(cost_of(*plan, pricier, base_card()) > base);
}

int main() {
    test_lab_profile_loads();
    test_source_seam();
    test_cardinality_estimates();
    test_cost_is_deterministic_and_concrete();
    test_monotonic_in_cardinality_and_coefficients();

    if (g_failures == 0) {
        std::printf("cost/calibration tests: all passed\n");
        return 0;
    }
    std::printf("cost/calibration tests: %d failure(s)\n", g_failures);
    return 1;
}
