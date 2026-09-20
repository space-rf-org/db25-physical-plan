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
#include <cstdint>
#include <cstdio>
#include <span>
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

    // Filter shrinks; project passes through; the join on one key emits the
    // LARGER side, not the product - and takes its key count off the node, which
    // is the wiring a JoinSpec{} here would silently drop.
    auto plan = build_plan();  // Project<-Filter<-HashJoin<-(a,b)
    CHECK(approx(card.rows(*plan), 1000.0 * 0.1));  // max(1000, 500) * filter_sel
}

// ---- G13: a join reads its predicate, not its inputs' product -------------
// The rule was |L||R|s, and its error compounded with every further join: the
// six-way key join below returns 98000 rows and was estimated at 2000000000
// times that. What these pin is the SHAPE of the replacement - the larger side,
// narrowed once per conjunct beyond the one that contains the join - because a
// mistuned constant was never what was wrong.
static double join_rows(double l, double r, std::uint32_t keys, std::uint32_t residual,
                        const CardinalityModel& card,
                        PhysicalOp op = PhysicalOp::HashJoin) {
    const double in[2] = {l, r};
    return operator_rows(op, std::span<const double>{in, 2}, "", card, LimitSpec{},
                         GroupingSpec{}, db25::ast::SetOp::Union, 0.0, JoinSpec{keys, residual});
}

static void test_join_cardinality() {
    std::printf("test_join_cardinality\n");
    const CardinalityModel card;

    // One equi-key: the larger side. 20000 customers to 50000 orders is 50000
    // rows, which is what a foreign key MEANS - not 100 million.
    CHECK(approx(join_rows(20000, 50000, 1, 0, card), 50000.0));
    CHECK(join_rows(20000, 50000, 1, 0, card) < 20000.0 * 50000.0);
    // And it does not matter which side is written first.
    CHECK(approx(join_rows(50000, 20000, 1, 0, card), 50000.0));

    // No equi-key and no condition at all is a CROSS JOIN, and a cross join
    // really does return the product - undiscounted, because there is no
    // predicate for a selectivity to stand for.
    CHECK(approx(join_rows(1000, 500, 0, 0, card), 500000.0));
    // A theta join is that, narrowed once per condition.
    CHECK(approx(join_rows(1000, 500, 0, 2, card),
                 500000.0 * card.join_selectivity * card.join_selectivity));

    // Conjuncts beyond the first narrow, whether they are keys or residuals.
    CHECK(approx(join_rows(20000, 50000, 2, 0, card), 50000.0 * card.join_selectivity));
    CHECK(approx(join_rows(20000, 50000, 1, 1, card), 50000.0 * card.join_selectivity));
    CHECK(approx(join_rows(20000, 50000, 3, 0, card),
                 50000.0 * card.join_selectivity * card.join_selectivity));

    // Never more than the cartesian product. An empty input means no rows out -
    // max() alone would answer 50000, a join against nothing producing rows.
    CHECK(approx(join_rows(0, 50000, 1, 0, card), 0.0));
    CHECK(approx(join_rows(3, 2, 1, 0, card), 3.0));
    CHECK(approx(join_rows(1, 1, 1, 0, card), 1.0));

    // The ALGORITHM does not change the count. If it did, the search would be
    // choosing between the three on the strength of an estimate rather than a
    // cost - and the cheapest estimate is not the cheapest plan.
    for (const std::uint32_t keys : {0u, 1u, 2u}) {
        const double h = join_rows(20000, 50000, keys, 1, card, PhysicalOp::HashJoin);
        CHECK(approx(join_rows(20000, 50000, keys, 1, card, PhysicalOp::MergeJoin), h));
        CHECK(approx(join_rows(20000, 50000, keys, 1, card, PhysicalOp::NestedLoopJoin), h));
    }

    // Growing either input can never LOWER the estimate. A property rather than
    // a number, and the one that stops a future statistics source turning the
    // estimate non-monotone without anything noticing.
    double prev = join_rows(10, 50000, 1, 0, card);
    for (double l = 100; l <= 1000000; l *= 10) {
        const double now = join_rows(l, 50000, 1, 0, card);
        CHECK(now >= prev);
        prev = now;
    }
}

static void test_join_chain_does_not_explode() {
    std::printf("test_join_chain_does_not_explode\n");
    const CardinalityModel card;
    // The benchmark's six-way join, every predicate a foreign key onto a primary
    // key: cust-orders-items-prod, plus emp and dept hanging off cust.
    //   actual 98000   PostgreSQL's estimate 99491   DB25 before this: 2e18
    const double sides[] = {50000.0, 100000.0, 2000.0, 20000.0, 50.0};
    double rows = 20000.0;  // cust
    for (const double r : sides) rows = join_rows(rows, r, 1, 0, card);
    CHECK(approx(rows, 100000.0));

    // The error this replaces was not a constant, it was a RATE: each further
    // join multiplied the estimate by another table's cardinality. So the test
    // that matters is that adding joins does not grow the estimate at all when
    // the tables they add are smaller than what is already there.
    double narrow = 100000.0;
    for (int i = 0; i < 20; ++i) narrow = join_rows(narrow, 50.0, 1, 0, card);
    CHECK(approx(narrow, 100000.0));
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
    //   filter: input rows (join out 1000) * 0.5     = 500
    //   project: input rows (filter out 100) * 0.3   = 30
    CHECK(approx(c1, 1500.0 + 1400.0 + 500.0 + 30.0));  // 3430
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
    test_join_cardinality();
    test_join_chain_does_not_explode();
    test_cost_is_deterministic_and_concrete();
    test_monotonic_in_cardinality_and_coefficients();

    if (g_failures == 0) {
        std::printf("cost/calibration tests: all passed\n");
        return 0;
    }
    std::printf("cost/calibration tests: %d failure(s)\n", g_failures);
    return 1;
}
