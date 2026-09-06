// PARALLELISM IS DECLARED, NOT COSTED - and this file is what keeps that a
// decision rather than a drift.
//
// The planner records a degree of parallelism and each operator's parallel
// behaviour, and the cost model reads NEITHER. That was settled against
// measurements (see "parallelism is declared, not costed" in spec.hpp). Three
// kinds of test here, and only the first is about the feature:
//
//   1. the declarations exist and are complete;
//   2. the cost model really does ignore them - the decision, made checkable;
//   3. the TRIGGERS. The measurements hold for the candidates that exist and the
//      coefficients in force. Each trigger fails when one of those stops being
//      true, so the question is reopened by the change itself rather than by
//      somebody remembering to reopen it.
#include "db25/physical/cost.hpp"
#include "db25/physical/lowering.hpp"
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/spec.hpp"

#include <cstdio>
#include <memory>
#include <utility>
#include <string>
#include <vector>

using namespace db25::physical;
using db25::ast::DataType;
namespace plan = db25::plan;

static int g_failures = 0;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

#ifndef DB25_PHYSICAL_SPEC_DIR
#define DB25_PHYSICAL_SPEC_DIR "."
#endif

static Schema cols(const std::string& t, int n) {
    Schema s;
    for (int i = 0; i < n; ++i)
        s.push_back({t + std::to_string(i), DataType::Integer, false, 0,
                     static_cast<std::uint32_t>(i), t, false});
    return s;
}
static PhysicalNodePtr nd(PhysicalOp o) { return std::make_unique<PhysicalNode>(o); }
static PhysicalNodePtr scan(const std::string& t) {
    auto s = nd(PhysicalOp::SeqScan);
    s->table_name = t;
    s->output = cols(t, 4);
    return s;
}
static PhysicalNodePtr wrap(PhysicalOp o, PhysicalNodePtr in) {
    auto x = nd(o);
    x->output = in->output;
    x->children.push_back(std::move(in));
    return x;
}

// ---- 1. the declarations -------------------------------------------------

static void test_every_operator_declares_how_it_parallelizes() {
    std::printf("test_every_operator_declares_how_it_parallelizes\n");
    std::string err;
    auto spec = load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", err);
    CHECK(spec.has_value());
    if (!spec) return;
    for (const PhysicalOp op : kAllPhysicalOps) {
        const std::string name = physical_op_to_string(op);
        const OperatorSpec* os = spec->find_operator(name);
        CHECK(os != nullptr);
        if (os == nullptr) continue;
        if (os->parallelism.empty()) {
            std::printf("  operator %s does not say how it parallelizes\n", name.c_str());
            ++g_failures;
        }
    }
    // And conformance agrees - an unknown word is a problem, not a shrug.
    CHECK(check_conformance(*spec).empty());
}

// "The shipped spec conforms" passes when the conformance check checks nothing -
// exactly the hole that let nine wrong pipeline labels ship before increment 4.1.
// So the check is exercised NEGATIVELY: a spec that declares a bad parallelism
// must be REPORTED, and neutering the check makes these fail.
static void test_a_bad_parallelism_declaration_is_reported() {
    std::printf("test_a_bad_parallelism_declaration_is_reported\n");
    std::string err;
    const std::string path = std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr";

    const auto mentions_parallelism = [](const std::vector<std::string>& problems) {
        for (const std::string& p : problems)
            if (p.find("parallelism") != std::string::npos) return true;
        return false;
    };

    {   // a word that is not one of the four
        auto spec = load_spec(path, err);
        CHECK(spec.has_value());
        if (!spec) return;
        CHECK(check_conformance(*spec).empty());   // control: it conforms as shipped
        spec->operators.front().parallelism = "sometimes";
        CHECK(mentions_parallelism(check_conformance(*spec)));
    }
    {   // no declaration at all
        auto spec = load_spec(path, err);
        CHECK(spec.has_value());
        if (!spec) return;
        spec->operators.front().parallelism.clear();
        CHECK(mentions_parallelism(check_conformance(*spec)));
    }
}

static void test_the_two_profiles_keep_their_separate_facts() {
    std::printf("test_the_two_profiles_keep_their_separate_facts\n");
    // "This machine has N cores" is a hardware fact and lives in the calibration
    // profile. "This engine uses at most M workers" is a capability and lives in
    // the capability profile. They are different statements from different
    // sources; one number would lose the distinction.
    CalibrationProfile cal = default_calibration();
    CHECK(cal.cores >= 1);
    std::string err;
    auto spec = load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", err);
    CHECK(spec.has_value());
    if (spec) CHECK(spec->profile.max_dop >= 1);
}

// ---- 2. the decision, made checkable -------------------------------------

// The load-bearing test in this file. If somebody wires a degree of parallelism
// into a cost function without reopening the decision, this fails.
static void test_the_cost_model_ignores_parallelism() {
    std::printf("test_the_cost_model_ignores_parallelism\n");
    CardinalityModel card;
    card.base_rows["big"] = 20000000.0;
    card.base_rows["sml"] = 40000.0;

    const auto build = [&] {
        auto j = nd(PhysicalOp::HashJoin);
        j->hash_keys.push_back(HashKey{0, 0});
        j->output = cols("big", 4);
        j->children.push_back(scan("big"));
        j->children.push_back(scan("sml"));
        auto s = wrap(PhysicalOp::Sort, std::move(j));
        s->sort_keys = {SortKey{0}};
        return wrap(PhysicalOp::HashAggregate, std::move(s));
    };

    CalibrationProfile serial = default_calibration();
    serial.cores = 1;
    CalibrationProfile parallel = default_calibration();
    parallel.cores = 64;

    const auto plan = build();
    const double a = cost_of(*plan, serial, card);
    const double b = cost_of(*plan, parallel, card);
    if (a != b) {
        std::printf("  cost changed with the core count (%.0f -> %.0f). The search is\n"
                    "  declared parallelism-blind; if that is being changed on purpose,\n"
                    "  reopen the decision in spec.hpp rather than deleting this test.\n", a, b);
        ++g_failures;
    }
}

// ---- 3. the triggers -----------------------------------------------------

// An operator with ONE candidate has no ranking a cost model could invert, which
// is why Window, the fixpoint, grouping sets and the write path are outside the
// question entirely. A SECOND candidate for any of them puts them back in it.
static void test_single_candidate_operators_still_have_one_candidate() {
    std::printf("test_single_candidate_operators_still_have_one_candidate\n");
    const std::pair<plan::LogicalOp, const char*> single[] = {
        {plan::LogicalOp::Window, "Window"},
        {plan::LogicalOp::RecursiveCTE, "RecursiveCTE"},
        {plan::LogicalOp::Values, "Values"},
        {plan::LogicalOp::Insert, "Insert"},
        {plan::LogicalOp::Update, "Update"},
        {plan::LogicalOp::Delete, "Delete"},
        {plan::LogicalOp::CreateTableAs, "CreateTableAs"},
    };
    for (const auto& [op, name] : single) {
        const auto c = builtin_physical(op);
        if (c.size() != 1) {
            std::printf("  %s now has %zu candidates, not 1. It has acquired a cost\n"
                        "  ranking, so the parallelism-blind analysis no longer covers\n"
                        "  it - see spec.hpp and re-run the break-even measurement.\n",
                        name, c.size());
            ++g_failures;
        }
    }
}

// The analysis covers the candidates that exist. A PARALLEL-AWARE candidate - one
// that deliberately does more work to scale better - is exactly the trade a
// scalar cost cannot express, and adding one voids the analysis. There is no such
// operator today; this fails on the day there is.
static void test_no_parallel_aware_candidate_has_appeared() {
    std::printf("test_no_parallel_aware_candidate_has_appeared\n");
    // Exchange is the one operator that moves rows between workers, and today it
    // exists only to enforce an INTER-NODE distribution. An intra-node variant -
    // a repartition that buys parallelism rather than placement - would be the
    // trigger, and it would need a distribution kind that does not exist yet.
    CHECK(kAllPhysicalOps.size() == 29);
    if (kAllPhysicalOps.size() != 29) {
        std::printf("  the operator set changed. If an operator was added that trades\n"
                    "  work for parallelism (a partitioned join, an intra-node\n"
                    "  repartition, a split-aware scan), the parallelism-blind decision\n"
                    "  in spec.hpp is void and must be re-measured.\n");
    }
}

// The break-even that made the decision: where a Sort must be paid for, the
// hash-based alternative wins by so much that no parallelism difference could
// reverse it. The gap is n*log(n) against n, so it cannot be closed by
// recalibration - but it CAN be closed by someone changing the cost model's
// shape, and this notices.
static void test_sort_based_decisions_are_not_close() {
    std::printf("test_sort_based_decisions_are_not_close\n");
    CalibrationProfile cal = default_calibration();
    CardinalityModel card;
    // Small AND large: the gap is smallest at small cardinalities and still wide.
    for (const double rows : {100.0, 20000000.0}) {
        card.base_rows["t"] = rows;
        auto ha = wrap(PhysicalOp::HashAggregate, scan("t"));
        auto so = wrap(PhysicalOp::Sort, scan("t"));
        so->sort_keys = {SortKey{0}};
        auto sa = wrap(PhysicalOp::StreamingAggregate, std::move(so));
        const double hash = cost_of(*ha, cal, card), sorted = cost_of(*sa, cal, card);
        const double ratio = sorted / hash;
        // 2.0 is well below the measured 3.88x (at 100 rows) and 12.26x (at 20M),
        // and well above the ~1.35x at which a parallelism difference could
        // plausibly flip a decision.
        if (!(ratio > 2.0)) {
            std::printf("  at %.0f rows the sort-based aggregate is only %.2fx the hash\n"
                        "  one. That is close enough for parallelism to decide it, and\n"
                        "  the parallelism-blind decision rests on it NOT being.\n",
                        rows, ratio);
            ++g_failures;
        }
    }
}

int main() {
    test_every_operator_declares_how_it_parallelizes();
    test_a_bad_parallelism_declaration_is_reported();
    test_the_two_profiles_keep_their_separate_facts();
    test_the_cost_model_ignores_parallelism();
    test_single_candidate_operators_still_have_one_candidate();
    test_no_parallel_aware_candidate_has_appeared();
    test_sort_based_decisions_are_not_close();

    if (g_failures == 0) {
        std::printf("parallelism tests: all passed\n");
        return 0;
    }
    std::printf("parallelism tests: %d failure(s)\n", g_failures);
    return 1;
}
