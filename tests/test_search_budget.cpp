// THE SEARCH BUDGET IS SAFE AT ANY SETTING - which is what lets its value be a
// tuning knob rather than a number somebody has to defend.
//
// Open question 5 asked how the fast-path threshold is DERIVED. The honest
// answer is that it cannot be, not from anything this repo owns: a planning-time
// budget is a ratio against execution time, and nothing here estimates execution
// time. Picking a number and pinning goldens to it would manufacture exactly the
// unfalsifiable constant the project spends its effort removing.
//
// So the number stops being load-bearing instead. There are not two planners to
// choose between: the guard makes the SAME search greedy, skipping the second
// route through each goal (pushing a requirement into an input rather than
// enforcing it on the output). Same IR, same rules, same cost model - only less
// exploration. If MORE budget never yields a WORSE plan, then any setting trades
// planning effort for plan quality monotonically: a badly chosen budget makes a
// plan suboptimal, never wrong, and never surprising.
//
// That is a property, it needs nothing that does not exist, and it is what this
// file enforces.
#include "db25/physical/cost.hpp"
#include "db25/physical/lowering.hpp"
#include "db25/physical/physical_plan.hpp"

#include "db25/plan/logical_plan.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace db25::physical;
namespace plan = db25::plan;
using db25::ast::BinaryOp;
using db25::ast::DataType;

static int g_failures = 0;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

static plan::ExprPtr col(std::uint32_t i) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    e->input_index = i;
    e->type = DataType::Integer;
    return e;
}
static plan::ExprPtr eq(std::uint32_t l, std::uint32_t r) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::BinaryOp);
    e->bin_op = BinaryOp::Equal;
    e->type = DataType::Boolean;
    e->children.push_back(col(l));
    e->children.push_back(col(r));
    return e;
}
static plan::LogicalNodePtr scan(const std::string& t) {
    auto s = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    s->table_name = t;
    s->output = {{t + "0", DataType::Integer, false, 0, 0, t, false},
                 {t + "1", DataType::Integer, true, 0, 1, t, false}};
    return s;
}
// A chain of equi-joins over `rels`, optionally under a Sort.
static plan::LogicalNodePtr chain(const std::vector<std::string>& rels, bool sorted) {
    auto acc = scan(rels[0]);
    for (std::size_t i = 1; i < rels.size(); ++i) {
        auto r = scan(rels[i]);
        auto j = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
        j->join_type = db25::ast::JoinType::Inner;
        j->output = acc->output;
        for (const auto& c : r->output) j->output.push_back(c);
        j->predicate = eq(static_cast<std::uint32_t>(acc->output.size() - 2),
                          static_cast<std::uint32_t>(acc->output.size()));
        j->add_child(std::move(acc));
        j->add_child(std::move(r));
        acc = std::move(j);
    }
    if (!sorted) return acc;
    auto s = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Sort);
    s->output = acc->output;
    plan::SortKeyIR k;
    k.expr = col(0);
    s->sort_keys.push_back(std::move(k));
    s->add_child(std::move(acc));
    return s;
}

struct Shape {
    const char* name;
    std::vector<std::string> rels;
    bool sorted;
    bool require_order;   // ask the planner for a sorted OUTPUT too
};

static const std::vector<Shape>& shapes() {
    static const std::vector<Shape> v{
        {"6-way chain", {"ra", "rb", "rc", "rd", "re", "rf"}, false, false},
        {"6-way chain, ordered output", {"ra", "rb", "rc", "rd", "re", "rf"}, false, true},
        {"6-way chain under a Sort", {"ra", "rb", "rc", "rd", "re", "rf"}, true, false},
        {"5-way chain, ordered output", {"ra", "rb", "rc", "rd", "re"}, false, true},
        {"4-way chain", {"ra", "rb", "rc", "rd"}, false, false},
    };
    return v;
}

static void seed(CardinalityModel& card) {
    const double sizes[] = {8000000, 40000, 900000, 12000, 3000000, 250};
    const char* names[] = {"ra", "rb", "rc", "rd", "re", "rf"};
    for (int i = 0; i < 6; ++i) card.base_rows[names[i]] = sizes[i];
}

// The property. Sweep the budget upward; the chosen plan must never get worse.
static void test_more_budget_never_yields_a_worse_plan() {
    std::printf("test_more_budget_never_yields_a_worse_plan\n");
    CalibrationProfile cal = default_calibration();
    CardinalityModel card;
    seed(card);

    for (const Shape& s : shapes()) {
        double prev = -1.0;
        for (std::uint32_t b = 1; b <= 9; ++b) {
            const auto q = chain(s.rels, s.sorted);
            LoweringContext ctx;
            ctx.calibration = &cal;
            ctx.cardinality = &card;
            ctx.max_join_count_override = b;
            if (s.require_order) ctx.required_output.sort = {SortKey{0, false, false, false}};
            const LoweringResult r = lower(*q, ctx);
            CHECK(r.ok && r.plan != nullptr);
            if (!r.ok || !r.plan) break;
            const double c = cost_of(*r.plan, cal, card);
            // A strict tolerance: this is the same cost model on both sides, so
            // equality is exact when the plan is the same one.
            if (prev >= 0.0 && c > prev) {
                std::printf("  %s: budget %u costs %.0f, but budget %u cost %.0f -\n"
                            "  MORE search produced a WORSE plan, so the budget is not a\n"
                            "  safe knob and its value becomes a correctness decision.\n",
                            s.name, b, c, b - 1, prev);
                ++g_failures;
            }
            prev = c;
        }
    }
}

// Without this the property above is vacuous: if the guard never engages for any
// shape, monotonicity holds because nothing ever changed. The guard's effect is
// invisible in every other counter, so this is the only way to know it ran.
static void test_the_guard_actually_engages_and_is_observable() {
    std::printf("test_the_guard_actually_engages_and_is_observable\n");
    CalibrationProfile cal = default_calibration();
    CardinalityModel card;
    seed(card);
    const Shape& s = shapes()[1];  // 6-way chain with an ordered output

    const auto run = [&](std::uint32_t budget) {
        const auto q = chain(s.rels, s.sorted);
        LoweringContext ctx;
        ctx.calibration = &cal;
        ctx.cardinality = &card;
        ctx.max_join_count_override = budget;
        ctx.required_output.sort = {SortKey{0, false, false, false}};
        return lower(*q, ctx);
    };

    const LoweringResult tight = run(1);   // 5 joins > 1, guard engages
    const LoweringResult loose = run(9);   // 5 joins < 9, guard off

    CHECK(tight.budget_guard_engaged);
    CHECK(!loose.budget_guard_engaged);

    // The guard skipped work, and the amount is REPORTED. Before this counter
    // existed an engaged guard and a disengaged one were indistinguishable from
    // outside the planner: same candidates, same goals, same cost.
    CHECK(tight.pushdown_explorations_skipped > 0);
    CHECK(tight.pushdown_explorations == 0);
    CHECK(loose.pushdown_explorations > 0);
    CHECK(loose.pushdown_explorations_skipped == 0);
    // The two routes are the same population, just explored or not.
    CHECK(tight.pushdown_explorations_skipped == loose.pushdown_explorations);

    // And the counters the result already had cannot show any of this, which is
    // exactly why the two above had to be added.
    CHECK(tight.candidates_considered == loose.candidates_considered);
    CHECK(tight.optimization_goals == loose.optimization_goals);

    std::printf("    guard engaged: %zu push-down explorations skipped "
                "(candidates and goals unchanged at %zu / %zu)\n",
                tight.pushdown_explorations_skipped, tight.candidates_considered,
                tight.optimization_goals);
}

// The guard must be a pure LOSS of search, never a change of answer to something
// the full search would have rejected: the plan it returns has to be one the
// unbounded search also considers legal. Costing both with the same model and
// requiring the guarded plan to be no cheaper is how that is checked - a guarded
// plan cheaper than the unbounded one would mean the full search overlooked it.
static void test_the_guard_never_beats_the_full_search() {
    std::printf("test_the_guard_never_beats_the_full_search\n");
    CalibrationProfile cal = default_calibration();
    CardinalityModel card;
    seed(card);
    for (const Shape& s : shapes()) {
        const auto build = [&](std::uint32_t b) {
            const auto q = chain(s.rels, s.sorted);
            LoweringContext ctx;
            ctx.calibration = &cal;
            ctx.cardinality = &card;
            ctx.max_join_count_override = b;
            if (s.require_order) ctx.required_output.sort = {SortKey{0, false, false, false}};
            LoweringResult r = lower(*q, ctx);
            return r.ok && r.plan ? cost_of(*r.plan, cal, card) : -1.0;
        };
        const double guarded = build(1), full = build(9);
        CHECK(guarded >= 0.0 && full >= 0.0);
        if (guarded >= 0.0 && full >= 0.0 && guarded < full) {
            std::printf("  %s: the GUARDED search found a cheaper plan (%.0f) than the\n"
                        "  full one (%.0f). The full search is missing a candidate it\n"
                        "  should have considered.\n", s.name, guarded, full);
            ++g_failures;
        }
    }
}

int main() {
    test_more_budget_never_yields_a_worse_plan();
    test_the_guard_actually_engages_and_is_observable();
    test_the_guard_never_beats_the_full_search();

    if (g_failures == 0) {
        std::printf("search budget tests: all passed\n");
        return 0;
    }
    std::printf("search budget tests: %d failure(s)\n", g_failures);
    return 1;
}
