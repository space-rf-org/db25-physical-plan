// Increment 3.8b: associative join reordering.
//
// (A join B) join C and A join (B join C) return the same rows and can cost very
// differently, so which one the planner builds should be a cost decision - not a
// transcription of the order the FROM clause happened to list its tables in.
//
// What makes this hard in DB25's IR is that a ColumnRef is a POSITIONAL index into
// `child0.output ++ child1.output`, so moving a join changes what every index
// above it means. The way through (join_order.hpp) is that an in-order traversal
// gives every subtree a CONTIGUOUS range of the region's columns, in the region's
// own order - so re-association needs no permutation anywhere and translating an
// index is one subtraction.
//
// These tests are about that arithmetic being right as much as about the search
// finding the cheaper tree. A reordering that picks a cheaper plan and joins on
// the wrong columns is not a faster planner, it is a wrong answer.
#include "db25/physical/cost.hpp"
#include "db25/physical/join_order.hpp"
#include "db25/physical/lowering.hpp"
#include "db25/physical/sexpr.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

using db25::ast::BinaryOp;
using db25::ast::DataType;
using db25::ast::JoinType;
using db25::physical::CardinalityModel;
using db25::physical::lower;
using db25::physical::LoweringContext;
using db25::physical::LoweringResult;
using db25::physical::physical_to_sexpr;
namespace plan = db25::plan;

static int g_failures = 0;
#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            std::printf("  FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                   \
        }                                                                   \
    } while (0)

static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

static std::size_t count(const std::string& hay, const std::string& needle) {
    std::size_t n = 0;
    for (std::size_t i = hay.find(needle); i != std::string::npos;
         i = hay.find(needle, i + needle.size())) {
        ++n;
    }
    return n;
}

// ---- builders -------------------------------------------------------------

static plan::ExprPtr col(std::uint32_t i, DataType t = DataType::Integer) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    e->input_index = i;
    e->type = t;
    return e;
}
static plan::ExprPtr binop(BinaryOp op, plan::ExprPtr l, plan::ExprPtr r) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::BinaryOp);
    e->bin_op = op;
    e->children.push_back(std::move(l));
    e->children.push_back(std::move(r));
    return e;
}
static plan::ExprPtr int_lit(std::int64_t v) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::Literal);
    e->type = DataType::Integer;
    e->value.value = v;
    return e;
}

// Two columns per table, named so the rendered schema says which table a column
// came from: `a` has (a0, a1), `b` has (b0, b1), `c` has (c0, c1).
static plan::LogicalNodePtr scan(const std::string& name) {
    auto s = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    s->table_name = name;
    s->output = {{name + "0", DataType::Integer, false, 0, 0, name, false},
                 {name + "1", DataType::Integer, true, 0, 0, name, false}};
    return s;
}

// A join whose output is its children's schemas concatenated - which is what the
// logical binder builds for an inner join, and what the whole of the index
// arithmetic below rests on.
static plan::LogicalNodePtr join(plan::LogicalNodePtr l, plan::LogicalNodePtr r,
                                 plan::ExprPtr pred, JoinType jt = JoinType::Inner) {
    auto j = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
    j->join_type = jt;
    j->output = l->output;
    for (const auto& c : r->output) j->output.push_back(c);
    j->predicate = std::move(pred);
    j->add_child(std::move(l));
    j->add_child(std::move(r));
    return j;
}

// The chain `((a JOIN b ON a.a0 = b.b1) JOIN c ON b.b0 = c.c1)`, written LEFT
// DEEP - the shape a FROM clause listing a, b, c produces. Region columns:
// a = [0,2), b = [2,4), c = [4,6).
static plan::LogicalNodePtr chain_abc() {
    auto ab = join(scan("a"), scan("b"), binop(BinaryOp::Equal, col(0), col(3)));
    return join(std::move(ab), scan("c"), binop(BinaryOp::Equal, col(2), col(5)));
}

static std::string render(const plan::LogicalNode& q, const CardinalityModel& card,
                          bool reorder, LoweringResult* out = nullptr) {
    LoweringContext ctx;
    ctx.cardinality = &card;
    ctx.reorder_joins = reorder;
    LoweringResult r = lower(q, ctx);
    if (!r.ok) {
        std::printf("  lowering failed: %s\n", r.error.c_str());
        return {};
    }
    std::string s = physical_to_sexpr(*r.plan);
    if (out != nullptr) *out = std::move(r);
    return s;
}

// The plan's total cost under the same model that chose it. Comparing the two
// lowerings' costs is the only comparison that means anything: "reordering picked
// a different plan" is not a claim, "reordering picked a cheaper one" is.
static double plan_cost(const plan::LogicalNode& q, const CardinalityModel& card,
                        bool reorder) {
    LoweringContext ctx;
    ctx.cardinality = &card;
    ctx.reorder_joins = reorder;
    const LoweringResult r = lower(q, ctx);
    if (!r.ok || !r.plan) return -1.0;
    return db25::physical::cost_of(*r.plan, db25::physical::default_calibration(), card);
}

// ---- tests ----------------------------------------------------------------

// The headline. ONE query text, two cardinality models, two different join trees.
// If the planner produced the same shape both times it has not reordered - it has
// transcribed.
static void test_a_three_way_join_is_re_associated_by_cost() {
    std::printf("test_a_three_way_join_is_re_associated_by_cost\n");

    // `a` enormous, `b` and `c` tiny: joining b and c FIRST keeps the big table
    // out of every intermediate, so the cheap tree is a JOIN (b JOIN c) - the one
    // the query did not write.
    CardinalityModel big_a;
    big_a.base_rows["a"] = 1000000.0;
    big_a.base_rows["b"] = 100.0;
    big_a.base_rows["c"] = 100.0;

    // `c` enormous instead: now the written left-deep tree is already the right
    // one, and the planner must NOT move it just because it can.
    CardinalityModel big_c;
    big_c.base_rows["a"] = 100.0;
    big_c.base_rows["b"] = 100.0;
    big_c.base_rows["c"] = 1000000.0;

    const auto q = chain_abc();
    LoweringResult ra;
    LoweringResult rc;
    const std::string sa = render(*q, big_a, true, &ra);
    const std::string sc = render(*q, big_c, true, &rc);
    CHECK(!sa.empty());
    CHECK(!sc.empty());
    CHECK(sa != sc);
    // Both lowerings saw a region; that is what makes the difference above a
    // choice rather than an accident of costing.
    CHECK(ra.join_regions_enumerated == 1);
    CHECK(rc.join_regions_enumerated == 1);

    // WHICH tree, read off the keys rather than off the indentation. The join
    // over (b, c) is the one keyed `(L#0 R#1)` - b.b0 against c.c1 - and it can
    // only exist in the re-associated tree, because b and c are not adjacent
    // inputs of any join in the written one.
    CHECK(contains(sa, "keys=[(L#0 R#1)]"));
    // The written tree's lower join is (a, b) keyed a.a0 (#0) against b.b1
    // (#1 within b), and its upper join is b.b0 (#2 of a++b) against c.c1.
    CHECK(contains(sc, "keys=[(L#0 R#1)]"));
    CHECK(contains(sc, "keys=[(L#2 R#1)]"));
    CHECK(!contains(sa, "keys=[(L#2 R#1)]"));

    // BOTH joins are keyed, in both trees, and neither carries a residual. A
    // conjunct translated into the wrong numbering does not usually give a wrong
    // answer - it stops looking like a cross-side equality, so it is DEMOTED to a
    // residual and re-checked per row. The plan is then merely quadratic where it
    // should be linear, which no assertion about shape or cost reliably catches.
    // Counting the keys does.
    CHECK(count(sa, "keys=[") == 2);
    CHECK(count(sc, "keys=[") == 2);
    CHECK(count(sa, "residual=") == 0);
    CHECK(count(sc, "residual=") == 0);
}

// Reordering is a cost decision, so the only thing it is allowed to do is find a
// plan the cost model likes at least as much. This is the falsifiable form of the
// claim, and it does not depend on which tree either run happens to pick.
static void test_reordering_never_costs_more_than_the_written_order() {
    std::printf("test_reordering_never_costs_more_than_the_written_order\n");
    const auto q = chain_abc();
    const double shapes[][3] = {{1000000.0, 100.0, 100.0},
                                {100.0, 100.0, 1000000.0},
                                {100.0, 1000000.0, 100.0},
                                {1000.0, 1000.0, 1000.0},
                                {10.0, 500000.0, 20.0}};
    for (const auto& s : shapes) {
        CardinalityModel card;
        card.base_rows["a"] = s[0];
        card.base_rows["b"] = s[1];
        card.base_rows["c"] = s[2];
        const double written = plan_cost(*q, card, false);
        const double reordered = plan_cost(*q, card, true);
        CHECK(written > 0.0);
        CHECK(reordered > 0.0);
        // Strictly: never worse. A tolerance would let a real regression through,
        // and there is nothing to be tolerant of - the written tree is one of the
        // trees the DP enumerates, so it is always available to be chosen.
        CHECK(reordered <= written);
    }
}

// And on the shape that motivates the whole increment, it is not merely no worse:
// it is much better. A test that only asserts "<=" would pass on a planner that
// reorders nothing at all.
static void test_reordering_actually_wins_where_it_should() {
    std::printf("test_reordering_actually_wins_where_it_should\n");
    const auto q = chain_abc();
    CardinalityModel card;
    card.base_rows["a"] = 1000000.0;
    card.base_rows["b"] = 100.0;
    card.base_rows["c"] = 100.0;
    const double written = plan_cost(*q, card, false);
    const double reordered = plan_cost(*q, card, true);
    CHECK(written > 0.0 && reordered > 0.0);
    CHECK(reordered < written);
}

// The invariant the PARENT of a region depends on: whatever the DP does inside,
// the region's output columns are the same columns in the same order. A parent
// holds positional indices into that schema, so a reordering that changed it
// would silently re-address everything above.
static void test_reordering_does_not_change_the_output_schema() {
    std::printf("test_reordering_does_not_change_the_output_schema\n");
    CardinalityModel card;
    card.base_rows["a"] = 1000000.0;
    card.base_rows["b"] = 100.0;
    card.base_rows["c"] = 100.0;
    const auto q = chain_abc();
    const auto top_out = [](const std::string& s) -> std::string {
        const std::size_t j = s.find("out=[");
        if (j == std::string::npos) return "<no output schema>";
        const std::size_t end = s.find(']', j);
        return end == std::string::npos ? "<unterminated>" : s.substr(j, end - j);
    };
    const std::string written = render(*q, card, false);
    const std::string reordered = render(*q, card, true);
    CHECK(!written.empty() && !reordered.empty());
    CHECK(written != reordered);  // it really did reorder
    CHECK(top_out(written) == top_out(reordered));
    CHECK(contains(top_out(written), "a0"));
    CHECK(contains(top_out(written), "c1"));
}

// A projection ABOVE the region reads its columns positionally. This is the same
// invariant as above, checked end to end rather than on the rendered schema: the
// projected expressions are untouched by the reordering, so if the region's
// columns had moved the query would now return different columns.
static void test_a_projection_above_the_region_still_reads_the_same_columns() {
    std::printf("test_a_projection_above_the_region_still_reads_the_same_columns\n");
    CardinalityModel card;
    card.base_rows["a"] = 1000000.0;
    card.base_rows["b"] = 100.0;
    card.base_rows["c"] = 100.0;
    const auto make = [] {
        auto p = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Project);
        auto region = chain_abc();
        // a.a1 (#1) and c.c0 (#4) of the region's six columns.
        p->output = {region->output[1], region->output[4]};
        p->exprs.push_back(col(1));
        p->exprs.push_back(col(4));
        p->add_child(std::move(region));
        return p;
    };
    const auto q = make();
    const std::string written = render(*q, card, false);
    const std::string reordered = render(*q, card, true);
    CHECK(!written.empty() && !reordered.empty());
    CHECK(written != reordered);
    // The Project is identical in both - same expressions, same schema.
    const auto project_of = [](const std::string& s) {
        const std::size_t i = s.find("(Project");
        if (i == std::string::npos) return std::string("<no project>");
        const std::size_t end = s.find('\n', i);
        return end == std::string::npos ? s.substr(i) : s.substr(i, end - i);
    };
    CHECK(project_of(written) == project_of(reordered));
    CHECK(contains(project_of(written), "a1"));
    CHECK(contains(project_of(written), "c0"));
}

// A region whose joins cannot ALL carry a connecting predicate is left exactly as
// written. Not squeamishness about cross products: the cardinality model charges
// one flat join_selectivity per join, so it estimates a cartesian product at a
// fraction of |A|x|B| - and a search that enumerated those would be choosing
// between plans on the strength of that error.
static void test_a_region_with_a_cross_product_is_left_as_written() {
    std::printf("test_a_region_with_a_cross_product_is_left_as_written\n");
    // a, b, c with only a.a0 = c.c0: every tree over the leaves in this order has
    // a cross product in it somewhere.
    //
    // `a.a1 > 10` is here for a specific reason. Without it, no conjunct is ever
    // PLACED on the (a, b) join, so the both-sides test below is never reached and
    // a planner that had lost it would still pass - the placement rule alone would
    // be doing the work. With it, one conjunct is placed there and names only the
    // left side, which is exactly the case the connectedness rule exists for: a
    // predicate that constrains an input is not a predicate that joins anything.
    auto ab = join(scan("a"), scan("b"), nullptr, JoinType::Cross);
    auto q = join(std::move(ab), scan("c"),
                  binop(BinaryOp::And, binop(BinaryOp::Equal, col(0), col(4)),
                        binop(BinaryOp::GreaterThan, col(1), int_lit(10))));
    CardinalityModel card;
    card.base_rows["a"] = 1000.0;
    card.base_rows["b"] = 1000.0;
    card.base_rows["c"] = 1000.0;
    LoweringContext ctx;
    ctx.cardinality = &card;
    const LoweringResult r = lower(*q, ctx);
    CHECK(r.ok);
    CHECK(r.join_regions_enumerated == 0);
    // And it still plans - a region the DP declines is a region lowered the
    // ordinary way, not a query the planner refuses.
    LoweringContext off = ctx;
    off.reorder_joins = false;
    const LoweringResult ro = lower(*q, off);
    CHECK(ro.ok);
    CHECK(r.plan && ro.plan);
    if (r.plan && ro.plan) CHECK(physical_to_sexpr(*r.plan) == physical_to_sexpr(*ro.plan));
}

// An OUTER join is not associative - moving it changes which rows are
// null-extended - so it ENDS the region and becomes one of its leaves. With only
// two leaves left there is nothing to re-associate, which is the correct answer
// rather than a missed opportunity.
static void test_an_outer_join_ends_the_region() {
    std::printf("test_an_outer_join_ends_the_region\n");
    auto ab = join(scan("a"), scan("b"), binop(BinaryOp::Equal, col(0), col(3)),
                   JoinType::Left);
    auto q = join(std::move(ab), scan("c"), binop(BinaryOp::Equal, col(2), col(5)));
    CardinalityModel card;
    card.base_rows["a"] = 1000000.0;
    card.base_rows["b"] = 100.0;
    card.base_rows["c"] = 100.0;
    LoweringContext ctx;
    ctx.cardinality = &card;
    const LoweringResult r = lower(*q, ctx);
    CHECK(r.ok);
    CHECK(r.join_regions_enumerated == 0);
    if (r.plan) CHECK(contains(physical_to_sexpr(*r.plan), "kind=left"));
}

// A LATERAL join's right input READS its left, so it cannot move at all. Like the
// outer join it ends the region; unlike the outer join, moving it would not merely
// change which rows are null-extended, it would leave a correlated reference with
// nothing to correlate to.
static void test_a_lateral_join_is_not_reordered() {
    std::printf("test_a_lateral_join_is_not_reordered\n");
    auto ab = join(scan("a"), scan("b"), binop(BinaryOp::Equal, col(0), col(3)),
                   JoinType::Lateral);
    auto q = join(std::move(ab), scan("c"), binop(BinaryOp::Equal, col(2), col(5)));
    CardinalityModel card;
    card.base_rows["a"] = 1000000.0;
    card.base_rows["b"] = 100.0;
    card.base_rows["c"] = 100.0;
    LoweringContext ctx;
    ctx.cardinality = &card;
    const LoweringResult r = lower(*q, ctx);
    CHECK(r.ok);
    CHECK(r.join_regions_enumerated == 0);
}

// A conjunct that names ONE leaf is not a join predicate; it constrains an input.
// It has to be applied exactly once however the tree is associated, and the rule
// that does it - a conjunct goes to the LOWEST join whose subtree contains all its
// leaves, and a single leaf is not a join - is what this checks. Applying it twice
// is harmless for `>`; applying it zero times returns rows that should not exist.
static void test_a_single_leaf_conjunct_is_applied_exactly_once() {
    std::printf("test_a_single_leaf_conjunct_is_applied_exactly_once\n");
    auto ab = join(scan("a"), scan("b"), binop(BinaryOp::Equal, col(0), col(3)));
    // b.b1 > 10, written on the TOP join where its indices are region indices.
    auto top = binop(BinaryOp::And, binop(BinaryOp::Equal, col(2), col(5)),
                     binop(BinaryOp::GreaterThan, col(3), int_lit(10)));
    auto q = join(std::move(ab), scan("c"), std::move(top));
    CardinalityModel card;
    card.base_rows["a"] = 1000000.0;
    card.base_rows["b"] = 100.0;
    card.base_rows["c"] = 100.0;
    LoweringContext ctx;
    ctx.cardinality = &card;
    const LoweringResult r = lower(*q, ctx);
    CHECK(r.ok);
    CHECK(r.join_regions_enumerated == 1);
    const std::string s = r.plan ? physical_to_sexpr(*r.plan) : std::string{};
    // Exactly one residual in the whole plan.
    std::size_t n = 0;
    for (std::size_t i = s.find("residual="); i != std::string::npos;
         i = s.find("residual=", i + 1)) {
        ++n;
    }
    CHECK(n == 1);
}

// The region analysis itself, checked directly rather than through a plan: the
// leaves come out in traversal order with the offsets that make a subtree's
// columns contiguous, and each conjunct records the leaves it touches.
static void test_the_region_analysis_reports_contiguous_leaf_ranges() {
    std::printf("test_the_region_analysis_reports_contiguous_leaf_ranges\n");
    const auto q = chain_abc();
    db25::physical::JoinRegion region;
    CHECK(db25::physical::collect_join_region(*q, region));
    CHECK(region.leaf_count() == 3);
    CHECK(region.leaf_offset[0] == 0);
    CHECK(region.leaf_offset[1] == 2);
    CHECK(region.leaf_offset[2] == 4);
    CHECK(region.end_of(3) == 6);
    CHECK(region.conjuncts.size() == 2);
    // Collected children-first, so the lower join's conjunct comes first.
    CHECK(region.conjuncts[0].leaf_mask == 0b011);
    CHECK(region.conjuncts[1].leaf_mask == 0b110);
    // Placement: the (b, c) conjunct belongs to the join over [1,3), and NOT to
    // the top join when that join's right child already contains both leaves.
    CHECK(db25::physical::placed_here(region.conjuncts[1], 1, 2, 3));
    CHECK(!db25::physical::placed_here(region.conjuncts[1], 0, 1, 3));
    CHECK(db25::physical::placed_here(region.conjuncts[0], 0, 1, 3));

    // Translation into the (b, c) join's own numbering: b.b0 is region column 2
    // and becomes column 0, c.c1 is region column 5 and becomes column 3.
    db25::physical::LoweringArena arena;
    const plan::Expr* t =
        db25::physical::translate_conjunct(region.conjuncts[1], region, 1, arena);
    CHECK(t != nullptr);
    CHECK(t != region.conjuncts[1].expr);  // it moved, so it was cloned
    CHECK(arena.exprs.size() == 1);
    if (t != nullptr && t->children.size() == 2) {
        CHECK(t->children[0]->input_index == 0);
        CHECK(t->children[1]->input_index == 3);
    }
    // And a conjunct that did NOT move is BORROWED, not copied - so a plan that
    // keeps the written association allocates nothing here.
    const plan::Expr* same =
        db25::physical::translate_conjunct(region.conjuncts[1], region, 0, arena);
    CHECK(same == region.conjuncts[1].expr);
    CHECK(arena.exprs.size() == 1);
}

// A two-table join has exactly one association, so there is nothing for the DP to
// do - and it must not pay to find that out. This is the case every query hits.
static void test_a_two_table_join_is_not_a_region() {
    std::printf("test_a_two_table_join_is_not_a_region\n");
    auto q = join(scan("a"), scan("b"), binop(BinaryOp::Equal, col(0), col(3)));
    db25::physical::JoinRegion region;
    CHECK(!db25::physical::collect_join_region(*q, region));
    CHECK(region.leaves.empty());  // nothing was built
    CardinalityModel card;
    LoweringContext ctx;
    ctx.cardinality = &card;
    const LoweringResult r = lower(*q, ctx);
    CHECK(r.ok);
    CHECK(r.join_regions_enumerated == 0);
}

// Four leaves: the DP finds the balanced tree (a b)(c d), which no amount of
// re-associating a left-deep chain one join at a time would reach in one step.
// It does NOT find (a c)(b d) - that permutes leaves, which changes the output
// column order and is deliberately out of scope.
static void test_four_leaves_reach_a_balanced_tree() {
    std::printf("test_four_leaves_reach_a_balanced_tree\n");
    // ((( a JOIN b ) JOIN c ) JOIN d), chained a-b, b-c, c-d.
    auto ab = join(scan("a"), scan("b"), binop(BinaryOp::Equal, col(0), col(3)));
    auto abc = join(std::move(ab), scan("c"), binop(BinaryOp::Equal, col(2), col(5)));
    auto q = join(std::move(abc), scan("d"), binop(BinaryOp::Equal, col(4), col(7)));
    CardinalityModel card;
    card.base_rows["a"] = 1000000.0;
    card.base_rows["b"] = 10.0;
    card.base_rows["c"] = 10.0;
    card.base_rows["d"] = 1000000.0;
    LoweringResult r;
    const std::string s = render(*q, card, true, &r);
    CHECK(!s.empty());
    CHECK(r.join_regions_enumerated == 1);
    CHECK(plan_cost(*q, card, true) <= plan_cost(*q, card, false));
    // The balanced tree's (b, c) join is keyed b.b0 (#0 of b) against c.c1 (#1 of
    // c); no left-deep tree over these leaves has b and c as its two inputs.
    CHECK(contains(s, "keys=[(L#0 R#1)]"));
}

// Pruning and dedup are SEARCH optimizations: they must never change the plan.
// Reordering is where that claim gets hard - branch-and-bound now prunes across
// candidates that read different inputs, and dedup now has repeated subtrees to
// find, which is the workload it was built for and never had. So it is checked on
// the query that has both rather than only on the ones that predate them.
static void test_search_options_never_change_a_reordered_plan() {
    std::printf("test_search_options_never_change_a_reordered_plan\n");
    const auto q = chain_abc();
    const double shapes[][3] = {{1000000.0, 100.0, 100.0},
                                {100.0, 100.0, 1000000.0},
                                {1000.0, 1000.0, 1000.0}};
    for (const auto& sh : shapes) {
        CardinalityModel card;
        card.base_rows["a"] = sh[0];
        card.base_rows["b"] = sh[1];
        card.base_rows["c"] = sh[2];
        const auto plan_with = [&](bool prune, bool dedup) {
            LoweringContext ctx;
            ctx.cardinality = &card;
            ctx.prune = prune;
            ctx.dedup = dedup;
            const LoweringResult r = lower(*q, ctx);
            CHECK(r.ok);
            return r.plan ? physical_to_sexpr(*r.plan) : std::string{};
        };
        const std::string base = plan_with(true, false);
        CHECK(!base.empty());
        CHECK(plan_with(false, false) == base);
        CHECK(plan_with(true, true) == base);
        CHECK(plan_with(false, true) == base);
    }
}

// The invariant every index in join_order.cpp rests on: a region's columns ARE
// its leaves' columns concatenated. An inner or cross join's output is exactly
// that, so this holds for anything the binder builds - and it is checked rather
// than assumed, because a region that did not satisfy it would be re-addressed by
// arithmetic that silently pointed at other columns. Here it is violated
// deliberately, which is the only way to reach the check.
static void test_a_join_whose_output_is_not_its_inputs_concatenated_is_not_a_region() {
    std::printf("test_a_join_whose_output_is_not_its_inputs_concatenated_is_not_a_region\n");
    auto q = chain_abc();
    CHECK(q->output.size() == 6);
    db25::physical::JoinRegion accepted;
    CHECK(db25::physical::collect_join_region(*q, accepted));
    // Drop a column from the region root's schema, so it no longer matches what
    // its leaves produce.
    q->output.pop_back();
    db25::physical::JoinRegion rejected;
    CHECK(!db25::physical::collect_join_region(*q, rejected));
}

int main() {
    test_a_three_way_join_is_re_associated_by_cost();
    test_search_options_never_change_a_reordered_plan();
    test_reordering_never_costs_more_than_the_written_order();
    test_reordering_actually_wins_where_it_should();
    test_reordering_does_not_change_the_output_schema();
    test_a_projection_above_the_region_still_reads_the_same_columns();
    test_a_region_with_a_cross_product_is_left_as_written();
    test_an_outer_join_ends_the_region();
    test_a_lateral_join_is_not_reordered();
    test_a_single_leaf_conjunct_is_applied_exactly_once();
    test_the_region_analysis_reports_contiguous_leaf_ranges();
    test_a_two_table_join_is_not_a_region();
    test_a_join_whose_output_is_not_its_inputs_concatenated_is_not_a_region();
    test_four_leaves_reach_a_balanced_tree();

    if (g_failures == 0) {
        std::printf("join order tests: all passed\n");
        return 0;
    }
    std::printf("join order tests: %d failure(s)\n", g_failures);
    return 1;
}
