// Unit 0.6: logical -> physical lowering (the payload that ties the units
// together). Builds a real db25-logical-plan tree for the Increment-0 query
//   SELECT a.x, b.y FROM a JOIN b ON a.id = b.id WHERE a.x > 10
// and lowers it through the memo, then checks: the physical plan has the expected
// structure, the join predicate lowered to a hash key, the memo held one group
// per logical node, the spec's implementation rules and the built-in mapping
// agree, a non-equi join keeps a residual, and an unsupported operator is an
// honest error.
#include "db25/physical/lowering.hpp"
#include "db25/physical/sexpr.hpp"
#include "db25/physical/spec.hpp"
#include "db25/physical/memo.hpp"
#include "db25/physical/properties.hpp"
#include "db25/physical/storage_catalog.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <utility>

using db25::ast::BinaryOp;
using db25::ast::DataType;
using db25::physical::lower;
using db25::physical::LoweringContext;
using db25::physical::LoweringResult;
using db25::physical::physical_to_sexpr;
using db25::physical::PhysicalOp;
using db25::physical::SortKey;
using db25::physical::StorageFormat;
using db25::physical::Freshness;
using db25::physical::StorageCatalog;
using db25::physical::FormatAvailability;
using db25::physical::PhysicalProperties;
using db25::physical::satisfies;
using db25::physical::derive;
namespace plan = db25::plan;

static int g_failures = 0;
#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("  FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

#ifndef DB25_PHYSICAL_SPEC_DIR
#define DB25_PHYSICAL_SPEC_DIR "."
#endif

// ---- logical-plan builders ------------------------------------------------
static plan::ExprPtr col(std::uint32_t idx, DataType t) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    e->input_index = idx;
    e->type = t;
    return e;
}
static plan::ExprPtr int_lit(std::int64_t v) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::Literal);
    e->type = DataType::Integer;
    e->value.value = v;
    return e;
}
static plan::ExprPtr binop(BinaryOp op, plan::ExprPtr l, plan::ExprPtr r) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::BinaryOp);
    e->bin_op = op;
    e->children.push_back(std::move(l));
    e->children.push_back(std::move(r));
    return e;
}
static plan::Schema a_schema() {
    return {{"id", DataType::Integer, false}, {"x", DataType::Integer, true}};
}
static plan::Schema b_schema() {
    return {{"id", DataType::Integer, false}, {"y", DataType::VarChar, true}};
}
static plan::Schema join_schema() {
    return {{"id", DataType::Integer, false},
            {"x", DataType::Integer, true},
            {"id", DataType::Integer, false},
            {"y", DataType::VarChar, true}};
}

// Build the Increment-0 query as a logical plan. `join_pred` lets a test swap in
// a non-equi predicate; nullptr keeps the equi-join a.id = b.id.
static plan::LogicalNodePtr build_logical(plan::ExprPtr join_pred = nullptr,
                                          db25::ast::JoinType kind = db25::ast::JoinType::Inner) {
    auto scan_a = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan_a->table_name = "a";
    scan_a->output = a_schema();

    auto scan_b = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan_b->table_name = "b";
    scan_b->output = b_schema();

    auto join = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
    join->join_type = kind;
    join->output = join_schema();
    join->predicate = join_pred ? std::move(join_pred)
                                : binop(BinaryOp::Equal, col(0, DataType::Integer),
                                        col(2, DataType::Integer));  // a.id = b.id
    join->add_child(std::move(scan_a));
    join->add_child(std::move(scan_b));

    auto filter = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
    filter->output = join_schema();
    filter->predicate = binop(BinaryOp::GreaterThan, col(1, DataType::Integer), int_lit(10));
    filter->add_child(std::move(join));

    auto project = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Project);
    project->output = {{"x", DataType::Integer, true}, {"y", DataType::VarChar, true}};
    project->exprs.push_back(col(1, DataType::Integer));
    project->exprs.push_back(col(3, DataType::VarChar));
    project->add_child(std::move(filter));
    return project;
}

static void test_lowers_the_increment0_query() {
    std::printf("test_lowers_the_increment0_query\n");
    auto logical = build_logical();
    const LoweringResult r = lower(*logical);  // built-in mapping
    CHECK(r.ok);
    CHECK(r.plan != nullptr);
    CHECK(r.memo_groups == 5);  // Project, Filter, HashJoin, SeqScan, SeqScan
    if (!r.plan) return;

    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "(Project "));
    CHECK(contains(s, "(Filter "));
    CHECK(contains(s, "(HashJoin kind=inner keys=[(L#0 R#0)]"));  // a.id=b.id -> key pair
    CHECK(contains(s, "(SeqScan table=a"));
    CHECK(contains(s, "(SeqScan table=b"));
    CHECK(contains(s, "exprs=[(col #1) (col #3)]"));
    CHECK(contains(s, "pred=("));   // the filter predicate is carried
    CHECK(contains(s, "(lit 10)"));
}

static void test_spec_rules_and_builtin_agree() {
    std::printf("test_spec_rules_and_builtin_agree\n");
    std::string error;
    auto spec =
        db25::physical::load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr",
                                  error);
    CHECK(spec.has_value());
    if (!spec) { std::printf("  spec load error: %s\n", error.c_str()); return; }

    auto logical = build_logical();
    const std::string builtin = physical_to_sexpr(*lower(*logical).plan);

    LoweringContext ctx;
    ctx.spec = &*spec;
    auto via_spec = lower(*logical, ctx);
    CHECK(via_spec.ok);
    if (via_spec.plan) CHECK(physical_to_sexpr(*via_spec.plan) == builtin);
}

static void test_non_equi_join_keeps_residual() {
    std::printf("test_non_equi_join_keeps_residual\n");
    // a.x > b.y-ish: a GreaterThan between two columns is not a hash key.
    auto pred = binop(BinaryOp::GreaterThan, col(1, DataType::Integer), col(2, DataType::Integer));
    auto logical = build_logical(std::move(pred));
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "keys=[]"));                       // no equi key extracted
    CHECK(contains(s, "residual=[(> (col #1) (col #2))]"));  // kept, as a conjunct list
}

// The defect this pins: key extraction used to be ALL-OR-NOTHING - a conjunction
// walk in which one non-equi conjunct discarded every key already found. So the
// ordinary shape `a.id = b.id AND a.x > b.y` lowered to a KEYLESS join carrying
// the whole predicate as a residual: a cross product with a filter, quadratic
// where it should be linear, and with MergeJoin made inapplicable (no keys left
// to merge on), which put the whole Increment-1 merge-join capability out of
// reach for any join with a compound predicate.
static void test_equi_key_survives_an_extra_conjunct() {
    std::printf("test_equi_key_survives_an_extra_conjunct\n");
    auto pred = binop(BinaryOp::And,
                      binop(BinaryOp::Equal, col(0, DataType::Integer),
                            col(2, DataType::Integer)),           // a.id = b.id  -> key
                      binop(BinaryOp::GreaterThan, col(1, DataType::Integer),
                            col(3, DataType::VarChar)));          // a.x > b.y    -> residual
    auto logical = build_logical(std::move(pred));
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "keys=[(L#0 R#0)]"));                  // the key IS extracted
    CHECK(contains(s, "residual=[(> (col #1) (col #3))]"));  // only the non-key conjunct
    CHECK(!contains(s, "keys=[]"));                          // not degraded to a cross product
    CHECK(!contains(s, "(AND "));                            // the conjunction was split, not kept whole
}

// Two keys and a residual from a three-way conjunction, and - the subtle case -
// a SAME-side equality is not a join key: it constrains one input rather than
// relating the two, so it belongs in the residual.
static void test_multiple_keys_and_same_side_equality() {
    std::printf("test_multiple_keys_and_same_side_equality\n");
    auto pred = binop(BinaryOp::And,
                      binop(BinaryOp::And,
                            binop(BinaryOp::Equal, col(0, DataType::Integer),
                                  col(2, DataType::Integer)),     // a.id = b.id   -> key
                            binop(BinaryOp::Equal, col(1, DataType::Integer),
                                  col(3, DataType::VarChar))),    // a.x  = b.y    -> key
                      binop(BinaryOp::Equal, col(0, DataType::Integer),
                            col(1, DataType::Integer)));          // a.id = a.x    -> residual
    auto logical = build_logical(std::move(pred));
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "keys=[(L#0 R#0) (L#1 R#1)]"));        // both cross-side equalities
    CHECK(contains(s, "residual=[(= (col #0) (col #1))]"));  // the same-side one only
}

// ---- Unit 2.2: property-directed (top-down) optimization -------------------

// The motivating case from the design doc. Unconstrained, the join group takes
// the cheapest candidate - a HashJoin. Asked for its output IN KEY ORDER, the
// same group answers with a MergeJoin instead: the merge join's output order
// satisfies the requirement natively, so the alternative is a HashJoin plus a
// Sort of the (much larger) JOIN OUTPUT. Bottom-up single-winner search could
// not even ask this question - it settled the join before the ORDER BY was
// considered, and then sorted whatever came out.
static void test_required_order_selects_the_merge_join() {
    std::printf("test_required_order_selects_the_merge_join\n");
    std::string error;
    auto spec = db25::physical::load_spec(
        std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    CHECK(spec.has_value());
    if (!spec) return;

    auto logical = build_logical();  // a.id = b.id, with a Filter and Project above

    LoweringContext any;
    any.spec = &*spec;
    const LoweringResult ra = lower(*logical, any);
    CHECK(ra.ok);
    if (!ra.plan) return;
    const std::string sa = physical_to_sexpr(*ra.plan);
    CHECK(contains(sa, "HashJoin"));
    CHECK(!contains(sa, "MergeJoin"));

    LoweringContext ordered;
    ordered.spec = &*spec;
    ordered.required_output.sort = {SortKey{0, false}};
    const LoweringResult rb = lower(*logical, ordered);
    CHECK(rb.ok);
    if (!rb.plan) return;
    const std::string sb = physical_to_sexpr(*rb.plan);
    CHECK(contains(sb, "MergeJoin"));   // a DIFFERENT plan for the same query
    CHECK(!contains(sb, "HashJoin"));

    // And the plan really does satisfy what was asked of it.
    CHECK(satisfies(derive(*rb.plan), ordered.required_output));

    // A group optimized for more than one requirement is the whole point; goals
    // exceeding groups is the direct evidence of it.
    CHECK(rb.optimization_goals > rb.memo_groups);
}

// The root has no parent, and every enforcer is placed by a parent onto its
// child - so a plan whose top operator does not itself provide the required
// output would be returned UNENFORCED, reported ok while not satisfying the
// requirement it was optimized for. (The same shape of defect the audit found in
// enforce(): a function reporting success on a result that does not meet its own
// contract.) lower() enforces at the root and then CHECKS the postcondition.
static void test_root_enforcer_is_inserted() {
    std::printf("test_root_enforcer_is_inserted\n");
    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();
    auto filter = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
    filter->output = a_schema();
    filter->predicate = binop(BinaryOp::GreaterThan, col(1, DataType::Integer), int_lit(10));
    filter->add_child(std::move(scan));

    LoweringContext ctx;
    ctx.required_output.sort = {SortKey{0, false}};
    const LoweringResult r = lower(*filter, ctx);
    CHECK(r.ok);
    if (!r.plan) return;
    CHECK(satisfies(derive(*r.plan), ctx.required_output));

    // Nothing below provides order, so a Sort must appear - ABOVE the Filter, not
    // below it. That placement is the enforce-vs-push-down comparison coming out
    // the right way: sorting below would sort the rows the Filter is about to
    // discard (1000 of them, against 100 surviving it), so pushing down loses on
    // cost here even though it is available. The search compares; it does not
    // assume earlier is better.
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(s.rfind("(Sort", 0) == 0);            // the Sort is the ROOT
    CHECK(contains(s, "(Sort by=[#0 asc]"));
    CHECK(s.find("(Sort") < s.find("(Filter"));  // above, not below
}

// An output requirement nothing can satisfy is an honest failure, not a plan
// that quietly ignores it. Freshness is the one property no enforcer can
// establish, so it is the case that must fail rather than be patched up.
static void test_unsatisfiable_required_output_fails_honestly() {
    std::printf("test_unsatisfiable_required_output_fails_honestly\n");
    StorageCatalog cat;
    cat.formats["a"] = {FormatAvailability{StorageFormat::Row, Freshness::Stale}};

    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();

    LoweringContext ctx;
    ctx.storage = &cat;
    ctx.required_output.freshness = Freshness::Fresh;
    const LoweringResult r = lower(*scan, ctx);
    CHECK(!r.ok);
    CHECK(r.plan == nullptr);
    CHECK(!r.error.empty());
}

// ---- Unit 2.3: branch-and-bound -------------------------------------------

// THE gate for this unit. Pruning is a pure search optimization: it may only ever
// make the search cheaper, never make it answer differently. So every case is
// lowered twice - pruned and exhaustive - and the rendered plans must be
// IDENTICAL. A pruning bug that discards the optimum shows up here and nowhere
// else, because a cheaper-but-wrong plan still looks like a plan.
static void test_pruning_never_changes_the_plan() {
    std::printf("test_pruning_never_changes_the_plan\n");
    std::string error;
    auto spec = db25::physical::load_spec(
        std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    CHECK(spec.has_value());
    if (!spec) return;

    StorageCatalog both;   // a table with two substrates widens the search
    both.formats["a"] = {FormatAvailability{StorageFormat::Row},
                         FormatAvailability{StorageFormat::Column}};
    both.formats["b"] = {FormatAvailability{StorageFormat::Row},
                         FormatAvailability{StorageFormat::Column}};

    // Requirements worth exploring: none, an order, a format, and both at once.
    std::vector<PhysicalProperties> goals(4);
    goals[1].sort = {SortKey{0, false}};
    goals[2].format = StorageFormat::Row;
    goals[3].sort = {SortKey{0, false}};
    goals[3].format = StorageFormat::Row;

    std::size_t total_pruned = 0;
    for (const PhysicalProperties& goal : goals) {
        auto logical = build_logical();

        LoweringContext pruned;
        pruned.spec = &*spec;
        pruned.storage = &both;
        pruned.required_output = goal;
        pruned.prune = true;

        LoweringContext exhaustive = pruned;
        exhaustive.prune = false;

        const LoweringResult rp = lower(*logical, pruned);
        const LoweringResult re = lower(*logical, exhaustive);
        CHECK(rp.ok == re.ok);
        if (!rp.ok || !re.ok) continue;
        CHECK(physical_to_sexpr(*rp.plan) == physical_to_sexpr(*re.plan));
        // The chosen plan's COST must agree too, not merely its shape.
        CHECK(rp.optimization_goals <= re.optimization_goals);
        total_pruned += rp.candidates_pruned;
    }

    // And pruning must actually be doing something, or the check above is vacuous.
    CHECK(total_pruned > 0);
}

// ---- Unit 2.4: structural memo dedup ---------------------------------------

// Two identical scans of the same relation share ONE group instead of being
// explored and optimized twice. Built by hand because no query in the corpus
// repeats a table - see the negative tests below for why the bar for sharing is
// deliberately high.
static plan::LogicalNodePtr twin_scan_join(bool same_schema) {
    auto left = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    left->table_name = "a";
    left->output = a_schema();
    auto right = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    right->table_name = "a";
    right->output = a_schema();
    if (!same_schema) right->output[0].alias = "other";  // a self-join's two sides
    auto join = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
    join->join_type = db25::ast::JoinType::Inner;
    join->output = a_schema();
    join->predicate = binop(BinaryOp::Equal, col(0, DataType::Integer),
                            col(2, DataType::Integer));
    join->add_child(std::move(left));
    join->add_child(std::move(right));
    return join;
}

static void test_identical_subtrees_share_a_group() {
    std::printf("test_identical_subtrees_share_a_group\n");
    // Dedup is off by default (it does not pay for itself yet - see
    // LoweringContext::dedup); these tests exercise the mechanism explicitly.
    auto logical = twin_scan_join(true);
    LoweringContext ctx;
    ctx.dedup = true;
    const LoweringResult r = lower(*logical, ctx);
    CHECK(r.ok);
    if (!r.plan) return;
    CHECK(r.groups_shared == 1);   // the second scan reused the first group
    CHECK(r.memo_groups == 2);     // one scan group + the join, not three

    // Sharing a group must not change the plan: both sides still appear.
    const std::string s = physical_to_sexpr(*r.plan);
    std::size_t scans = 0;
    for (std::size_t i = s.find("(SeqScan"); i != std::string::npos;
         i = s.find("(SeqScan", i + 1)) {
        ++scans;
    }
    CHECK(scans == 2);
}

// The dangerous failure for dedup is not a MISSED share, it is a WRONG one:
// merging two subtrees that are not equivalent plans one query as another. These
// pin the cases that must NOT merge.
static void test_non_equivalent_subtrees_never_share() {
    std::printf("test_non_equivalent_subtrees_never_share\n");

    // (a) Same table, different alias in the output schema - a self-join's two
    //     sides. They share (table_id, column_id), so the alias is the only thing
    //     that distinguishes them.
    {
        auto logical = twin_scan_join(false);
        LoweringContext ctx;
        ctx.dedup = true;
        const LoweringResult r = lower(*logical, ctx);
        CHECK(r.ok);
        CHECK(r.groups_shared == 0);
        CHECK(r.memo_groups == 3);
    }

    // (b) Filters over the same scan differing ONLY in a literal.
    {
        auto mk = [](std::int64_t k) {
            auto sc = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
            sc->table_name = "a";
            sc->output = a_schema();
            auto f = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
            f->output = a_schema();
            f->predicate = binop(BinaryOp::GreaterThan, col(1, DataType::Integer), int_lit(k));
            f->add_child(std::move(sc));
            return f;
        };
        auto join = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
        join->join_type = db25::ast::JoinType::Inner;
        join->output = a_schema();
        join->predicate = binop(BinaryOp::Equal, col(0, DataType::Integer),
                                col(2, DataType::Integer));
        join->add_child(mk(10));
        join->add_child(mk(20));   // differs only in the literal
        LoweringContext ctx;
        ctx.dedup = true;
        const LoweringResult r = lower(*join, ctx);
        CHECK(r.ok);
        // The two SCANS underneath are identical and legitimately share; the two
        // FILTERS are not equivalent and must not.
        CHECK(r.groups_shared == 1);
        CHECK(r.memo_groups == 4);  // scan, filter(>10), filter(>20), join
    }
}

// ---- Unit 2.5: the D5 budget guard ----------------------------------------

// The threshold is DATA. It is read from the spec, not compiled in, so it can be
// tuned per capability profile without a rebuild - and so a test can state the
// value it depends on instead of inheriting one.
static void test_search_budget_comes_from_the_spec() {
    std::printf("test_search_budget_comes_from_the_spec\n");
    std::string error;
    auto spec = db25::physical::load_spec(
        std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    CHECK(spec.has_value());
    if (!spec) return;
    CHECK(spec->budget.max_join_count == 8);

    // Absent from a spec, the documented default stands in rather than zero -
    // zero would put every query, including a single scan, past the guard.
    auto minimal = db25::physical::parse_spec(
        "(physical-spec (version 0)"
        "  (operators (operator (name SeqScan) (arity 0) (kind access-path)))"
        "  (capability-profile (name reference) (executes SeqScan)))",
        error);
    CHECK(minimal.has_value());
    if (minimal) CHECK(minimal->budget.max_join_count == 8);
}

// Below the budget the full property-directed search runs; above it the planner
// falls back to a bounded enumeration. Both produce a valid plan costed by the
// same model - what changes is how much of the space was examined.
//
// The guard is REPORTED, not silent. A caller that cares whether it got the
// cheapest plan the search could find, or merely an affordable one, has to be
// able to tell the difference; a planner that quietly degrades is a planner whose
// goldens mean different things on different days.
static void test_budget_guard_engages_and_is_reported() {
    std::printf("test_budget_guard_engages_and_is_reported\n");
    std::string error;
    auto spec = db25::physical::load_spec(
        std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    CHECK(spec.has_value());
    if (!spec) return;

    // One join, well under the shipped budget of 8.
    {
        auto logical = build_logical();
        LoweringContext ctx;
        ctx.spec = &*spec;
        const LoweringResult r = lower(*logical, ctx);
        CHECK(r.ok);
        CHECK(r.join_count == 1);
        CHECK(!r.budget_guard_engaged);   // the full search ran
    }

    // The same query with a budget it exceeds: the guard engages and says so.
    // The threshold is a maximum, so it trips when joins EXCEED it - a budget of
    // zero trips on any join at all.
    {
        auto logical = build_logical();
        auto tight = *spec;
        tight.budget.max_join_count = 0;
        LoweringContext ctx;
        ctx.spec = &tight;
        const LoweringResult r = lower(*logical, ctx);
        CHECK(r.ok);
        CHECK(r.join_count == 1);
        CHECK(r.budget_guard_engaged);   // reported, not silent
        CHECK(r.plan != nullptr);        // and still a real plan
    }

    // The context override exists so a caller can force either side of the
    // threshold without editing a spec; zero means "defer to the spec".
    {
        auto logical = build_logical();
        LoweringContext ctx;
        ctx.spec = &*spec;
        ctx.max_join_count_override = 1;
        CHECK(!lower(*logical, ctx).budget_guard_engaged);  // 1 join is not > 1
    }
}

// The guard must actually BOUND something, not merely set a flag. This is where
// it is observable: carrying a required order down to the join group happens on
// the pushed-down route, and that route is what the guard trades away. So the
// same query, asked for the same ordered output, plans differently on each side
// of the threshold - a MergeJoin whose own output is ordered when the full search
// runs, a HashJoin with a Sort stacked on top when it does not.
//
// Both are valid and both are costed by the same model. The guarded one is worse,
// which is the trade the guard exists to make.
static void test_guard_changes_what_the_search_finds() {
    std::printf("test_guard_changes_what_the_search_finds\n");
    std::string error;
    auto spec = db25::physical::load_spec(
        std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    if (!spec) { CHECK(false); return; }

    PhysicalProperties ordered;
    ordered.sort = {SortKey{0, false}};

    auto full_q = build_logical();
    LoweringContext full;
    full.spec = &*spec;
    full.required_output = ordered;
    const LoweringResult rf = lower(*full_q, full);
    CHECK(rf.ok);
    CHECK(!rf.budget_guard_engaged);

    auto guarded_q = build_logical();
    auto tight = *spec;
    tight.budget.max_join_count = 0;   // any join trips it
    LoweringContext guarded;
    guarded.spec = &tight;
    guarded.required_output = ordered;
    const LoweringResult rg = lower(*guarded_q, guarded);
    CHECK(rg.ok);
    CHECK(rg.budget_guard_engaged);
    if (!rf.plan || !rg.plan) return;

    const std::string sf = physical_to_sexpr(*rf.plan);
    const std::string sg = physical_to_sexpr(*rg.plan);
    CHECK(sf != sg);                      // the guard is not decorative
    CHECK(contains(sf, "MergeJoin"));     // full search carries the order down
    CHECK(contains(sg, "HashJoin"));      // guarded search enforces on top instead
    CHECK(contains(sg, "(Sort by="));

    // Whatever it chose, it still satisfies what was asked of it. A bounded
    // search may return a dearer plan; it may never return a wrong one.
    CHECK(satisfies(derive(*rg.plan), ordered));
}

// Bounded search must still be DETERMINISTIC. A guard that made planning depend
// on anything but its declared inputs would make goldens meaningless above the
// threshold, which is worse than being slow.
static void test_guarded_search_is_deterministic() {
    std::printf("test_guarded_search_is_deterministic\n");
    std::string error;
    auto spec = db25::physical::load_spec(
        std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    if (!spec) { CHECK(false); return; }
    auto tight = *spec;
    tight.budget.max_join_count = 0;

    std::string first;
    for (int i = 0; i < 5; ++i) {
        auto logical = build_logical();
        LoweringContext ctx;
        ctx.spec = &tight;
        const LoweringResult r = lower(*logical, ctx);
        CHECK(r.ok);
        if (!r.plan) return;
        const std::string s = physical_to_sexpr(*r.plan);
        if (i == 0) first = s;
        CHECK(s == first);
    }

    // And the guarded plan is a real, costed plan - not a degenerate one.
    CHECK(contains(first, "SeqScan"));
    CHECK(contains(first, "Join"));
}

static void test_unsupported_operator_is_an_error() {
    std::printf("test_unsupported_operator_is_an_error\n");
    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();
    // Deliberately an operator with NO implementation rule. This was Aggregate
    // until Increment 3.2 gave Aggregate one, at which point the test started
    // asserting that a supported operator fails - which is why it is written
    // against the furthest-out operator in the roadmap rather than the nearest.
    // When RecursiveCTE lowers, move this to whatever still does not.
    auto unsupported = std::make_unique<plan::LogicalNode>(plan::LogicalOp::RecursiveCTE);
    unsupported->output = a_schema();
    unsupported->add_child(std::move(scan));

    const LoweringResult r = lower(*unsupported);
    CHECK(!r.ok);
    CHECK(!r.error.empty());
    CHECK(r.plan == nullptr);
}

// MACHINERY test for cost-based selection. The two candidates here are not
// semantic substitutes for one another - Unit 1.2 supplies the first REAL
// alternative (MergeJoin) - but they exercise exactly the machinery that makes a
// plan CHOSEN rather than merely derived: the planner must enumerate both, cost
// them, keep the cheaper, and the winner must FLIP when the calibration says the
// other one is cheaper. That flip is what makes this falsifiable.
static void test_cost_chooses_between_candidates() {
    std::printf("test_cost_chooses_between_candidates\n");
    const std::string text =
        "(physical-spec (version 0)"
        "  (operators"
        "    (operator (name SeqScan)       (arity 0) (kind access-path))"
        "    (operator (name Filter)        (arity 1) (kind pipeline))"
        "    (operator (name Project)       (arity 1) (kind pipeline))"
        "    (operator (name HashJoin)      (arity 2) (kind pipeline-breaker))"
        "    (operator (name Sort)          (arity 1) (kind pipeline-breaker))"
        "    (operator (name FormatConvert) (arity 1) (kind pipeline)))"
        "  (implementation-rules"
        "    (rule (logical Scan)   (physical SeqScan))"
        "    (rule (logical Filter) (physical Filter))"
        "    (rule (logical Filter) (physical FormatConvert)))"
        "  (capability-profile (name reference)"
        "    (executes SeqScan Filter Project HashJoin Sort FormatConvert)))";
    std::string error;
    auto spec = db25::physical::parse_spec(text, error);
    CHECK(spec.has_value());
    if (!spec) return;
    CHECK(spec->physicals_for_logical("Filter").size() == 2);

    // Scan -> Filter.
    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();
    auto filt = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
    filt->output = a_schema();
    filt->predicate = binop(BinaryOp::GreaterThan, col(1, DataType::Integer), int_lit(10));
    filt->add_child(std::move(scan));

    LoweringContext ctx;
    ctx.spec = &*spec;

    // Default calibration: filter_row (0.5) < convert_row (0.6), so Filter wins.
    const LoweringResult by_default = lower(*filt, ctx);
    CHECK(by_default.ok);
    CHECK(by_default.candidates_considered == 3);  // 1 scan + 2 filter candidates
    if (by_default.plan) CHECK(by_default.plan->op == PhysicalOp::Filter);

    // Make the convert far cheaper: the OTHER candidate must now win. Same plan,
    // same spec - only the cost model changed.
    db25::physical::CalibrationProfile flipped = db25::physical::default_calibration();
    flipped.convert_row = 0.01;
    ctx.calibration = &flipped;
    const LoweringResult when_flipped = lower(*filt, ctx);
    CHECK(when_flipped.ok);
    if (when_flipped.plan) CHECK(when_flipped.plan->op == PhysicalOp::FormatConvert);
}

// The first REAL algorithm choice: HashJoin vs MergeJoin. MergeJoin is cheaper
// per row but needs both inputs sorted on the join keys; HashJoin needs nothing.
// So the decision turns on whether the Sorts MergeJoin would cause are worth its
// cheaper merge - which is precisely what enforcement_cost charges it for. Both
// directions are checked: whichever way the calibration leans, the planner must
// follow it, and when MergeJoin wins the Sorts it requires must actually be in
// the plan that comes out (the plan built is the plan that was costed).
static void test_join_algorithm_is_chosen_by_cost() {
    std::printf("test_join_algorithm_is_chosen_by_cost\n");
    std::string error;
    auto spec =
        db25::physical::load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr",
                                  error);
    CHECK(spec.has_value());
    if (!spec) { std::printf("  spec load error: %s\n", error.c_str()); return; }
    CHECK(spec->physicals_for_logical("Join").size() == 3);  // Hash + Merge + NestedLoop

    auto logical = build_logical();
    LoweringContext ctx;
    ctx.spec = &*spec;

    // Sorting is expensive by default, so a MergeJoin would have to pay for two
    // Sorts its inputs do not provide. The HashJoin wins.
    const LoweringResult hashed = lower(*logical, ctx);
    CHECK(hashed.ok);
    if (hashed.plan) {
        const std::string s = physical_to_sexpr(*hashed.plan);
        CHECK(contains(s, "HashJoin"));
        CHECK(!contains(s, "MergeJoin"));
        CHECK(!contains(s, "(Sort by="));  // nothing needed enforcing
    }

    // Make sorting free: the merge's cheaper per-row cost is no longer outweighed
    // by the enforcement it causes, so it wins - and the Sorts must appear.
    db25::physical::CalibrationProfile free_sort = db25::physical::default_calibration();
    free_sort.sort_row = 0.0;
    ctx.calibration = &free_sort;
    const LoweringResult merged = lower(*logical, ctx);
    CHECK(merged.ok);
    if (merged.plan) {
        const std::string s = physical_to_sexpr(*merged.plan);
        CHECK(contains(s, "MergeJoin"));
        CHECK(contains(s, "(Sort by="));  // the enforcers its requirement caused
    }
}

// The HTAP substrate as an ordinary costed choice. A table available in both
// formats gives the scan one candidate per format; nothing above a scan-and-
// filter pipeline requires a particular format, so the cheaper columnar read
// simply wins - and with a row-only catalog (the default) nothing changes.
static void test_storage_substrate_is_chosen_by_cost() {
    std::printf("test_storage_substrate_is_chosen_by_cost\n");

    const auto make_pipeline = []() {
        auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
        scan->table_name = "a";
        scan->output = a_schema();
        auto filt = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
        filt->output = a_schema();
        filt->predicate = binop(BinaryOp::GreaterThan, col(1, DataType::Integer), int_lit(10));
        filt->add_child(std::move(scan));
        return filt;
    };

    // NOTE: each logical plan is held in a named local. The physical plan BORROWS
    // its expression payloads, so the logical plan must outlive it - lowering a
    // temporary would leave those borrows dangling.
    const auto p_row = make_pipeline();
    const LoweringResult row_only = lower(*p_row);
    CHECK(row_only.ok);
    if (row_only.plan) {
        const std::string s = physical_to_sexpr(*row_only.plan);
        CHECK(contains(s, "fmt=row"));
        CHECK(!contains(s, "fmt=column"));
    }

    // Available in both: the columnar read is cheaper and nothing here needs rows.
    db25::physical::StorageCatalog dual;
    dual.formats["a"] = {db25::physical::StorageFormat::Row,
                         db25::physical::StorageFormat::Column};
    LoweringContext ctx;
    ctx.storage = &dual;
    const auto p_both = make_pipeline();
    const LoweringResult both = lower(*p_both, ctx);
    CHECK(both.ok);
    CHECK(both.candidates_considered == 3);  // 2 scan formats + 1 filter
    if (both.plan) {
        const std::string s = physical_to_sexpr(*both.plan);
        CHECK(contains(s, "fmt=column"));
        CHECK(!contains(s, "FormatConvert"));  // nothing required a conversion
    }

    // Make the columnar read the expensive one: the choice must follow the cost.
    db25::physical::CalibrationProfile dear_column = db25::physical::default_calibration();
    dear_column.column_scan_row = 10.0;
    ctx.calibration = &dear_column;
    const auto p_flip = make_pipeline();
    const LoweringResult flipped = lower(*p_flip, ctx);
    CHECK(flipped.ok);
    if (flipped.plan) CHECK(contains(physical_to_sexpr(*flipped.plan), "fmt=row"));
}

// A join consumes row-format input, so a columnar scan feeding one must be
// converted - and that conversion is PRICED, which is what stops "columnar is
// always cheaper" from being the answer.
//
// This also pins a known LIMITATION rather than hiding it: each group commits to
// a single winner chosen bottom-up, so the scan takes the locally-cheaper
// columnar format without knowing the join above will demand rows. When the
// conversion is dear enough that a row scan would have been globally better, this
// bottom-up choice cannot see it. Lifting that needs a winner per required-
// property set (property-directed group optimization, Increment 2). The assertion
// here is therefore about the MECHANISM - the requirement is enforced and paid
// for - not about global optimality, which this increment does not yet claim.
static void test_join_requires_rows_and_conversion_is_priced() {
    std::printf("test_join_requires_rows_and_conversion_is_priced\n");
    db25::physical::StorageCatalog dual;
    dual.formats["a"] = {db25::physical::StorageFormat::Row,
                         db25::physical::StorageFormat::Column};
    dual.formats["b"] = {db25::physical::StorageFormat::Row,
                         db25::physical::StorageFormat::Column};
    LoweringContext ctx;
    ctx.storage = &dual;

    auto logical = build_logical();
    const LoweringResult r = lower(*logical, ctx);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    // The join's row requirement was enforced on the columnar inputs.
    CHECK(contains(s, "fmt=column"));
    CHECK(contains(s, "FormatConvert to=row"));

    // And the requirement is real: a join always ends up fed row-format input.
    const auto reqs = db25::physical::required_input_properties(
        db25::physical::PhysicalOp::HashJoin, {{0, 0}});
    CHECK(reqs.size() == 2);
    CHECK(reqs[0].format == db25::physical::StorageFormat::Row);
    CHECK(reqs[1].format == db25::physical::StorageFormat::Row);
}

// Freshness is where the property model stops being about cost. In Unit 1.3 the
// cheaper columnar copy simply won. Here the cheaper columnar copy LAGS, and a
// query that must see the latest writes cannot use it at any price - so the
// planner must pick the dearer fresh copy, and when no fresh copy exists it must
// fail honestly rather than answer from stale data.
static void test_freshness_overrides_cost() {
    std::printf("test_freshness_overrides_cost\n");

    const auto make_pipeline = []() {
        auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
        scan->table_name = "a";
        scan->output = a_schema();
        auto filt = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
        filt->output = a_schema();
        filt->predicate = binop(BinaryOp::GreaterThan, col(1, DataType::Integer), int_lit(10));
        filt->add_child(std::move(scan));
        return filt;
    };

    // The row copy is fresh; the columnar replica is cheaper but lags.
    db25::physical::StorageCatalog cat;
    cat.formats["a"] = {
        db25::physical::FormatAvailability{db25::physical::StorageFormat::Row,
                                           db25::physical::Freshness::Fresh},
        db25::physical::FormatAvailability{db25::physical::StorageFormat::Column,
                                           db25::physical::Freshness::Stale}};

    // No freshness demand: cost decides, and the cheap stale replica wins.
    LoweringContext relaxed;
    relaxed.storage = &cat;
    const auto p1 = make_pipeline();
    const LoweringResult cheap = lower(*p1, relaxed);
    CHECK(cheap.ok);
    if (cheap.plan) {
        const std::string s2 = physical_to_sexpr(*cheap.plan);
        CHECK(contains(s2, "fmt=column"));
        CHECK(contains(s2, "stale"));
    }

    // Demand fresh data: the cheaper copy is now INELIGIBLE, not merely dearer.
    LoweringContext strict;
    strict.storage = &cat;
    strict.required_freshness = db25::physical::Freshness::Fresh;
    const auto p2 = make_pipeline();
    const LoweringResult correct = lower(*p2, strict);
    CHECK(correct.ok);
    if (correct.plan) {
        const std::string s2 = physical_to_sexpr(*correct.plan);
        CHECK(contains(s2, "fmt=row"));     // the dearer copy, chosen for correctness
        CHECK(!contains(s2, "fmt=column"));
        CHECK(!contains(s2, "stale"));
    }
    // Only one scan candidate survived the constraint (plus the filter).
    CHECK(correct.candidates_considered < cheap.candidates_considered);
}

// When nothing can answer the query correctly, say so - do not quietly serve
// stale rows.
static void test_no_fresh_copy_is_an_honest_failure() {
    std::printf("test_no_fresh_copy_is_an_honest_failure\n");
    db25::physical::StorageCatalog stale_only;
    stale_only.formats["a"] = {
        db25::physical::FormatAvailability{db25::physical::StorageFormat::Column,
                                           db25::physical::Freshness::Stale}};

    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();

    LoweringContext ctx;
    ctx.storage = &stale_only;
    ctx.required_freshness = db25::physical::Freshness::Fresh;
    const LoweringResult r = lower(*scan, ctx);
    CHECK(!r.ok);
    CHECK(r.plan == nullptr);
    CHECK(r.error.find("fresh") != std::string::npos);
}

// A merge join merges two sorted streams ON THE JOIN KEYS. A keyless (CROSS)
// join has nothing to merge on, so MergeJoin is not a valid implementation of it
// - at any cost. This is not something costing can be trusted to settle: with no
// keys a MergeJoin also requires no sort, so it incurs no enforcement and its
// cheaper per-row cost WINS. Applicability has to be decided before cost.
//
// Regression: the umbrella's physical golden stage caught exactly this on the
// cross-join and LATERAL fixtures, where the planner had started emitting
// MergeJoin over a cross product.
static void test_keyless_join_never_merges() {
    std::printf("test_keyless_join_never_merges\n");
    std::string error;
    auto spec =
        db25::physical::load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr",
                                  error);
    CHECK(spec.has_value());
    if (!spec) return;

    // A CROSS join: no ON condition, hence no equi-keys.
    auto scan_a = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan_a->table_name = "a";
    scan_a->output = a_schema();
    auto scan_b = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan_b->table_name = "b";
    scan_b->output = b_schema();
    auto cross = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
    cross->join_type = db25::ast::JoinType::Cross;
    cross->output = join_schema();
    cross->predicate = nullptr;  // no keys
    cross->add_child(std::move(scan_a));
    cross->add_child(std::move(scan_b));

    LoweringContext ctx;
    ctx.spec = &*spec;
    const LoweringResult r = lower(*cross, ctx);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s2 = physical_to_sexpr(*r.plan);
    CHECK(contains(s2, "keys=[]"));
    CHECK(contains(s2, "NestedLoopJoin"));  // the only join that needs no key
    CHECK(!contains(s2, "MergeJoin"));      // nothing to merge on
    CHECK(!contains(s2, "HashJoin"));       // nothing to hash on, either

    // And the predicate itself. This originally asserted that HashJoin is
    // "applicable either way", which blessed the very plan it was written to
    // prevent: a keyless HashJoin has nothing to hash on and is exactly as
    // unexecutable as the keyless MergeJoin. Guarding one keyed join and not the
    // other did not close the class, it moved the defect next door.
    CHECK(!db25::physical::is_applicable(db25::physical::PhysicalOp::MergeJoin, {}));
    CHECK(!db25::physical::is_applicable(db25::physical::PhysicalOp::HashJoin, {}));
    CHECK(db25::physical::is_applicable(db25::physical::PhysicalOp::MergeJoin, {{0, 0}}));
    CHECK(db25::physical::is_applicable(db25::physical::PhysicalOp::HashJoin, {{0, 0}}));
    // The nested loop is applicable with or without keys - that is its role.
    CHECK(db25::physical::is_applicable(db25::physical::PhysicalOp::NestedLoopJoin, {}));
}

// A purely non-equi join has no keys either, so it too is a nested loop - and the
// condition survives as a residual rather than being silently dropped.
static void test_non_equi_join_is_a_nested_loop() {
    std::printf("test_non_equi_join_is_a_nested_loop\n");
    auto pred = binop(BinaryOp::GreaterThan, col(1, DataType::Integer), col(2, DataType::Integer));
    auto logical = build_logical(std::move(pred));
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "NestedLoopJoin kind=inner keys=[]"));
    CHECK(contains(s, "residual=[(> (col #1) (col #2))]"));
}

// The keyed joins must still win where they apply: a nested loop is quadratic, so
// it is only ever selected because it is the sole APPLICABLE candidate, never
// because it is cheap. This is what keeps adding it from degrading equi-joins.
static void test_nested_loop_never_wins_an_equi_join() {
    std::printf("test_nested_loop_never_wins_an_equi_join\n");
    auto logical = build_logical();  // a.id = b.id
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(!contains(s, "NestedLoopJoin"));  // considered, and correctly rejected
    CHECK(contains(s, "HashJoin kind=inner keys=[(L#0 R#0)]"));
}

// ---------------------------------------------------------------------------
// The join kind must survive lowering.
//
// It did not. `lower()` switched on LogicalOp::Join and never read `join_type`,
// so an INNER and a LEFT join over the same inputs with the same predicate came
// out as BYTE-IDENTICAL physical plans - the same HashJoin node, the same keys,
// the same rendering. The only trace of "LEFT" anywhere was a nullability flag
// in the output schema, which the logical plan had already computed and which
// describes the RESULT, not the operator's obligation. An executor handed that
// plan has nothing to tell it to null-extend unmatched left rows.
//
// The failure mode is the dangerous one: `lower()` returned ok, the plan looked
// entirely reasonable, and the goldens blessed it.
static void test_join_kind_reaches_the_physical_plan() {
    std::printf("test_join_kind_reaches_the_physical_plan\n");

    auto render = [](db25::ast::JoinType kind) {
        auto logical = build_logical(nullptr, kind);
        const LoweringResult r = lower(*logical);
        CHECK(r.ok);
        return r.plan ? physical_to_sexpr(*r.plan) : std::string{};
    };

    const std::string inner = render(db25::ast::JoinType::Inner);
    const std::string left  = render(db25::ast::JoinType::Left);
    const std::string right = render(db25::ast::JoinType::Right);
    const std::string full  = render(db25::ast::JoinType::Full);

    CHECK(contains(inner, "HashJoin kind=inner"));
    CHECK(contains(left,  "HashJoin kind=left"));
    CHECK(contains(right, "HashJoin kind=right"));
    CHECK(contains(full,  "HashJoin kind=full"));

    // The point of the test, stated as the thing that was false before: four
    // different relational joins must not produce one plan.
    CHECK(inner != left);
    CHECK(left != right);
    CHECK(right != full);

    // The semantics an executor reads off the kind. These are the whole reason
    // the field exists, so they are pinned rather than left to be re-derived.
    using db25::ast::JoinType;
    using db25::physical::join_is_lateral;
    using db25::physical::join_null_extends_left;
    using db25::physical::join_null_extends_right;
    CHECK(!join_null_extends_left(JoinType::Inner));
    CHECK(!join_null_extends_right(JoinType::Inner));
    CHECK(join_null_extends_right(JoinType::Left));
    CHECK(!join_null_extends_left(JoinType::Left));
    CHECK(join_null_extends_left(JoinType::Right));
    CHECK(!join_null_extends_right(JoinType::Right));
    CHECK(join_null_extends_left(JoinType::Full));
    CHECK(join_null_extends_right(JoinType::Full));
    // A LEFT JOIN LATERAL null-extends its right side too - it is an outer join
    // that also happens to be correlated, and losing either half is a bug.
    CHECK(join_null_extends_right(JoinType::LeftLateral));
    CHECK(join_is_lateral(JoinType::LeftLateral));
    CHECK(join_is_lateral(JoinType::Lateral));
    CHECK(!join_is_lateral(JoinType::Left));
}

// Two joins that differ ONLY in kind are not equivalent, so memo dedup must not
// merge them. `logical_op` cannot separate them - both are LogicalOp::Join - so
// without the kind in the group key, one query's LEFT join could be answered by
// another's INNER join winner. This checks the key directly, because arranging
// two differently-keyed joins over identical inputs inside one plan is exactly
// the situation dedup exists to collapse.
static void test_inner_and_left_are_not_the_same_group() {
    std::printf("test_inner_and_left_are_not_the_same_group\n");
    const db25::plan::Schema out = join_schema();
    const std::string table = "a";
    auto pred = binop(BinaryOp::Equal, col(0, DataType::Integer), col(2, DataType::Integer));

    auto make = [&](db25::ast::JoinType kind) {
        db25::physical::GroupKey k;
        k.logical_op = static_cast<int>(plan::LogicalOp::Join);
        k.table_name = nullptr;
        k.output = &out;
        k.predicate = nullptr;
        k.hash_keys = {{0, 0}};
        k.join_kind = kind;
        k.finish();
        return k;
    };
    const db25::physical::GroupKey inner = make(db25::ast::JoinType::Inner);
    const db25::physical::GroupKey left  = make(db25::ast::JoinType::Left);
    const db25::physical::GroupKey inner2 = make(db25::ast::JoinType::Inner);

    CHECK(!inner.equals(left));       // the whole point
    CHECK(inner.equals(inner2));      // and it still shares what it should
    CHECK(inner.hash != left.hash);   // separated by the hash too, not only by equals
}

// A LATERAL join's right input is correlated: its subtree reads the current left
// row through an OuterRef. Only a nested loop can evaluate that. A hash join
// builds its table from the right input ONCE, standalone - which for a
// correlated right input is not slower, it is not executable.
//
// The bug was shape-dependent, which is why it survived: `LEFT JOIN LATERAL (..)
// ON true` yields no equi-key, so it fell to NestedLoopJoin anyway and the
// golden looked right. Give the ON clause an equality and the identical query
// lowered to a HashJoin over a correlated right input, with lower() ok. This
// test therefore uses an equi predicate deliberately - without one it would pass
// against the broken code and prove nothing.
static void test_lateral_join_is_never_a_hash_join() {
    std::printf("test_lateral_join_is_never_a_hash_join\n");
    for (const db25::ast::JoinType kind :
         {db25::ast::JoinType::Lateral, db25::ast::JoinType::LeftLateral}) {
        auto logical = build_logical(nullptr, kind);  // predicate is a.id = b.id
        const LoweringResult r = lower(*logical);
        CHECK(r.ok);
        if (!r.plan) continue;
        const std::string s = physical_to_sexpr(*r.plan);
        CHECK(contains(s, "NestedLoopJoin"));
        CHECK(!contains(s, "HashJoin"));   // the equi-key must NOT buy a hash join
        CHECK(!contains(s, "MergeJoin"));
        // The keys are still recorded - the nested loop uses them as a per-row
        // match condition. Dropping them would be a different bug.
        CHECK(contains(s, "keys=[(L#0 R#0)]"));
    }

    // And directly on the predicate, so the guard is pinned independently of
    // whatever the cost model happens to prefer today.
    using db25::physical::PhysicalOp;
    CHECK(!db25::physical::is_applicable(PhysicalOp::HashJoin, {{0, 0}},
                                         db25::ast::JoinType::LeftLateral));
    CHECK(!db25::physical::is_applicable(PhysicalOp::MergeJoin, {{0, 0}},
                                         db25::ast::JoinType::Lateral));
    CHECK(db25::physical::is_applicable(PhysicalOp::NestedLoopJoin, {{0, 0}},
                                        db25::ast::JoinType::LeftLateral));
    // A non-lateral join is untouched by the guard.
    CHECK(db25::physical::is_applicable(PhysicalOp::HashJoin, {{0, 0}},
                                        db25::ast::JoinType::Left));
}

// ---------------------------------------------------------------------------
// Increment 3.1: ORDER BY and LIMIT lower.
//
// Before this, LogicalOp::Sort and LogicalOp::Limit had no implementation rule,
// so any query with an ORDER BY or a LIMIT failed lowering outright - 18 of the
// umbrella's 37 staged fixtures could not reach a physical plan at all, and the
// reference query was one of them.
static plan::LogicalNodePtr build_sort_limit(bool with_limit, bool descending = false,
                                             bool nulls_specified = false,
                                             bool nulls_first = false) {
    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();

    auto sort = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Sort);
    sort->output = a_schema();
    plan::SortKeyIR k;
    k.expr = col(1, DataType::Integer);   // ORDER BY a.x
    k.descending = descending;
    k.nulls_order_explicit = nulls_specified;
    k.nulls_first = nulls_first;
    sort->sort_keys.push_back(std::move(k));
    sort->add_child(std::move(scan));
    if (!with_limit) return sort;

    auto limit = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Limit);
    limit->output = a_schema();
    limit->has_limit = true;
    limit->limit = 10;
    limit->has_offset = true;
    limit->offset = 5;
    limit->add_child(std::move(sort));
    return limit;
}

static void test_order_by_and_limit_lower() {
    std::printf("test_order_by_and_limit_lower\n");
    auto logical = build_sort_limit(/*with_limit=*/true);
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "(Limit limit=10 offset=5"));
    CHECK(contains(s, "(Sort by=[#1 asc]"));
    CHECK(contains(s, "(SeqScan table=a"));

    // The Limit is ABOVE the Sort, matching the logical plan. Inverting them
    // would keep a different set of rows entirely.
    CHECK(s.find("Limit") < s.find("Sort"));

    // Descending is carried, not flattened to the default.
    auto desc = build_sort_limit(false, /*descending=*/true);
    const LoweringResult rd = lower(*desc);
    CHECK(rd.ok);
    if (rd.plan) CHECK(contains(physical_to_sexpr(*rd.plan), "(Sort by=[#1 desc]"));
}

// NULLS FIRST / NULLS LAST changes which rows come out first, so a Sort that
// dropped it would answer a different query - the same shape of defect as a join
// that forgot it was an outer join. The rendering distinguishes all three states:
// unspecified (nothing printed), explicit FIRST, explicit LAST.
static void test_nulls_ordering_survives_lowering() {
    std::printf("test_nulls_ordering_survives_lowering\n");
    auto render = [](bool specified, bool first) {
        auto l = build_sort_limit(false, false, specified, first);
        const LoweringResult r = lower(*l);
        CHECK(r.ok);
        return r.plan ? physical_to_sexpr(*r.plan) : std::string{};
    };
    const std::string none  = render(false, false);
    const std::string first = render(true, true);
    const std::string last  = render(true, false);

    CHECK(contains(none,  "by=[#1 asc]"));
    CHECK(contains(first, "by=[#1 asc nulls-first]"));
    CHECK(contains(last,  "by=[#1 asc nulls-last]"));
    CHECK(none != first);
    CHECK(first != last);
    CHECK(none != last);

    // And the satisfaction rule that makes the field meaningful: a requirement
    // with no opinion about NULLs is met by any placement, while an explicit one
    // is met only by itself. Ignoring the field would make all four of these
    // true; requiring an exact match unconditionally would make all four false.
    using db25::physical::PhysicalProperties;
    const SortKey any_nulls{1, false, false, false};
    const SortKey nulls_first_key{1, false, true, true};
    const SortKey nulls_last_key{1, false, true, false};
    const auto sat = [](SortKey provided, SortKey required) {
        return satisfies(PhysicalProperties{{provided}, StorageFormat::Any, Freshness::Any},
                         PhysicalProperties{{required}, StorageFormat::Any, Freshness::Any});
    };
    CHECK(sat(nulls_first_key, any_nulls));        // no opinion: satisfied
    CHECK(sat(nulls_first_key, nulls_first_key));  // exact: satisfied
    CHECK(!sat(nulls_first_key, nulls_last_key));  // opposite: NOT satisfied
    CHECK(!sat(any_nulls, nulls_first_key));       // unknown cannot serve explicit
}

// A Sort must ADVERTISE the order it establishes, or the search cannot see that
// an ORDER BY already satisfies a consumer's requirement and will stack a second
// Sort on top of the first. derive_op is the single place that answers this, for
// both the memo form and the tree form.
static void test_a_lowered_sort_provides_its_order() {
    std::printf("test_a_lowered_sort_provides_its_order\n");
    auto logical = build_sort_limit(/*with_limit=*/false);
    LoweringContext ctx;
    ctx.required_output.sort = {SortKey{1, false, false, false}};  // the order it already has
    const LoweringResult r = lower(*logical, ctx);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    // Exactly one Sort: the one the query asked for. A second would mean the
    // search could not tell that the first had already established the order.
    std::size_t sorts = 0;
    for (std::size_t i = s.find("(Sort "); i != std::string::npos; i = s.find("(Sort ", i + 1)) {
        ++sorts;
    }
    CHECK(sorts == 1);
    CHECK(satisfies(derive(*r.plan), ctx.required_output));
}

// A Limit is ORDER-SENSITIVE: which rows survive depends on the order its input
// arrives in. So the planner must not discharge a consumer's order requirement by
// pushing it INTO the Limit's input - that changes the result set, not just the
// plan. Sort_x(Limit_10(c)) takes ten rows in c's own order then sorts those ten;
// Limit_10(Sort_x(c)) takes the ten smallest by x. Different answers.
static void test_limit_never_pushes_an_order_requirement_down() {
    std::printf("test_limit_never_pushes_an_order_requirement_down\n");
    using db25::physical::PhysicalOp;
    PhysicalProperties want;
    want.sort = {SortKey{0, false, false, false}};
    CHECK(!pushdown_requirement(PhysicalOp::Limit, want).has_value());
    // The order-preserving pipeline operators still do push down - the guard is
    // specific to Limit, not a blanket refusal that would cost every plan.
    CHECK(pushdown_requirement(PhysicalOp::Filter, want).has_value());
    CHECK(pushdown_requirement(PhysicalOp::Project, want).has_value());
}

// A Limit's OUTPUT cardinality is not its input's. Estimating it as the input's
// would over-count every operator above it and skew their cost comparisons.
static void test_limit_bounds_the_row_estimate() {
    std::printf("test_limit_bounds_the_row_estimate\n");
    const db25::physical::LimitSpec l10{true, false, 10, 0};
    const db25::physical::LimitSpec off5{false, true, -1, 5};
    const db25::physical::LimitSpec both{true, true, 10, 5};
    CHECK(l10.rows_out(1000.0) == 10.0);
    CHECK(l10.rows_out(3.0) == 3.0);           // a limit is a cap, not a promise
    CHECK(off5.rows_out(1000.0) == 995.0);
    CHECK(off5.rows_out(2.0) == 0.0);          // an offset past the end leaves none
    // OFFSET applies FIRST, then LIMIT caps the remainder. The other order would
    // report 10 here rather than the correct 10 (of the 995 that remain) - and
    // would report 5 for an input of 12.
    CHECK(both.rows_out(1000.0) == 10.0);
    CHECK(both.rows_out(12.0) == 7.0);

    // And that the cost model actually CONSULTS it. Testing rows_out alone left
    // the wiring untested: replacing operator_rows' Limit case with `return
    // in(0)` - the exact over-estimate this exists to prevent - kept every
    // assertion above green.
    const db25::physical::CardinalityModel card;
    const double in_rows[1] = {1000.0};
    CHECK(operator_rows(db25::physical::PhysicalOp::Limit,
                        std::span<const double>{in_rows, 1}, "", card, both) == 10.0);
    CHECK(operator_rows(db25::physical::PhysicalOp::Limit,
                        std::span<const double>{in_rows, 1}, "", card, {}) == 1000.0);

    // End to end, through a real lowered plan: the estimate at the root of
    // `ORDER BY x LIMIT 10 OFFSET 5` is bounded by the limit, whatever the base
    // table's cardinality is.
    auto logical = build_sort_limit(/*with_limit=*/true);
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (r.plan) CHECK(card.rows(*r.plan) == 10.0);
}

// A sort key must be a positional column index, so a key that is not a plain
// column reference has no index to be. Inventing one would sort by whatever
// column that index happened to name - a wrong answer reported as success. Fail
// honestly instead, with an error that says why.
static void test_a_computed_sort_key_fails_honestly() {
    std::printf("test_a_computed_sort_key_fails_honestly\n");
    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();
    auto sort = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Sort);
    sort->output = a_schema();
    plan::SortKeyIR k;
    k.expr = binop(BinaryOp::Add, col(0, DataType::Integer), col(1, DataType::Integer));
    sort->sort_keys.push_back(std::move(k));
    sort->add_child(std::move(scan));

    const LoweringResult r = lower(*sort);
    CHECK(!r.ok);
    CHECK(r.plan == nullptr);
    CHECK(contains(r.error, "sort key"));
}

// Two Sorts differing only in their keys, or two Limits differing only in their
// bounds, are different operators sharing one LogicalOp - so memo dedup must not
// merge them, exactly as for the join kind.
static void test_sorts_and_limits_are_not_interchangeable_groups() {
    std::printf("test_sorts_and_limits_are_not_interchangeable_groups\n");
    const db25::plan::Schema out = a_schema();
    auto make_sort = [&](std::uint32_t column, bool descending) {
        db25::physical::GroupKey k;
        k.logical_op = static_cast<int>(plan::LogicalOp::Sort);
        k.output = &out;
        k.sort_keys = {SortKey{column, descending, false, false}};
        k.finish();
        return k;
    };
    CHECK(!make_sort(0, false).equals(make_sort(1, false)));  // different column
    CHECK(!make_sort(0, false).equals(make_sort(0, true)));   // different direction
    CHECK(make_sort(0, false).equals(make_sort(0, false)));   // and still shares

    auto make_limit = [&](std::int64_t n) {
        db25::physical::GroupKey k;
        k.logical_op = static_cast<int>(plan::LogicalOp::Limit);
        k.output = &out;
        k.limits = db25::physical::LimitSpec{true, false, n, 0};
        k.finish();
        return k;
    };
    CHECK(!make_limit(10).equals(make_limit(20)));
    CHECK(make_limit(10).equals(make_limit(10)));

    // And the aggregate split. `GROUP BY a` computing agg(b) and `GROUP BY a, b`
    // computing nothing carry the SAME expression list on the group - the payload
    // shares one vector - and differ only in where it splits. Without the split
    // count in the key they would dedup into one group and one would silently
    // answer for the other. (Found by mutation: removing group_key_count from the
    // key broke nothing until this existed.)
    auto keya = col(0, DataType::Integer);
    auto keyb = col(1, DataType::Integer);
    auto make_agg = [&](std::uint32_t split) {
        db25::physical::GroupKey k;
        k.logical_op = static_cast<int>(plan::LogicalOp::Aggregate);
        k.output = &out;
        k.op_exprs = {keya.get(), keyb.get()};
        k.group_key_count = split;
        k.finish();
        return k;
    };
    CHECK(!make_agg(1).equals(make_agg(2)));
    CHECK(make_agg(1).hash != make_agg(2).hash);
    CHECK(make_agg(1).equals(make_agg(1)));
}

// ---------------------------------------------------------------------------
// Increment 3.2: GROUP BY lowers, and WHICH aggregate is a cost decision.
//
// This is the first property-directed choice outside a join. A streaming
// aggregate is cheaper per row but needs its input sorted on the grouping keys;
// hashing needs nothing but produces no order. Neither is universally better,
// which is exactly the shape Increment 2's search was built for.
static plan::LogicalNodePtr build_aggregate(bool computed_key = false,
                                            bool scalar = false,
                                            bool grouping_sets = false) {
    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();

    auto agg = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Aggregate);
    agg->output = a_schema();
    if (!scalar) {
        agg->group_keys.push_back(computed_key
            ? binop(BinaryOp::Add, col(0, DataType::Integer), col(1, DataType::Integer))
            : col(0, DataType::Integer));
    }
    agg->aggregates.push_back(col(1, DataType::Integer));  // stands in for an agg call
    if (grouping_sets) agg->grouping_sets.push_back({0});
    agg->add_child(std::move(scan));
    return agg;
}

static void test_group_by_lowers() {
    std::printf("test_group_by_lowers\n");
    auto logical = build_aggregate();
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "Aggregate keys=[(col #0)] aggs=[(col #1)]"));
    CHECK(contains(s, "(SeqScan table=a"));
    // Both candidates were built - the choice is real, not a single option
    // dressed up as one.
    CHECK(r.candidates_considered >= 2);
}

// The trade, made visible - and it does NOT go the way one first guesses.
//
// A streaming aggregate is cheaper per row (0.5 vs 1.1) but must have its input
// sorted on the grouping keys. The naive expectation is that requiring the
// aggregate's output in grouping-key order tips the balance to streaming, since
// the sort is then needed anyway. It does not, and the cost model is right to say
// so: an aggregate REDUCES its input, so a Sort placed ABOVE it sorts the groups
// while a Sort placed BELOW it sorts every row. With the default 10% group
// selectivity that is 100 rows against 1000, and hashing wins comfortably:
//
//   groups   hash + sort-above    sort-below + stream    winner
//      100              1764.4                10465.8    HASH
//      500              5582.9                10465.8    HASH
//     1000             11065.8                10465.8    STREAM
//
// Streaming wins where the grouping key is HIGH-CARDINALITY - the aggregate
// barely reduces, both routes sort the same number of rows, and streaming's
// cheaper per-row rate is all that is left to separate them. That is the real
// condition, so that is what this test sets up. The first version of this test
// asserted the naive expectation and failed; the planner was correct and the
// test was not.
//
// If both halves ever show the same operator, the second candidate has become
// decoration.
static void test_aggregate_algorithm_is_chosen_by_cost() {
    std::printf("test_aggregate_algorithm_is_chosen_by_cost\n");

    auto unconstrained = build_aggregate();
    const LoweringResult ru = lower(*unconstrained);
    CHECK(ru.ok);
    const std::string su = ru.plan ? physical_to_sexpr(*ru.plan) : std::string{};
    CHECK(contains(su, "HashAggregate"));

    // Every row its own group: the aggregate reduces nothing, so the Sort costs
    // the same on either side of it and only the per-row rate decides.
    db25::physical::CardinalityModel distinct_keys;
    distinct_keys.group_selectivity = 1.0;
    LoweringContext ctx;
    ctx.cardinality = &distinct_keys;
    ctx.required_output.sort = {SortKey{0, false, false, false}};

    auto ordered = build_aggregate();
    const LoweringResult ro = lower(*ordered, ctx);
    CHECK(ro.ok);
    const std::string so = ro.plan ? physical_to_sexpr(*ro.plan) : std::string{};
    CHECK(contains(so, "StreamingAggregate"));
    CHECK(su != so);

    // The chosen plan must actually satisfy what was asked of it, not merely pick
    // the operator this test hoped for.
    if (ro.plan) CHECK(satisfies(derive(*ro.plan), ctx.required_output));
    // And no Sort may sit ABOVE the streaming aggregate: it already emits the
    // grouping order, and a Sort there would mean the search cannot see that.
    CHECK(so.find("Sort") > so.find("StreamingAggregate"));
    // The Sort that IS there feeds the aggregate, so it must be below it.
    CHECK(contains(so, "Sort"));
}

// A grouping key that is not a plain column reference cannot be expressed as a
// sort requirement, so streaming is inapplicable - but the query still lowers,
// via hashing, which hashes the expression itself. A missing capability costs a
// plan ALTERNATIVE here, not an answer.
static void test_a_computed_grouping_key_still_lowers_via_hash() {
    std::printf("test_a_computed_grouping_key_still_lowers_via_hash\n");
    auto logical = build_aggregate(/*computed_key=*/true);
    LoweringContext ctx;
    // Ask for the order that would ordinarily tempt the search into streaming.
    ctx.required_output.sort = {SortKey{0, false, false, false}};
    const LoweringResult r = lower(*logical, ctx);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "HashAggregate"));
    CHECK(!contains(s, "StreamingAggregate"));

    using db25::physical::PhysicalOp;
    CHECK(!db25::physical::is_applicable(PhysicalOp::StreamingAggregate, {},
                                         db25::ast::JoinType::Inner,
                                         db25::physical::GroupingSpec{1, false}));
    CHECK(db25::physical::is_applicable(PhysicalOp::StreamingAggregate, {},
                                        db25::ast::JoinType::Inner,
                                        db25::physical::GroupingSpec{1, true}));
    CHECK(db25::physical::is_applicable(PhysicalOp::HashAggregate, {},
                                        db25::ast::JoinType::Inner,
                                        db25::physical::GroupingSpec{1, false}));
}

// A scalar aggregate - SELECT COUNT(*) FROM t - returns exactly ONE row, however
// large t is. Estimating it as a fraction of the input would be wrong rather than
// imprecise, and every operator above it would inherit the error.
static void test_a_scalar_aggregate_estimates_one_row() {
    std::printf("test_a_scalar_aggregate_estimates_one_row\n");
    const db25::physical::CardinalityModel card;
    const double in_rows[1] = {1000.0};
    const auto rows = [&](db25::physical::GroupingSpec g) {
        return operator_rows(db25::physical::PhysicalOp::HashAggregate,
                             std::span<const double>{in_rows, 1}, "", card, {}, g);
    };
    CHECK(rows(db25::physical::GroupingSpec{0, false}) == 1.0);
    CHECK(rows(db25::physical::GroupingSpec{1, true}) == 1000.0 * card.group_selectivity);
    // The ALGORITHM cannot change how many rows come out, exactly as for the joins.
    CHECK(operator_rows(db25::physical::PhysicalOp::StreamingAggregate,
                        std::span<const double>{in_rows, 1}, "", card, {},
                        db25::physical::GroupingSpec{0, true}) == 1.0);

    auto logical = build_aggregate(false, /*scalar=*/true);
    const LoweringResult r = lower(*logical);
    CHECK(r.ok);
    if (r.plan) CHECK(card.rows(*r.plan) == 1.0);
}

// ROLLUP / CUBE / GROUPING SETS ask for several grouping-key combinations at
// once. Neither operator computes that, and treating it as a plain GROUP BY would
// return the WRONG ROWS while reporting success - the failure mode this project
// keeps finding and keeps refusing to ship.
static void test_grouping_sets_fail_rather_than_silently_flatten() {
    std::printf("test_grouping_sets_fail_rather_than_silently_flatten\n");
    auto logical = build_aggregate(false, false, /*grouping_sets=*/true);
    const LoweringResult r = lower(*logical);
    CHECK(!r.ok);
    CHECK(r.plan == nullptr);
    CHECK(contains(r.error, "grouping sets"));
}

int main() {
    test_lowers_the_increment0_query();
    test_keyless_join_never_merges();
    test_non_equi_join_is_a_nested_loop();
    test_nested_loop_never_wins_an_equi_join();
    test_freshness_overrides_cost();
    test_no_fresh_copy_is_an_honest_failure();
    test_storage_substrate_is_chosen_by_cost();
    test_join_requires_rows_and_conversion_is_priced();
    test_join_algorithm_is_chosen_by_cost();
    test_cost_chooses_between_candidates();
    test_spec_rules_and_builtin_agree();
    test_non_equi_join_keeps_residual();
    test_equi_key_survives_an_extra_conjunct();
    test_multiple_keys_and_same_side_equality();
    test_required_order_selects_the_merge_join();
    test_root_enforcer_is_inserted();
    test_unsatisfiable_required_output_fails_honestly();
    test_pruning_never_changes_the_plan();
    test_identical_subtrees_share_a_group();
    test_non_equivalent_subtrees_never_share();
    test_search_budget_comes_from_the_spec();
    test_budget_guard_engages_and_is_reported();
    test_guard_changes_what_the_search_finds();
    test_guarded_search_is_deterministic();
    test_unsupported_operator_is_an_error();
    test_join_kind_reaches_the_physical_plan();
    test_inner_and_left_are_not_the_same_group();
    test_lateral_join_is_never_a_hash_join();
    test_order_by_and_limit_lower();
    test_nulls_ordering_survives_lowering();
    test_a_lowered_sort_provides_its_order();
    test_limit_never_pushes_an_order_requirement_down();
    test_limit_bounds_the_row_estimate();
    test_a_computed_sort_key_fails_honestly();
    test_sorts_and_limits_are_not_interchangeable_groups();
    test_group_by_lowers();
    test_aggregate_algorithm_is_chosen_by_cost();
    test_a_computed_grouping_key_still_lowers_via_hash();
    test_a_scalar_aggregate_estimates_one_row();
    test_grouping_sets_fail_rather_than_silently_flatten();

    if (g_failures == 0) {
        std::printf("lowering tests: all passed\n");
        return 0;
    }
    std::printf("lowering tests: %d failure(s)\n", g_failures);
    return 1;
}
