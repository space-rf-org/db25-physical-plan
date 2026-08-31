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

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <utility>

using db25::ast::BinaryOp;
using db25::ast::DataType;
using db25::physical::lower;
using db25::physical::LoweringContext;
using db25::physical::LoweringResult;
using db25::physical::physical_to_sexpr;
using db25::physical::PhysicalOp;
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
static plan::LogicalNodePtr build_logical(plan::ExprPtr join_pred = nullptr) {
    auto scan_a = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan_a->table_name = "a";
    scan_a->output = a_schema();

    auto scan_b = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan_b->table_name = "b";
    scan_b->output = b_schema();

    auto join = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
    join->join_type = db25::ast::JoinType::Inner;
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
    CHECK(contains(s, "(HashJoin keys=[(L#0 R#0)]"));  // a.id=b.id -> key pair
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
    CHECK(contains(s, "keys=[]"));       // no equi key extracted
    CHECK(contains(s, "residual="));     // the predicate is kept as a residual
}

static void test_unsupported_operator_is_an_error() {
    std::printf("test_unsupported_operator_is_an_error\n");
    auto scan = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    scan->table_name = "a";
    scan->output = a_schema();
    auto agg = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Aggregate);  // not in Inc 0
    agg->output = a_schema();
    agg->add_child(std::move(scan));

    const LoweringResult r = lower(*agg);
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

int main() {
    test_lowers_the_increment0_query();
    test_cost_chooses_between_candidates();
    test_spec_rules_and_builtin_agree();
    test_non_equi_join_keeps_residual();
    test_unsupported_operator_is_an_error();

    if (g_failures == 0) {
        std::printf("lowering tests: all passed\n");
        return 0;
    }
    std::printf("lowering tests: %d failure(s)\n", g_failures);
    return 1;
}
