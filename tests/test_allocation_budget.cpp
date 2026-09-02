// The guardrail this repo was missing.
//
// Unit 2.1 nearly doubled T6 while changing no plans, and it was merged because
// NOTHING MEASURED IT. T6 lives in a design document, not in a build. The obvious
// fix - assert a wall-clock ceiling - is the wrong one: absolute times are not
// comparable across machines, and a threshold loose enough for the slowest CI
// runner would not have caught the very regression that motivated this.
//
// Heap allocations per lower() are a better invariant. They are DETERMINISTIC and
// machine-independent, they were the actual mechanism of every regression this
// planner has had, and a budget on them is stable enough to fail a build without
// being flaky. This test would have caught 2.1 on the commit that introduced it.
//
// When a change legitimately needs more allocations, the budget moves - in a
// commit that says so, which is the point.
#include "db25/physical/lowering.hpp"
#include "db25/physical/spec.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>

namespace plan = db25::plan;
using db25::ast::BinaryOp;
using db25::ast::DataType;
using namespace db25::physical;

namespace {
bool g_counting = false;
long g_allocations = 0;
}  // namespace

// Replacing global operator new is safe here because this is its own test binary.
void* operator new(std::size_t n) {
    if (g_counting) ++g_allocations;
    void* p = std::malloc(n != 0 ? n : 1);
    return p;
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void* operator new[](std::size_t n) {
    if (g_counting) ++g_allocations;
    return std::malloc(n != 0 ? n : 1);
}
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

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

static plan::ExprPtr col(std::uint32_t i, DataType t = DataType::Integer) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    e->input_index = i;
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

// The reference query, the same one the umbrella's T6 stage times:
//   SELECT a.x, b.y FROM a JOIN b ON a.id = b.id WHERE a.x > 10
static plan::LogicalNodePtr reference_query() {
    auto sa = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    sa->table_name = "a";
    sa->output = {{"id", DataType::Integer, false}, {"x", DataType::Integer, true}};
    auto sb = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    sb->table_name = "b";
    sb->output = {{"id", DataType::Integer, false}, {"y", DataType::VarChar, true}};
    auto j = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
    j->join_type = db25::ast::JoinType::Inner;
    j->output = {{"id", DataType::Integer, false},
                 {"x", DataType::Integer, true},
                 {"id", DataType::Integer, false},
                 {"y", DataType::VarChar, true}};
    j->predicate = binop(BinaryOp::Equal, col(0), col(2));
    j->add_child(std::move(sa));
    j->add_child(std::move(sb));
    auto f = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
    f->output = j->output;
    f->predicate = binop(BinaryOp::GreaterThan, col(1), int_lit(10));
    f->add_child(std::move(j));
    auto p = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Project);
    p->output = {{"x", DataType::Integer, true}, {"y", DataType::VarChar, true}};
    p->exprs.push_back(col(1));
    p->exprs.push_back(col(3, DataType::VarChar));
    p->add_child(std::move(f));
    return p;
}

// Measured budgets, deliberately TIGHT. The count is deterministic for a given
// toolchain, so headroom buys nothing except insensitivity: a first attempt set
// these ~29% above the measured value and a real 8-allocation regression slid
// underneath. A few slots absorb an incidental container change; anything larger
// should fail and be looked at.
//
// A legitimate increase means editing these numbers in a commit that explains
// why. That is the mechanism, not an obstacle to it.
// Raised by 5 in Increment 3.8a, which moved `inputs`, `hash_keys` and `residual`
// from the Group to each GroupExpr. The cost is exact and was measured by removing
// each copy in turn: the join group has FOUR candidates (HashJoin building either
// side, MergeJoin, NestedLoopJoin) and each now owns its own copy of the join
// keys, where all four previously shared one on the group. `residual` costs
// nothing here - it is empty in this query, and an empty vector copy does not
// allocate - and `inputs` is an ArityVec, stored inline.
//
// That is the model being right rather than waste: a candidate is (operator, input
// groups), and join reordering needs two candidates in one group to read different
// inputs and join on different keys. Paying four allocations for it on a five-group
// query is the correct trade, and it is recorded here rather than absorbed.
//
// The obvious saving - a small-vector with inline capacity for hash keys, which
// are one or two columns in almost every join - is a separate change and tracked
// as one.
constexpr long kBudgetDefault = 74;   // measured 70 (was 65 before Increment 3.8a)
constexpr long kBudgetDedup = 89;     // measured 85 (dedup builds a key per group)

static long allocations_for(const LoweringContext& ctx, const plan::LogicalNode& q) {
    { const LoweringResult warm = lower(q, ctx); (void)warm; }  // one-off caches
    g_allocations = 0;
    g_counting = true;
    const LoweringResult r = lower(q, ctx);
    g_counting = false;
    if (!r.ok) {
        std::printf("  lowering failed: %s\n", r.error.c_str());
        return -1;
    }
    return g_allocations;
}

int main() {
    std::string error;
    auto spec = load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    if (!spec) {
        std::printf("spec load failed: %s\n", error.c_str());
        return 1;
    }
    auto query = reference_query();

    std::printf("test_allocation_budget\n");
    {
        LoweringContext ctx;
        ctx.spec = &*spec;
        const long n = allocations_for(ctx, *query);
        std::printf("  lower() allocations (default)      = %ld  (budget %ld)\n", n,
                    kBudgetDefault);
        CHECK(n >= 0);
        CHECK(n <= kBudgetDefault);
    }
    {
        LoweringContext ctx;
        ctx.spec = &*spec;
        ctx.dedup = true;
        const long n = allocations_for(ctx, *query);
        std::printf("  lower() allocations (dedup on)     = %ld  (budget %ld)\n", n,
                    kBudgetDedup);
        CHECK(n >= 0);
        CHECK(n <= kBudgetDedup);
    }

    if (g_failures == 0) {
        std::printf("allocation budget: within budget\n");
        return 0;
    }
    std::printf("allocation budget: %d failure(s)\n", g_failures);
    return 1;
}
