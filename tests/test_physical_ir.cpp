// Unit 0.2: the physical IR, the Cascades memo scaffold, and s-expr rendering.
//
// Builds the Increment-0 shape by hand - the simple query
//   SELECT a.x, b.y FROM a JOIN b ON a.id = b.id WHERE a.x > 10
// as a physical plan (SeqScan/SeqScan -> HashJoin -> Filter -> Project) - and
// checks: (1) the render carries the structural facts, (2) a memo built to the
// same shape extracts to an identical render (memo path == direct path), and
// (3) a structural mutation changes the render (falsifiability).
#include "db25/physical/memo.hpp"
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/sexpr.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace db25::physical;
using db25::ast::BinaryOp;
using db25::ast::DataType;
using db25::plan::Expr;
using db25::plan::ExprKind;
using db25::plan::Schema;

static int g_failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::printf("  FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

// Owned expressions for the plan under test. Physical nodes borrow these, so the
// store must outlive every plan built from it.
struct ExprStore {
    std::vector<std::unique_ptr<Expr>> owned;

    Expr* col(std::uint32_t index, DataType type) {
        auto e = std::make_unique<Expr>(ExprKind::ColumnRef);
        e->input_index = index;
        e->type = type;
        Expr* p = e.get();
        owned.push_back(std::move(e));
        return p;
    }
    Expr* int_lit(std::int64_t v) {
        auto e = std::make_unique<Expr>(ExprKind::Literal);
        e->type = DataType::Integer;
        e->value.value = v;
        Expr* p = e.get();
        owned.push_back(std::move(e));
        return p;
    }
    Expr* binop(BinaryOp op, Expr* lhs, Expr* rhs) {
        auto e = std::make_unique<Expr>(ExprKind::BinaryOp);
        e->bin_op = op;
        // Re-own lhs/rhs under this node: the store keeps the flat list alive,
        // but BinaryOp needs its operands as owned children for rendering.
        e->children.push_back(take(lhs));
        e->children.push_back(take(rhs));
        Expr* p = e.get();
        owned.push_back(std::move(e));
        return p;
    }

private:
    // Move a previously-stored expr out of the flat store into a child slot.
    std::unique_ptr<Expr> take(Expr* p) {
        for (auto& u : owned) {
            if (u.get() == p) {
                auto out = std::move(u);
                u = nullptr;
                return out;
            }
        }
        return nullptr;  // not found (shouldn't happen in these tests)
    }
};

static Schema a_schema() {
    return {{"id", DataType::Integer, false}, {"x", DataType::Integer, true}};
}
static Schema b_schema() {
    return {{"id", DataType::Integer, false}, {"y", DataType::VarChar, true}};
}
static Schema join_schema() {
    return {{"id", DataType::Integer, false},
            {"x", DataType::Integer, true},
            {"id", DataType::Integer, false},
            {"y", DataType::VarChar, true}};
}
static Schema proj_schema() {
    return {{"x", DataType::Integer, true}, {"y", DataType::VarChar, true}};
}

// Build the plan directly (owning tree). `pred`, `px`, `py` are borrowed.
static PhysicalNodePtr build_plan(const Expr* pred, const Expr* px, const Expr* py,
                                  std::vector<HashKey> keys) {
    auto scan_a = make_seq_scan("a", a_schema());
    auto scan_b = make_seq_scan("b", b_schema());
    auto join = make_hash_join(std::move(scan_a), std::move(scan_b), std::move(keys),
                               join_schema());
    auto filter = make_filter(std::move(join), pred);
    return make_project(std::move(filter), proj_schema(), {px, py});
}

static void test_render_carries_structure() {
    std::printf("test_render_carries_structure\n");
    ExprStore es;
    const Expr* pred = es.binop(BinaryOp::GreaterThan, es.col(1, DataType::Integer),
                                es.int_lit(10));
    const Expr* px = es.col(1, DataType::Integer);
    const Expr* py = es.col(3, DataType::VarChar);

    auto plan = build_plan(pred, px, py, {{0, 0}});
    const std::string s = physical_to_sexpr(*plan);

    // Operators, in nesting order.
    CHECK(contains(s, "(Project "));
    CHECK(contains(s, "(Filter "));
    CHECK(contains(s, "(HashJoin "));
    CHECK(contains(s, "(SeqScan table=a"));
    CHECK(contains(s, "(SeqScan table=b"));
    // Structural payloads (spelling-independent of the DataType printer).
    CHECK(contains(s, "keys=[(L#0 R#0)]"));
    CHECK(contains(s, "exprs=[(col #1) (col #3)]"));
    CHECK(contains(s, "pred=("));
    CHECK(contains(s, "(col #1)"));
    CHECK(contains(s, "(lit 10)"));
}

static void test_memo_extracts_to_same_render() {
    std::printf("test_memo_extracts_to_same_render\n");
    ExprStore es;
    const Expr* pred = es.binop(BinaryOp::GreaterThan, es.col(1, DataType::Integer),
                                es.int_lit(10));
    const Expr* px = es.col(1, DataType::Integer);
    const Expr* py = es.col(3, DataType::VarChar);

    const std::string direct = physical_to_sexpr(*build_plan(pred, px, py, {{0, 0}}));

    // Same shape, built through the memo.
    Memo m;
    const GroupId g_a = m.add_group(a_schema());
    m.add_expr(g_a, GroupExpr{PhysicalOp::SeqScan, {}, "a", nullptr, {}, {}, 0.0});
    m.set_winner(g_a, 0);

    const GroupId g_b = m.add_group(b_schema());
    m.add_expr(g_b, GroupExpr{PhysicalOp::SeqScan, {}, "b", nullptr, {}, {}, 0.0});
    m.set_winner(g_b, 0);

    const GroupId g_join = m.add_group(join_schema());
    m.add_expr(g_join,
               GroupExpr{PhysicalOp::HashJoin, {g_a, g_b}, "", nullptr, {}, {{0, 0}}, 0.0});
    m.set_winner(g_join, 0);

    const GroupId g_filter = m.add_group(join_schema());
    m.add_expr(g_filter,
               GroupExpr{PhysicalOp::Filter, {g_join}, "", pred, {}, {}, 0.0});
    m.set_winner(g_filter, 0);

    const GroupId g_proj = m.add_group(proj_schema());
    m.add_expr(g_proj,
               GroupExpr{PhysicalOp::Project, {g_filter}, "", nullptr, {px, py}, {}, 0.0});
    m.set_winner(g_proj, 0);
    m.set_root(g_proj);

    auto extracted = m.extract_winner();
    CHECK(extracted != nullptr);
    if (extracted) {
        CHECK(physical_to_sexpr(*extracted) == direct);
    }
    CHECK(m.size() == 5);
    CHECK(m.group(g_proj).best_cost() == 0.0);
}

static void test_extract_without_winner_fails() {
    std::printf("test_extract_without_winner_fails\n");
    Memo m;
    const GroupId g = m.add_group(a_schema());
    m.add_expr(g, GroupExpr{PhysicalOp::SeqScan, {}, "a", nullptr, {}, {}, 0.0});
    // No winner set.
    m.set_root(g);
    CHECK(m.extract_winner() == nullptr);
}

static void test_falsifiability_mutation_changes_render() {
    std::printf("test_falsifiability_mutation_changes_render\n");
    ExprStore es;
    const Expr* pred = es.binop(BinaryOp::GreaterThan, es.col(1, DataType::Integer),
                                es.int_lit(10));
    const Expr* px = es.col(1, DataType::Integer);
    const Expr* py = es.col(3, DataType::VarChar);

    const std::string base = physical_to_sexpr(*build_plan(pred, px, py, {{0, 0}}));

    // Mutant A: different equi-join key.
    ExprStore es2;
    const Expr* pred2 = es2.binop(BinaryOp::GreaterThan, es2.col(1, DataType::Integer),
                                  es2.int_lit(10));
    const Expr* px2 = es2.col(1, DataType::Integer);
    const Expr* py2 = es2.col(3, DataType::VarChar);
    const std::string mut_key = physical_to_sexpr(*build_plan(pred2, px2, py2, {{1, 1}}));
    CHECK(base != mut_key);

    // Mutant B: different literal in the predicate.
    ExprStore es3;
    const Expr* pred3 = es3.binop(BinaryOp::GreaterThan, es3.col(1, DataType::Integer),
                                  es3.int_lit(20));
    const Expr* px3 = es3.col(1, DataType::Integer);
    const Expr* py3 = es3.col(3, DataType::VarChar);
    const std::string mut_lit = physical_to_sexpr(*build_plan(pred3, px3, py3, {{0, 0}}));
    CHECK(base != mut_lit);
}

int main() {
    test_render_carries_structure();
    test_memo_extracts_to_same_render();
    test_extract_without_winner_fails();
    test_falsifiability_mutation_changes_render();

    if (g_failures == 0) {
        std::printf("physical IR tests: all passed\n");
        return 0;
    }
    std::printf("physical IR tests: %d failure(s)\n", g_failures);
    return 1;
}
