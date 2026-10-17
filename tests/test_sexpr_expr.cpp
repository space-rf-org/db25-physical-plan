// The s-expression writer's expression renderer: every operator that carries its
// meaning in a FLAG or an ENUM rather than in its children must render
// differently from the operator it would otherwise be confused with.
//
// WHY THIS IS A TEST AND NOT A FORMATTING PREFERENCE. physical_to_sexpr is what
// the staged goldens (corpus/staged/*.fixture in the umbrella) are made of. A
// golden that renders `age IS NULL` and `age IS NOT NULL` as the same text
// cannot fail when the lowering of one silently becomes the other - the golden
// stays green and the plan is wrong. The Aggregate and WindowFunction cases in
// src/sexpr.cpp were written to close exactly that hole for DISTINCT, FILTER and
// OVER; IsNull, Like, InList, BooleanTest, Cast and Parameter were left in the
// generic `(<kind> <child>...)` branch, which kept it open for six more.
//
// So the assertions come in pairs, and the load-bearing half of each pair is the
// INEQUALITY: two different operators, two different renderings.
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/sexpr.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"

#include <cstdio>
#include <memory>
#include <string>

using db25::ast::DataType;
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

namespace {

bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

plan::ExprPtr col(std::uint32_t index) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    e->input_index = index;
    e->type = DataType::Integer;
    return e;
}

plan::ExprPtr lit_int(std::int64_t v) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::Literal);
    e->value.value = v;
    e->type = DataType::Integer;
    return e;
}

plan::ExprPtr lit_text(const char* v) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::Literal);
    e->value.value = std::string(v);
    e->type = DataType::Text;
    return e;
}

// Render `pred` the way a plan would: as a Filter's predicate over a scan. Going
// through physical_to_sexpr rather than calling an internal renderer is the
// point - this is the function the goldens are written by.
std::string render(const plan::Expr& pred) {
    auto scan = std::make_unique<db25::physical::PhysicalNode>(PhysicalOp::SeqScan);
    scan->table_name = "t";

    auto filter = std::make_unique<db25::physical::PhysicalNode>(PhysicalOp::Filter);
    filter->predicate = &pred;
    filter->children.push_back(std::move(scan));

    return physical_to_sexpr(*filter);
}

void test_is_null_and_is_not_null_differ() {
    std::printf("test_is_null_and_is_not_null_differ\n");
    auto positive = std::make_unique<plan::Expr>(plan::ExprKind::IsNull);
    positive->children.push_back(col(1));

    auto negated = std::make_unique<plan::Expr>(plan::ExprKind::IsNull);
    negated->children.push_back(col(1));
    negated->expr_flags |= plan::ExprFlagNegated;

    const std::string p = render(*positive);
    const std::string n = render(*negated);
    CHECK(p != n);
    CHECK(contains(p, "(IsNull (col #1))"));
    CHECK(contains(n, "(IsNotNull (col #1))"));
}

void test_the_four_like_operators_differ() {
    std::printf("test_the_four_like_operators_differ\n");
    // LIKE / NOT LIKE / ILIKE / NOT ILIKE are four operators sharing one
    // ExprKind, discriminated by two flag bits.
    std::string seen[4];
    for (int i = 0; i < 4; ++i) {
        auto e = std::make_unique<plan::Expr>(plan::ExprKind::Like);
        e->children.push_back(col(1));
        e->children.push_back(lit_text("a%"));
        if (i & 1) e->expr_flags |= plan::ExprFlagNegated;
        if (i & 2) e->expr_flags |= plan::ExprFlagCaseInsensitive;
        seen[i] = render(*e);
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            CHECK(seen[i] != seen[j]);
        }
    }
    CHECK(contains(seen[0], "(Like "));
    CHECK(contains(seen[1], "(NotLike "));
    CHECK(contains(seen[2], "(ILike "));
    CHECK(contains(seen[3], "(NotILike "));
}

void test_in_and_not_in_differ() {
    std::printf("test_in_and_not_in_differ\n");
    auto positive = std::make_unique<plan::Expr>(plan::ExprKind::InList);
    positive->children.push_back(col(1));
    positive->children.push_back(lit_int(1));
    positive->children.push_back(lit_int(2));

    auto negated = std::make_unique<plan::Expr>(plan::ExprKind::InList);
    negated->children.push_back(col(1));
    negated->children.push_back(lit_int(1));
    negated->children.push_back(lit_int(2));
    negated->expr_flags |= plan::ExprFlagNegated;

    CHECK(render(*positive) != render(*negated));
    CHECK(contains(render(*positive), "(InList (col #1) (lit 1) (lit 2))"));
    CHECK(contains(render(*negated), "(NotInList (col #1) (lit 1) (lit 2))"));
}

void test_all_six_boolean_tests_differ() {
    std::printf("test_all_six_boolean_tests_differ\n");
    // IS TRUE / IS FALSE / IS UNKNOWN and their three NOT forms: six operators,
    // one ExprKind, discriminated by an enum and a flag.
    const plan::BoolTest targets[3] = {plan::BoolTest::True, plan::BoolTest::False,
                                       plan::BoolTest::Unknown};
    std::string seen[6];
    for (int t = 0; t < 3; ++t) {
        for (int neg = 0; neg < 2; ++neg) {
            auto e = std::make_unique<plan::Expr>(plan::ExprKind::BooleanTest);
            e->children.push_back(col(1));
            e->bool_test = targets[t];
            if (neg != 0) e->expr_flags |= plan::ExprFlagNegated;
            seen[t * 2 + neg] = render(*e);
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = i + 1; j < 6; ++j) {
            CHECK(seen[i] != seen[j]);
        }
    }
    CHECK(contains(seen[0], ":is true"));
    CHECK(contains(seen[1], ":is-not true"));
    CHECK(contains(seen[4], ":is unknown"));
}

void test_casts_to_different_types_differ() {
    std::printf("test_casts_to_different_types_differ\n");
    // Nested in a predicate there is no out= schema to recover the target type
    // from, so the render has to carry it.
    auto to_text = std::make_unique<plan::Expr>(plan::ExprKind::Cast);
    to_text->children.push_back(col(1));
    to_text->target_type = DataType::Text;
    to_text->type = DataType::Text;

    auto to_bigint = std::make_unique<plan::Expr>(plan::ExprKind::Cast);
    to_bigint->children.push_back(col(1));
    to_bigint->target_type = DataType::BigInt;
    to_bigint->type = DataType::BigInt;

    CHECK(render(*to_text) != render(*to_bigint));
    CHECK(contains(render(*to_text), "(Cast :to Text (col #1))"));
    CHECK(contains(render(*to_bigint), "(Cast :to BigInt (col #1))"));
}

void test_different_parameters_differ() {
    std::printf("test_different_parameters_differ\n");
    auto one = std::make_unique<plan::Expr>(plan::ExprKind::Parameter);
    one->param_index = 1;
    auto two = std::make_unique<plan::Expr>(plan::ExprKind::Parameter);
    two->param_index = 2;

    CHECK(render(*one) != render(*two));
    CHECK(contains(render(*one), "(param $1)"));
    CHECK(contains(render(*two), "(param $2)"));
}

void test_subquery_kinds_differ() {
    std::printf("test_subquery_kinds_differ\n");
    // Today's optimizer decorrelates these into semi / anti / apply joins, so a
    // Subquery expression does not survive lowering. Asserted anyway: the render
    // must not depend on an optimizer behaviour for its correctness, and the day
    // one does survive, "which kind was it" has to still be in the text.
    const plan::SubqueryKind kinds[4] = {
        plan::SubqueryKind::Scalar, plan::SubqueryKind::In,
        plan::SubqueryKind::Exists, plan::SubqueryKind::Quantified};
    std::string seen[8];
    for (int k = 0; k < 4; ++k) {
        for (int neg = 0; neg < 2; ++neg) {
            auto e = std::make_unique<plan::Expr>(plan::ExprKind::Subquery);
            e->subquery_kind = kinds[k];
            if (neg != 0) e->expr_flags |= plan::ExprFlagNegated;
            seen[k * 2 + neg] = render(*e);
        }
    }
    for (int i = 0; i < 8; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            CHECK(seen[i] != seen[j]);
        }
    }
    CHECK(contains(seen[4], ":kind exists"));
    CHECK(contains(seen[5], ":not"));
}

void test_quantified_any_and_all_differ() {
    std::printf("test_quantified_any_and_all_differ\n");
    auto any = std::make_unique<plan::Expr>(plan::ExprKind::Subquery);
    any->subquery_kind = plan::SubqueryKind::Quantified;
    any->bin_op = db25::ast::BinaryOp::GreaterThan;

    auto all = std::make_unique<plan::Expr>(plan::ExprKind::Subquery);
    all->subquery_kind = plan::SubqueryKind::Quantified;
    all->bin_op = db25::ast::BinaryOp::GreaterThan;
    all->expr_flags |= plan::ExprFlagQuantAll;

    CHECK(render(*any) != render(*all));
    CHECK(contains(render(*any), ":sense any"));
    CHECK(contains(render(*all), ":sense all"));
}

void test_a_negated_kind_in_the_generic_branch_still_shows_it() {
    std::printf("test_a_negated_kind_in_the_generic_branch_still_shows_it\n");
    // Between has no negated form today, so it falls to the generic branch. If a
    // kind there ever acquires one, the flag must appear rather than vanish -
    // that is what keeps this class of bug from coming back through a kind
    // nobody thought to add a case for.
    auto positive = std::make_unique<plan::Expr>(plan::ExprKind::Between);
    positive->children.push_back(col(1));
    positive->children.push_back(lit_int(1));
    positive->children.push_back(lit_int(5));

    auto negated = std::make_unique<plan::Expr>(plan::ExprKind::Between);
    negated->children.push_back(col(1));
    negated->children.push_back(lit_int(1));
    negated->children.push_back(lit_int(5));
    negated->expr_flags |= plan::ExprFlagNegated;

    CHECK(render(*positive) != render(*negated));
    CHECK(contains(render(*negated), ":not"));
}

}  // namespace

int main() {
    std::printf("=== sexpr expression discriminators ===\n");
    test_is_null_and_is_not_null_differ();
    test_the_four_like_operators_differ();
    test_in_and_not_in_differ();
    test_all_six_boolean_tests_differ();
    test_casts_to_different_types_differ();
    test_different_parameters_differ();
    test_subquery_kinds_differ();
    test_quantified_any_and_all_differ();
    test_a_negated_kind_in_the_generic_branch_still_shows_it();
    if (g_failures != 0) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("all checks passed\n");
    return 0;
}
