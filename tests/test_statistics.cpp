// Increment 6.1: per-column statistics, and the selectivity estimates that read
// them.
//
// Two things are under test and they are different. The LOADER must reject a
// file it does not understand rather than fall back silently - a statistics file
// that loads as nothing produces the un-statted plan and says nothing about why.
// The ESTIMATOR must answer predicates from the statistics where it can, and
// fall back BY PREDICATE CLASS where it cannot; the property that matters most
// is that an empty catalog reproduces the flat-constant model exactly, because
// that is what lets this land without moving a plan.
#include "db25/physical/cost.hpp"
#include "db25/physical/selectivity.hpp"
#include "db25/physical/statistics.hpp"
#include "db25/physical/physical_plan.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

using namespace db25::physical;
using db25::ast::BinaryOp;
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
static bool approx(double a, double b) { return std::fabs(a - b) < 1e-9; }

#ifndef DB25_PHYSICAL_SPEC_DIR
#define DB25_PHYSICAL_SPEC_DIR "."
#endif
static std::string lab() { return std::string(DB25_PHYSICAL_SPEC_DIR) + "/statistics.lab.sexpr"; }

// ---- expression builders --------------------------------------------------
static plan::ExprPtr col(std::uint32_t table_id, std::uint32_t column_id) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    e->type = DataType::Integer;
    e->ref_table_id = table_id;
    e->ref_column_id = column_id;
    return e;
}
static plan::ExprPtr lit(std::int64_t v) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::Literal);
    e->type = DataType::Integer;
    e->value.value = v;
    return e;
}
static plan::ExprPtr bin(BinaryOp op, plan::ExprPtr l, plan::ExprPtr r) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::BinaryOp);
    e->bin_op = op;
    e->children.push_back(std::move(l));
    e->children.push_back(std::move(r));
    return e;
}
static plan::ExprPtr is_null(plan::ExprPtr v, bool negated) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::IsNull);
    e->children.push_back(std::move(v));
    if (negated) e->expr_flags |= plan::ExprFlagNegated;
    return e;
}

static StatisticsCatalog emp_stats() {
    StatisticsCatalog c;
    TableStats t;
    t.rows = 20000;
    ColumnStats id;    id.ndv = 20000; id.has_range = true; id.min_value = 1; id.max_value = 20000; id.unique = true;
    ColumnStats dept;  dept.ndv = 50;  dept.has_range = true; dept.min_value = 0; dept.max_value = 49;
    ColumnStats sal;   sal.ndv = 20000; sal.has_range = true; sal.min_value = 0; sal.max_value = 19999;
    ColumnStats note;  note.ndv = 4;   note.null_fraction = 0.25;
    t.columns[1] = id;
    t.columns[3] = dept;
    t.columns[4] = sal;
    t.columns[5] = note;
    c.add_table(1, std::move(t));
    return c;
}

// ---- the loader -----------------------------------------------------------
static void test_lab_statistics_load() {
    std::printf("test_lab_statistics_load\n");
    std::string err;
    auto c = load_statistics(lab(), err);
    CHECK(c.has_value());
    if (!c) { std::printf("  load error: %s\n", err.c_str()); return; }
    CHECK(c->size() == 2);
    const TableStats* emp = c->table(1);
    CHECK(emp != nullptr && approx(emp->rows, 20000));
    const ColumnStats* dept = c->column(1, 3);
    CHECK(dept != nullptr);
    if (dept) {
        CHECK(approx(dept->ndv, 50));
        CHECK(dept->has_range && approx(dept->min_value, 0) && approx(dept->max_value, 49));
        CHECK(!dept->unique);
    }
    const ColumnStats* id = c->column(1, 1);
    CHECK(id != nullptr && id->unique);
    // A column with no (min ...) has no range, and that is distinguishable from
    // a column whose range happens to start at zero.
    const ColumnStats* name = c->column(1, 2);
    CHECK(name != nullptr && !name->has_range);
    // Asking for something not in the file is a null, not a zero-valued entry.
    CHECK(c->column(1, 99) == nullptr);
    CHECK(c->column(99, 1) == nullptr);
}

static void test_loader_rejects_rather_than_shrugs() {
    std::printf("test_loader_rejects_rather_than_shrugs\n");
    const struct { const char* text; const char* what; } bad[] = {
        {"(stats (table (id 1)))", "wrong top-level head"},
        {"(statistics (table (rows 10)))", "table with no id"},
        {"(statistics (table (id 0) (rows 10)))", "table id zero"},
        {"(statistics (table (id 1) (column (ndv 5))))", "column with no id"},
        {"(statistics (table (id 1) (ndvv 5)))", "unknown key on table"},
        {"(statistics (table (id 1) (column (id 1) (distinct 5))))", "unknown key on column"},
        {"(statistics (table (id 1) (column (id 1) (min 0))))", "min without max"},
        {"(statistics (table (id 1) (column (id 1) (max 9))))", "max without min"},
        {"(statistics (table (id 1) (column (id 1) (min 9) (max 0))))", "max below min"},
    };
    for (const auto& b : bad) {
        const std::string path = std::string(DB25_PHYSICAL_SPEC_DIR) + "/._stats_tmp.sexpr";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) { CHECK(false); return; }
        std::fputs(b.text, f);
        std::fclose(f);
        std::string err;
        auto c = load_statistics(path, err);
        if (c.has_value()) std::printf("  ACCEPTED bad input (%s): %s\n", b.what, b.text);
        CHECK(!c.has_value());
        CHECK(!err.empty());
        std::remove(path.c_str());
    }
    std::string err;
    CHECK(!load_statistics("/nonexistent/statistics.sexpr", err).has_value());
    CHECK(!err.empty());
}

// ---- the estimator --------------------------------------------------------
static void test_equality_uses_ndv() {
    std::printf("test_equality_uses_ndv\n");
    const StatisticsCatalog s = emp_stats();
    const SelectivityDefaults d;
    // dept_id = 7 over 50 departments: one in fifty, not the flat tenth.
    auto p = bin(BinaryOp::Equal, col(1, 3), lit(7));
    CHECK(approx(selectivity_of(p.get(), s, d), 1.0 / 50.0));
    // And the complement.
    auto ne = bin(BinaryOp::NotEqual, col(1, 3), lit(7));
    CHECK(approx(selectivity_of(ne.get(), s, d), 1.0 - 1.0 / 50.0));
    // A unique column: one row in twenty thousand.
    auto pk = bin(BinaryOp::Equal, col(1, 1), lit(5));
    CHECK(approx(selectivity_of(pk.get(), s, d), 1.0 / 20000.0));
    // Either spelling of the same predicate.
    auto mirrored = bin(BinaryOp::Equal, lit(7), col(1, 3));
    CHECK(approx(selectivity_of(mirrored.get(), s, d), 1.0 / 50.0));
}

static void test_range_uses_min_max() {
    std::printf("test_range_uses_min_max\n");
    const StatisticsCatalog s = emp_stats();
    const SelectivityDefaults d;
    // salary > 1000 over a 0..19999 domain: 18999/19999 of it. The flat model
    // said 0.1 and the true answer on the benchmark fixture is 0.95.
    auto p = bin(BinaryOp::GreaterThan, col(1, 4), lit(1000));
    const double expect = (19999.0 - 1000.0) / 19999.0;
    CHECK(approx(selectivity_of(p.get(), s, d), expect));
    CHECK(selectivity_of(p.get(), s, d) > 0.94);
    // Mirrored: 1000 < salary is the same predicate.
    auto m = bin(BinaryOp::LessThan, lit(1000), col(1, 4));
    CHECK(approx(selectivity_of(m.get(), s, d), expect));
    // The other direction.
    auto lt = bin(BinaryOp::LessThan, col(1, 4), lit(1000));
    CHECK(approx(selectivity_of(lt.get(), s, d), 1000.0 / 19999.0));
    // Outside the domain clamps rather than going negative or above one.
    auto all = bin(BinaryOp::GreaterThan, col(1, 4), lit(-5));
    auto none = bin(BinaryOp::GreaterThan, col(1, 4), lit(999999));
    CHECK(approx(selectivity_of(all.get(), s, d), 1.0));
    CHECK(selectivity_of(none.get(), s, d) <= 1e-6);
}

static void test_nulls() {
    std::printf("test_nulls\n");
    const StatisticsCatalog s = emp_stats();
    const SelectivityDefaults d;
    auto n = is_null(col(1, 5), false);
    auto nn = is_null(col(1, 5), true);
    CHECK(approx(selectivity_of(n.get(), s, d), 0.25));
    CHECK(approx(selectivity_of(nn.get(), s, d), 0.75));
    // IS NULL and IS NOT NULL must not answer the same number - the NOT is a
    // flag on the node, not a separate kind, so an estimator that ignored it
    // would.
    CHECK(!approx(selectivity_of(n.get(), s, d), selectivity_of(nn.get(), s, d)));
    // A comparison can never match a NULL, so the null fraction bounds it.
    auto eq = bin(BinaryOp::Equal, col(1, 5), lit(2));
    CHECK(approx(selectivity_of(eq.get(), s, d), 0.75 / 4.0));
}

static void test_conjunction_and_disjunction() {
    std::printf("test_conjunction_and_disjunction\n");
    const StatisticsCatalog s = emp_stats();
    const SelectivityDefaults d;
    auto a = bin(BinaryOp::Equal, col(1, 3), lit(7));          // 1/50
    auto b = bin(BinaryOp::GreaterThan, col(1, 4), lit(1000)); // 18999/19999
    const double sa = 1.0 / 50.0;
    const double sb = (19999.0 - 1000.0) / 19999.0;
    auto conj = bin(BinaryOp::And, std::move(a), std::move(b));
    CHECK(approx(selectivity_of(conj.get(), s, d), sa * sb));
    auto a2 = bin(BinaryOp::Equal, col(1, 3), lit(7));
    auto b2 = bin(BinaryOp::GreaterThan, col(1, 4), lit(1000));
    auto disj = bin(BinaryOp::Or, std::move(a2), std::move(b2));
    CHECK(approx(selectivity_of(disj.get(), s, d), 1.0 - (1.0 - sa) * (1.0 - sb)));
    // An OR is never more selective than either side; an AND never less.
    CHECK(selectivity_of(disj.get(), s, d) >= sb);
    CHECK(selectivity_of(conj.get(), s, d) <= sa);
}

static void test_fallbacks_are_by_class() {
    std::printf("test_fallbacks_are_by_class\n");
    const StatisticsCatalog empty;
    const SelectivityDefaults d;
    auto eq = bin(BinaryOp::Equal, col(1, 3), lit(7));
    auto gt = bin(BinaryOp::GreaterThan, col(1, 4), lit(1000));
    auto like = std::make_unique<plan::Expr>(plan::ExprKind::Like);
    // With no statistics at all, an equality and a range must NOT get the same
    // number. One flat constant for both is what this increment replaces.
    CHECK(approx(selectivity_of(eq.get(), empty, d), d.equality));
    CHECK(approx(selectivity_of(gt.get(), empty, d), d.inequality));
    CHECK(!approx(d.equality, d.inequality));
    CHECK(approx(selectivity_of(like.get(), empty, d), d.unknown));
    // A column the catalog does not cover falls back the same way, even when
    // OTHER columns of the same table are covered.
    const StatisticsCatalog s = emp_stats();
    auto unknown_col = bin(BinaryOp::Equal, col(1, 77), lit(7));
    CHECK(approx(selectivity_of(unknown_col.get(), s, d), d.equality));
    // A comparison between two columns is not a constant-selectivity question.
    auto col_vs_col = bin(BinaryOp::GreaterThan, col(1, 4), col(1, 3));
    CHECK(approx(selectivity_of(col_vs_col.get(), s, d), d.inequality));
    // No predicate filters nothing.
    CHECK(approx(selectivity_of(nullptr, s, d), 1.0));
}

static void test_distinct_groups() {
    std::printf("test_distinct_groups\n");
    const StatisticsCatalog s = emp_stats();
    auto dept = col(1, 3);
    auto sal = col(1, 4);
    const plan::Expr* one[1] = {dept.get()};
    const plan::Expr* two[2] = {dept.get(), sal.get()};
    // GROUP BY dept_id is FIFTY groups, not a tenth of the input. This is the
    // single largest estimate error the flat model had: 2000 against 50.
    CHECK(approx(distinct_groups(std::span<const plan::Expr* const>{one, 1}, 20000, s), 50.0));
    // Two keys multiply, CAPPED at the input - grouping cannot make more groups
    // than it has rows, and 50 x 20000 uncapped would claim it can.
    CHECK(approx(distinct_groups(std::span<const plan::Expr* const>{two, 2}, 20000, s), 20000.0));
    // One unknown key makes the whole product unknown rather than half-measured.
    auto unknown = col(1, 77);
    const plan::Expr* mixed[2] = {dept.get(), unknown.get()};
    CHECK(approx(distinct_groups(std::span<const plan::Expr* const>{mixed, 2}, 20000, s), 0.0));
    // No keys is a scalar aggregate, which the caller handles as exactly one row.
    CHECK(approx(distinct_groups(std::span<const plan::Expr* const>{}, 20000, s), 0.0));
}

// ---- the property that lets this land -------------------------------------
static void test_no_statistics_reproduces_the_flat_model() {
    std::printf("test_no_statistics_reproduces_the_flat_model\n");
    CardinalityModel card;                       // stats == nullptr
    CHECK(card.stats == nullptr);
    auto pred = bin(BinaryOp::Equal, col(1, 3), lit(7));
    const FilterSpec fs =
        filter_spec_from(PhysicalOp::Filter, pred.get(), {}, card, 20000);
    // The sentinels, not estimates: a negative selectivity and zero groups are
    // what tell operator_rows to use the flat constants.
    CHECK(fs.selectivity < 0.0);
    CHECK(approx(fs.groups, 0.0));
    const double in[1] = {20000.0};
    CHECK(approx(operator_rows(PhysicalOp::Filter, std::span<const double>{in, 1}, "", card,
                               LimitSpec{}, GroupingSpec{}, db25::ast::SetOp::Union, 0.0,
                               JoinSpec{}, fs),
                 20000.0 * card.filter_selectivity));
    // And an empty catalog is the same as no catalog.
    const StatisticsCatalog empty;
    card.stats = &empty;
    const FilterSpec fs2 = filter_spec_from(PhysicalOp::Filter, pred.get(), {}, card, 20000);
    CHECK(fs2.selectivity < 0.0);
}

static void test_statistics_reach_the_row_estimate() {
    std::printf("test_statistics_reach_the_row_estimate\n");
    const StatisticsCatalog s = emp_stats();
    CardinalityModel card;
    card.stats = &s;
    auto pred = bin(BinaryOp::GreaterThan, col(1, 4), lit(1000));
    const FilterSpec fs =
        filter_spec_from(PhysicalOp::Filter, pred.get(), {}, card, 20000);
    const double in[1] = {20000.0};
    const double rows = operator_rows(PhysicalOp::Filter, std::span<const double>{in, 1}, "",
                                      card, LimitSpec{}, GroupingSpec{}, db25::ast::SetOp::Union,
                                      0.0, JoinSpec{}, fs);
    // 20000 employees, salary > 1000: about 19000 of them. The flat model said
    // 2000, and the true answer on the fixture is 18999.
    CHECK(rows > 18000.0 && rows < 19100.0);

    // And the grouping count reaches the aggregate.
    auto dept = col(1, 3);
    const plan::Expr* keys[1] = {dept.get()};
    const FilterSpec gs = filter_spec_from(PhysicalOp::HashAggregate, nullptr,
                                           std::span<const plan::Expr* const>{keys, 1},
                                           card, 20000);
    CHECK(approx(gs.groups, 50.0));
    const double grows = operator_rows(PhysicalOp::HashAggregate,
                                       std::span<const double>{in, 1}, "", card, LimitSpec{},
                                       GroupingSpec{1, false, false, 0}, db25::ast::SetOp::Union,
                                       0.0, JoinSpec{}, gs);
    CHECK(approx(grows, 50.0));
    // The ALGORITHM still does not change the count.
    CHECK(approx(operator_rows(PhysicalOp::StreamingAggregate, std::span<const double>{in, 1},
                               "", card, LimitSpec{}, GroupingSpec{1, false, false, 0},
                               db25::ast::SetOp::Union, 0.0, JoinSpec{}, gs),
                 grows));
}

static void test_selectivity_is_always_a_fraction() {
    std::printf("test_selectivity_is_always_a_fraction\n");
    // A property rather than a number: whatever the predicate, whatever the
    // statistics, the answer is a fraction. An estimator that returned 1.4 would
    // make a filter produce more rows than it read.
    const StatisticsCatalog s = emp_stats();
    const SelectivityDefaults d;
    std::vector<plan::ExprPtr> preds;
    preds.push_back(bin(BinaryOp::Equal, col(1, 3), lit(7)));
    preds.push_back(bin(BinaryOp::GreaterThan, col(1, 4), lit(-99999)));
    preds.push_back(bin(BinaryOp::LessThan, col(1, 4), lit(99999)));
    preds.push_back(bin(BinaryOp::NotEqual, col(1, 1), lit(1)));
    preds.push_back(is_null(col(1, 5), true));
    preds.push_back(bin(BinaryOp::Or, bin(BinaryOp::Equal, col(1, 3), lit(1)),
                        bin(BinaryOp::GreaterThan, col(1, 4), lit(0))));
    preds.push_back(bin(BinaryOp::And, is_null(col(1, 5), false),
                        bin(BinaryOp::Equal, col(1, 3), lit(1))));
    for (const auto& p : preds) {
        const double v = selectivity_of(p.get(), s, d);
        CHECK(v >= 0.0 && v <= 1.0);
    }
}

// (0, 0) is the binder's mark for a column that is NOT a base-table column - an
// aggregate's output, an expression, a literal. It must never pick up
// statistics, and the loader refuses to write an entry under it. But the
// in-memory API can, and an estimator that looked one up would attribute a real
// column's distinct count to a computed one and be confidently wrong about a
// GROUP BY over an expression.
static void test_a_computed_column_is_never_looked_up() {
    std::printf("test_a_computed_column_is_never_looked_up\n");
    StatisticsCatalog poisoned;
    TableStats t;
    t.rows = 20000;
    ColumnStats cs;
    cs.ndv = 2;                 // a number nothing should ever pick up
    cs.has_range = true;
    cs.min_value = 0;
    cs.max_value = 1;
    t.columns[0] = cs;
    poisoned.add_table(0, std::move(t));

    const SelectivityDefaults d;
    // A ColumnRef with no catalog provenance: the default `col()` without ids.
    auto computed = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    computed->type = DataType::Integer;
    auto p = bin(BinaryOp::Equal, std::move(computed), lit(1));
    // Must fall back, NOT answer 1/2 from the entry sitting at (0, 0).
    CHECK(approx(selectivity_of(p.get(), poisoned, d), d.equality));

    auto key = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    key->type = DataType::Integer;
    const plan::Expr* keys[1] = {key.get()};
    // And a GROUP BY over a computed key must refuse rather than claim two groups.
    CHECK(approx(distinct_groups(std::span<const plan::Expr* const>{keys, 1}, 20000, poisoned),
                 0.0));

    // The loader will not even let such an entry be written down.
    const std::string path = std::string(DB25_PHYSICAL_SPEC_DIR) + "/._stats_zero.sexpr";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f != nullptr) {
        std::fputs("(statistics (table (id 0) (rows 1) (column (id 1) (ndv 1))))", f);
        std::fclose(f);
        std::string err;
        CHECK(!load_statistics(path, err).has_value());
        std::remove(path.c_str());
    }
}

// ---- the wiring, through a real plan --------------------------------------
// The tests above call the estimator directly, which proves the estimator and
// nothing about whether a PLAN reaches it. Both representations have to be
// checked separately and end to end: a mutation that made the tree form pass
// FilterSpec{} survived the entire suite until this existed - the same gap the
// join-cardinality work found, for the same reason.
static void test_the_tree_form_reaches_the_estimator() {
    std::printf("test_the_tree_form_reaches_the_estimator\n");
    const StatisticsCatalog s = emp_stats();
    CardinalityModel card;
    card.base_rows["emp"] = 20000.0;

    auto pred = bin(BinaryOp::GreaterThan, col(1, 4), lit(1000));
    auto plan_node = make_filter(make_seq_scan("emp", {}), pred.get());

    // Without statistics: the flat constant, exactly as before.
    CHECK(approx(card.rows(*plan_node), 20000.0 * card.filter_selectivity));

    // With them: the estimate, through PhysicalNode::predicate.
    card.stats = &s;
    const double with = card.rows(*plan_node);
    CHECK(with > 18000.0 && with < 19100.0);
    CHECK(!approx(with, 20000.0 * card.filter_selectivity));

    // And an aggregate's grouping keys, through PhysicalNode::group_keys.
    auto dept = col(1, 3);
    auto agg = std::make_unique<PhysicalNode>(PhysicalOp::HashAggregate);
    agg->children.push_back(make_seq_scan("emp", {}));
    agg->group_keys.push_back(dept.get());
    CHECK(approx(card.rows(*agg), 50.0));
    card.stats = nullptr;
    CHECK(approx(card.rows(*agg), 20000.0 * card.group_selectivity));
}

int main() {
    test_lab_statistics_load();
    test_loader_rejects_rather_than_shrugs();
    test_equality_uses_ndv();
    test_range_uses_min_max();
    test_nulls();
    test_conjunction_and_disjunction();
    test_fallbacks_are_by_class();
    test_distinct_groups();
    test_no_statistics_reproduces_the_flat_model();
    test_statistics_reach_the_row_estimate();
    test_selectivity_is_always_a_fraction();
    test_a_computed_column_is_never_looked_up();
    test_the_tree_form_reaches_the_estimator();
    if (g_failures == 0) { std::printf("statistics tests: all passed\n"); return 0; }
    std::printf("statistics tests: %d failure(s)\n", g_failures);
    return 1;
}
