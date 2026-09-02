// Unit 0.2: the physical IR, the Cascades memo scaffold, and s-expr rendering.
//
// Builds the Increment-0 shape by hand - the simple query
//   SELECT a.x, b.y FROM a JOIN b ON a.id = b.id WHERE a.x > 10
// as a physical plan (SeqScan/SeqScan -> HashJoin -> Filter -> Project) - and
// checks: (1) the render carries the structural facts, (2) a memo built to the
// same shape extracts to an identical render (memo path == direct path), and
// (3) a structural mutation changes the render (falsifiability).
#include "db25/physical/cost.hpp"
#include "db25/physical/memo.hpp"
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/sexpr.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"

#include <cstdio>
#include <optional>
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

// Build a group-expression by field. Positional aggregate initialisation breaks
// silently whenever GroupExpr gains a member, so name what we set.
// A candidate is now only its ALGORITHM; the logical payload it implements lives
// on the group, because every candidate in a group shares it. So building one for
// a test means setting the group's payload and appending the alternative.
static GroupExpr group_expr(PhysicalOp op) {
    GroupExpr ge;
    ge.op = op;
    ge.cost = 0.0;
    return ge;
}
// `table` is BORROWED, matching Group::table_name - the caller must keep the
// string alive for as long as the memo. Passing a temporary here is a
// use-after-free that ASan catches on the first run.
static std::vector<GroupId> g_pending_inputs;
static std::vector<HashKey> g_pending_keys;

static void set_payload(Memo& m, GroupId g, std::vector<GroupId> inputs,
                        const std::string* table = nullptr, const Expr* pred = nullptr,
                        std::vector<const Expr*> projections = {},
                        std::vector<HashKey> keys = {}) {
    Group& grp = m.group(g);
    grp.table_name = table;
    grp.predicate = pred;
    grp.op_exprs = std::move(projections);
    // Inputs and keys belong to the CANDIDATE now, so they are stashed for
    // add_candidate to attach - a group with no candidate has no inputs, which is
    // the point of the move.
    g_pending_inputs = std::move(inputs);
    g_pending_keys = std::move(keys);
}
static void add_candidate(Memo& m, GroupId g, PhysicalOp op, std::vector<GroupId> inputs,
                          const std::string* table = nullptr, const Expr* pred = nullptr,
                          std::vector<const Expr*> projections = {},
                          std::vector<HashKey> keys = {}) {
    set_payload(m, g, std::move(inputs), table, pred, std::move(projections),
                std::move(keys));
    GroupExpr ge = group_expr(op);
    for (const GroupId in : g_pending_inputs) ge.inputs.push_back(in);
    ge.hash_keys = g_pending_keys;
    m.add_expr(g, std::move(ge));
}

// Borrowed by the memo, so they live as long as the program rather than as long
// as the call. A string literal cannot be passed: `const char*` does not convert,
// which is the type system enforcing the lifetime contract rather than ASan
// discovering it later.
static const std::string kTableA = "a";
static const std::string kTableB = "b";

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

// Record a group's UNCONSTRAINED winner - "cheapest plan, no requirement stated".
// These are structural tests of extraction, written before a group could be
// optimized FOR a requirement; the unconstrained key is exactly what they mean.
static void win(db25::physical::Memo& m, db25::physical::GroupId g, std::uint32_t idx) {
    m.set_winner(g, db25::physical::Group::unconstrained(), idx, 0.0, {});
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
    const Schema sch_a = a_schema();
    const GroupId g_a = m.add_group(sch_a);
    add_candidate(m, g_a, PhysicalOp::SeqScan, {}, &kTableA);
    win(m, g_a, 0);

    const Schema sch_b = b_schema();
    const GroupId g_b = m.add_group(sch_b);
    add_candidate(m, g_b, PhysicalOp::SeqScan, {}, &kTableB);
    win(m, g_b, 0);

    const Schema sch_join = join_schema();
    const GroupId g_join = m.add_group(sch_join);
    add_candidate(m, g_join, PhysicalOp::HashJoin, {g_a, g_b}, nullptr, nullptr, {}, {{0, 0}});
    win(m, g_join, 0);

    const Schema sch_filter = join_schema();
    const GroupId g_filter = m.add_group(sch_filter);
    add_candidate(m, g_filter, PhysicalOp::Filter, {g_join}, nullptr, pred);
    win(m, g_filter, 0);

    const Schema sch_proj = proj_schema();
    const GroupId g_proj = m.add_group(sch_proj);
    add_candidate(m, g_proj, PhysicalOp::Project, {g_filter}, nullptr, nullptr, {px, py});
    win(m, g_proj, 0);
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
    const Schema sch = a_schema();  // borrowed by the memo: must outlive it
    const GroupId g = m.add_group(sch);
    add_candidate(m, g, PhysicalOp::SeqScan, {}, &kTableA);
    // No winner set.
    m.set_root(g);
    CHECK(m.extract_winner() == nullptr);
}

// Unit 2.1: a group's winner is keyed by the REQUIREMENT it was optimized for.
//
// The defect this lifts: a group used to hold ONE winner, chosen bottom-up before
// any consumer stated a requirement. That makes it impossible to ASK a group for
// a property - only to charge it for lacking one - which is why a scan can never
// be optimized for an order its parent will need. Here the same group answers
// three different questions with three different plans.
static void test_group_answers_per_requirement() {
    std::printf("test_group_answers_per_requirement\n");
    const CalibrationProfile cal = default_calibration();
    const double kRows = 1000.0;

    // Memo-level test: the candidates are synthetic, because what is under test is
    // the memo's bookkeeping, not which operator produced which property.
    Memo m;
    const Schema sch = a_schema();  // borrowed by the memo: must outlive it
    const GroupId g = m.add_group(sch);
    m.set_rows(g, kRows);

    // Cheap, but unordered.
    GroupExpr unsorted = group_expr(PhysicalOp::SeqScan);
    unsorted.cost = 1000.0;
    unsorted.provided = PhysicalProperties{{}, StorageFormat::Row, Freshness::Fresh};
    m.add_expr(g, unsorted);

    // Dearer, but already in the order a consumer might want. Sorting the cheap
    // one instead costs n*log2(n) ~ 9966 at these coefficients, which is what
    // makes the requirement decisive rather than marginal.
    GroupExpr presorted = group_expr(PhysicalOp::SeqScan);
    presorted.cost = 1200.0;
    presorted.provided = PhysicalProperties{{SortKey{0, false}}, StorageFormat::Row,
                                            Freshness::Fresh};
    m.add_expr(g, presorted);

    // Unconstrained - "cheapest plan, I need nothing in particular" - takes the
    // cheap unordered candidate. This is the only question the memo could ask
    // before this unit, and it is why a scan could never be optimized for an order
    // its parent was about to require.
    CHECK(m.select_cheapest(g, cal));
    const std::optional<std::uint32_t> any_index =
        m.group(g).winner_index_for(Group::unconstrained());
    const WinnerEntry* any = m.group(g).winner();
    CHECK(any != nullptr);
    if (any) CHECK(any->expr_index == 0);

    // Asked FOR the order, the same group answers with the other candidate: the
    // pre-sorted one pays no enforcement, the cheap one would pay a Sort.
    PhysicalProperties want_sorted;
    want_sorted.sort = {SortKey{0, false}};
    CHECK(m.select_cheapest(g, want_sorted, cal));
    const WinnerEntry* sorted = m.group(g).winner_for(want_sorted);
    CHECK(sorted != nullptr);
    if (sorted) {
        CHECK(sorted->expr_index == 1);       // a DIFFERENT plan for a different ask
        CHECK(sorted->cost == 1200.0);        // and no enforcement charged
        CHECK(sorted->provided.sort.size() == 1);
    }
    if (any_index && sorted) {
        CHECK(m.group(g).winners[*any_index].expr_index != sorted->expr_index);
    }

    // An INDEX taken before the second select still reads the unconstrained
    // answer. Indices, not pointers, are what survives a group gaining winners
    // while a recursive search holds a result.
    //
    // This originally pinned the opposite: the winners container was a deque so
    // that POINTERS stayed valid. That was safe and expensive - an empty
    // libstdc++ deque allocates immediately, so every group paid for one whether
    // or not it ever held a winner, and profiling put ~37% of the optimizer's
    // instructions in malloc/free. Indices give the same guarantee for nothing.
    CHECK(any_index.has_value());
    if (any_index) CHECK(m.group(g).winners[*any_index].expr_index == 0);

    // Both answers are retained: asking a new question does not overwrite the
    // answer to an old one. That memoization is the point of the unit.
    CHECK(m.group(g).winners.size() == 2);

    // Re-asking a question already answered replaces its entry rather than
    // appending a second one under the same key.
    CHECK(m.select_cheapest(g, want_sorted, cal));
    CHECK(m.group(g).winners.size() == 2);

    // A requirement nobody optimized for has no answer - not a wrong one.
    PhysicalProperties never_asked;
    never_asked.sort = {SortKey{7, true}};
    CHECK(m.group(g).winner_for(never_asked) == nullptr);
    CHECK(m.extract_winner_for(g, never_asked) == nullptr);

    // Extraction is per requirement too, and yields the two different plans.
    m.set_root(g);
    auto any_plan = m.extract_winner_for(g, Group::unconstrained());
    auto sorted_plan = m.extract_winner_for(g, want_sorted);
    CHECK(any_plan != nullptr);
    CHECK(sorted_plan != nullptr);
}

// A second requirement whose answer is NOT a different candidate - and that is
// the cost model being right, not the memo being wrong. Under the lab
// calibration a columnar read (350) plus a FormatConvert (600) still beats a row
// read (1000), so the columnar candidate wins the row-format question too. What
// the requirement changes is what the winner must PROVIDE, and therefore the
// enforcer extraction will insert.
static void test_requirement_may_change_only_what_is_provided() {
    std::printf("test_requirement_may_change_only_what_is_provided\n");
    const CalibrationProfile cal = default_calibration();
    const double kRows = 1000.0;

    Memo m;
    const Schema sch = a_schema();  // borrowed by the memo: must outlive it
    const GroupId g = m.add_group(sch);
    m.set_rows(g, kRows);

    GroupExpr row_scan = group_expr(PhysicalOp::SeqScan);
    row_scan.scan_format = StorageFormat::Row;
    row_scan.cost = kRows * cal.scan_row;             // 1000
    row_scan.provided = PhysicalProperties{{}, StorageFormat::Row, Freshness::Fresh};
    m.add_expr(g, row_scan);

    GroupExpr col_scan = group_expr(PhysicalOp::SeqScan);
    col_scan.scan_format = StorageFormat::Column;
    col_scan.cost = kRows * cal.column_scan_row;      // 350
    col_scan.provided = PhysicalProperties{{}, StorageFormat::Column, Freshness::Fresh};
    m.add_expr(g, col_scan);

    PhysicalProperties want_row;
    want_row.format = StorageFormat::Row;
    CHECK(m.select_cheapest(g, want_row, cal));
    const WinnerEntry* w = m.group(g).winner_for(want_row);
    CHECK(w != nullptr);
    if (w) {
        CHECK(w->expr_index == 1);                       // still the columnar read
        CHECK(w->cost == kRows * cal.column_scan_row + kRows * cal.convert_row);  // 950
        CHECK(w->cost < kRows * cal.scan_row);           // and it really is cheaper
        CHECK(w->provided.format == StorageFormat::Row); // upgraded by the requirement
    }
}

// An unmet Fresh requirement is not a winner to record: no enforcer establishes
// freshness, so the group must report that it cannot answer at all.
static void test_unsatisfiable_requirement_has_no_winner() {
    std::printf("test_unsatisfiable_requirement_has_no_winner\n");
    const CalibrationProfile cal = default_calibration();
    Memo m;
    const Schema sch = a_schema();  // borrowed by the memo: must outlive it
    const GroupId g = m.add_group(sch);
    m.set_rows(g, 100.0);

    GroupExpr stale = group_expr(PhysicalOp::SeqScan);
    stale.scan_freshness = Freshness::Stale;
    stale.provided = PhysicalProperties{{}, StorageFormat::Row, Freshness::Stale};
    stale.cost = 1.0;
    m.add_expr(g, stale);

    PhysicalProperties want_fresh;
    want_fresh.freshness = Freshness::Fresh;
    CHECK(!m.select_cheapest(g, want_fresh, cal));          // reported, not guessed
    CHECK(m.group(g).winner_for(want_fresh) == nullptr);
    CHECK(m.select_cheapest(g, cal));                       // unconstrained still fine
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
    test_group_answers_per_requirement();
    test_requirement_may_change_only_what_is_provided();
    test_unsatisfiable_requirement_has_no_winner();
    test_falsifiability_mutation_changes_render();

    if (g_failures == 0) {
        std::printf("physical IR tests: all passed\n");
        return 0;
    }
    std::printf("physical IR tests: %d failure(s)\n", g_failures);
    return 1;
}
