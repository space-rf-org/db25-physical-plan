// The applicability sweep: a MECHANISM for a defect class, not an audit of it.
//
// Four separate defects in this planner share one shape - a precondition the cost
// model cannot express, because the inadmissible candidate is the CHEAP one:
//
//   * a keyless MergeJoin (nothing to merge on, but with no keys it also needs no
//     sort, so it looked cheapest and won);
//   * a keyless HashJoin (the same hole, one operator over, found only after the
//     MergeJoin guard was written too narrowly);
//   * a HashJoin over a correlated LATERAL right input (cheaper than the nested
//     loop that is the only executable option);
//   * a UnionAll standing in for an INTERSECT (it compares nothing, so it is
//     cheaper than the hash set operation, and it silently keeps every row);
//   * a swapped hash-join build side on an outer or semi join (sometimes cheaper,
//     and computes a different relation).
//
// Each was caught individually, by someone thinking of it. Nothing checked that
// `is_applicable` was COMPLETE. These tests are that check, and they are written
// so that a NEW operator is covered by construction rather than by remembering:
// they iterate kAllPhysicalOps, so an operator added without a decision here fails
// the build or the assertion, not a later query.
#include "db25/physical/cost.hpp"
#include "db25/physical/lowering.hpp"
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/properties.hpp"
#include "db25/physical/spec.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace db25::physical;
using db25::ast::JoinType;
using db25::ast::SetOp;

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

namespace {

const std::vector<JoinType>& all_join_kinds() {
    static const std::vector<JoinType> v{JoinType::Inner, JoinType::Left,  JoinType::Right,
                                         JoinType::Full,  JoinType::Cross, JoinType::Lateral,
                                         JoinType::LeftLateral};
    return v;
}

const std::vector<SetOp>& all_set_ops() {
    static const std::vector<SetOp> v{SetOp::Union,        SetOp::UnionAll,
                                      SetOp::Intersect,    SetOp::Except,
                                      SetOp::IntersectAll, SetOp::ExceptAll};
    return v;
}

// Every context `is_applicable` can be asked about, as one list. Small enough to
// enumerate exhaustively, which is the point - a sampled grid would have the same
// blind spot the individual guards had.
struct Context {
    std::vector<HashKey> keys;
    JoinType join_kind;
    GroupingSpec grouping;
    SetOp set_op;
};

std::vector<Context> all_contexts() {
    std::vector<Context> out;
    const std::vector<std::vector<HashKey>> key_sets{{}, {{0, 0}}, {{0, 0}, {1, 1}}};
    const std::vector<GroupingSpec> groupings{
        GroupingSpec{0, false, false, 0}, GroupingSpec{0, true, false, 0},
        GroupingSpec{1, false, false, 0}, GroupingSpec{1, true, false, 0},
        GroupingSpec{1, true, true, 2},   GroupingSpec{2, false, true, 3}};
    for (const auto& k : key_sets) {
        for (const JoinType j : all_join_kinds()) {
            for (const GroupingSpec& gr : groupings) {
                for (const SetOp s : all_set_ops()) out.push_back(Context{k, j, gr, s});
            }
        }
    }
    return out;
}

const PhysicalSpec& spec() {
    static const PhysicalSpec s = [] {
        std::string error;
        auto loaded = load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
        if (!loaded) {
            std::printf("  cannot load spec: %s\n", error.c_str());
            std::exit(2);
        }
        return *loaded;
    }();
    return s;
}

}  // namespace

// 1. EVERY OPERATOR IS REACHABLE. An operator declared in the catalogue must be
// either the target of some implementation rule, or an enforcer the planner
// inserts directly. One that is neither can never appear in a plan - it is dead
// weight that the spec's conformance check cannot see, because conformance asks
// the opposite question (is everything the planner emits declared?).
static void test_every_operator_is_reachable() {
    std::printf("test_every_operator_is_reachable\n");
    for (const PhysicalOp op : kAllPhysicalOps) {
        // The two enforcers are inserted by enforce(), not by a rule.
        if (op == PhysicalOp::Sort || op == PhysicalOp::FormatConvert) continue;
        bool reachable = false;
        for (const auto& [logical, ops] : spec().resolved_rules) {
            (void)logical;
            for (const PhysicalOp p : ops) reachable = reachable || (p == op);
        }
        if (!reachable) {
            std::printf("  unreachable operator: %s\n", physical_op_to_string(op));
        }
        CHECK(reachable);
    }
}

// 2. EVERY OPERATOR'S PRECONDITION IS DECLARED, AND THE CODE AGREES WITH IT.
//
// The first version of this test asked `if (needs_equi_key(op)) CHECK(!ok)` for
// every context - which reads well and is worth nothing. It restates
// is_applicable's own implementation, so it can only fail if that function
// contradicts itself; and worse, an operator added WITHOUT being put into
// needs_equi_key makes the guard false and the check vacuous. It would have been
// silent about precisely the mistake it exists to catch.
//
// So the preconditions are DECLARED here, once, by hand, and the implementation
// must agree with the declaration. The table is asserted to cover every operator
// in kAllPhysicalOps, so a new operator cannot be added without someone writing
// down what it requires - which is the only step in this that a machine cannot do.
struct Precondition {
    PhysicalOp op;
    bool needs_keys = false;              // builds or merges on an equi-key
    bool needs_orderable_grouping = false; // states a positional sort requirement
    bool nested_loop_only = false;         // can serve a correlated right input
    // -1: no restriction. 0: only UNION ALL. 1: anything BUT union all.
    int set_op_restriction = -1;
    // -1: not an aggregate, so the question does not arise. 0: computes ONE key
    // combination, so a GROUPING SETS node is not its job. 1: computes several,
    // so a plain GROUP BY is not.
    int grouping_sets_restriction = -1;
};

const std::vector<Precondition>& declared_preconditions() {
    static const std::vector<Precondition> v{
        {PhysicalOp::SeqScan},
        {PhysicalOp::Filter},
        {PhysicalOp::Project},
        {PhysicalOp::HashJoin, true, false, false, -1},
        {PhysicalOp::MergeJoin, true, false, false, -1},
        {PhysicalOp::NestedLoopJoin, false, false, true, -1},
        {PhysicalOp::Sort},
        {PhysicalOp::FormatConvert},
        {PhysicalOp::Limit},
        {PhysicalOp::HashAggregate, false, false, false, -1, 0},
        {PhysicalOp::StreamingAggregate, false, true, false, -1, 0},
        {PhysicalOp::HashGroupingSets, false, false, false, -1, 1},
        {PhysicalOp::Window},
        {PhysicalOp::HashDistinct},
        {PhysicalOp::StreamingDistinct},
        {PhysicalOp::UnionAll, false, false, false, 0},
        {PhysicalOp::HashSetOp, false, false, false, 1},
        {PhysicalOp::ValuesScan},
        {PhysicalOp::HashSemiJoin, true, false, false, -1},
        {PhysicalOp::HashAntiJoin, true, false, false, -1},
        {PhysicalOp::NestedLoopSemiJoin, false, false, true, -1},
        {PhysicalOp::NestedLoopAntiJoin, false, false, true, -1},
        // The three of Increment 3.9a have NO precondition, and that is a claim
        // rather than an omission: each is the sole implementation of its logical
        // operator, so there is nothing for applicability to choose between. A
        // fixpoint is evaluated one way; a working table is read one way; a table
        // is written one way.
        {PhysicalOp::RecursiveFixpoint},
        {PhysicalOp::WorkingTableScan},
        {PhysicalOp::CreateTableAs},
        // The write path has no precondition either: each is the sole
        // implementation of its logical operator, and nothing about a row makes
        // one of them applicable where another is not - which of the three runs
        // is decided by the STATEMENT, upstream of any choice this planner makes.
        {PhysicalOp::Insert},
        {PhysicalOp::Update},
        {PhysicalOp::Delete},
    };
    return v;
}

static void test_the_declared_preconditions_cover_every_operator() {
    std::printf("test_the_declared_preconditions_cover_every_operator\n");
    CHECK(declared_preconditions().size() == kAllPhysicalOps.size());
    for (const PhysicalOp op : kAllPhysicalOps) {
        std::size_t found = 0;
        for (const Precondition& p : declared_preconditions()) found += (p.op == op) ? 1 : 0;
        if (found != 1) {
            std::printf("  operator with %zu declared preconditions (want exactly 1): %s\n",
                        found, physical_op_to_string(op));
        }
        CHECK(found == 1);
    }
}

static void test_the_code_agrees_with_the_declared_preconditions() {
    std::printf("test_the_code_agrees_with_the_declared_preconditions\n");
    for (const Precondition& p : declared_preconditions()) {
        for (const Context& c : all_contexts()) {
            bool want = true;
            if (p.needs_keys && c.keys.empty()) want = false;
            if (p.needs_orderable_grouping && !c.grouping.orderable) want = false;
            if (join_is_lateral(c.join_kind) && !p.nested_loop_only) want = false;
            if (p.set_op_restriction == 0 && c.set_op != SetOp::UnionAll) want = false;
            if (p.set_op_restriction == 1 && c.set_op == SetOp::UnionAll) want = false;
            if (p.grouping_sets_restriction == 0 && c.grouping.has_grouping_sets) want = false;
            if (p.grouping_sets_restriction == 1 && !c.grouping.has_grouping_sets) want = false;

            const bool got = is_applicable(p.op, c.keys, c.join_kind, c.grouping, c.set_op);
            if (got != want) {
                std::printf("  %s: declared %d, is_applicable %d "
                            "(keys=%zu kind=%s setop=%s grouping=%u/%d)\n",
                            physical_op_to_string(p.op), static_cast<int>(want),
                            static_cast<int>(got), c.keys.size(),
                            join_kind_to_string(c.join_kind), set_op_to_string(c.set_op),
                            c.grouping.key_count, static_cast<int>(c.grouping.orderable));
            }
            CHECK(got == want);
        }
    }
}

// 3. NO DEAD ENDS. For every logical operator the spec gives rules for, and every
// context that logical operator can actually occur in, at least ONE candidate must
// be applicable - or the query has no plan at all and lowering fails on a query
// the planner claims to support. This is the check that would have caught guarding
// MergeJoin without providing the NestedLoopJoin fallback.
static void test_every_logical_operator_has_a_plan_in_every_context() {
    std::printf("test_every_logical_operator_has_a_plan_in_every_context\n");
    for (const auto& [logical, ops] : spec().resolved_rules) {
        CHECK(!ops.empty());
        for (const Context& c : all_contexts()) {
            // Only ask about contexts this logical operator can be in. A join
            // carries a join kind and no set op; a set operation the reverse.
            // Asking otherwise would demand plans for situations that never arise.
            const bool is_join = logical == "Join";
            const bool is_semi = logical == "SemiJoin" || logical == "AntiJoin";
            const bool is_setop = logical == "SetOp";
            const bool is_agg = logical == "Aggregate";
            if (!is_join && !is_semi && c.join_kind != JoinType::Inner) continue;
            if (!is_setop && c.set_op != SetOp::Union) continue;
            if (!is_agg && !(c.grouping.key_count == 0 && !c.grouping.orderable)) continue;
            // A semi or anti join is never lateral - it comes from EXISTS / IN,
            // which the binder does not mark correlated at the join.
            if (is_semi && join_is_lateral(c.join_kind)) continue;

            std::size_t applicable = 0;
            for (const PhysicalOp op : ops) {
                if (is_applicable(op, c.keys, c.join_kind, c.grouping, c.set_op)) ++applicable;
            }
            if (applicable == 0) {
                std::printf("  DEAD END: logical '%s' has no applicable candidate "
                            "(keys=%zu kind=%s setop=%s grouping=%u/%d)\n",
                            logical.c_str(), c.keys.size(), join_kind_to_string(c.join_kind),
                            set_op_to_string(c.set_op), c.grouping.key_count,
                            static_cast<int>(c.grouping.orderable));
            }
            CHECK(applicable >= 1);
        }
    }
}

// 4. EVERY OPERATOR HAS A COST AND A CARDINALITY. An operator the cost model does
// not know falls through its switch and is priced at zero - which makes it the
// cheapest candidate available and therefore the one that always wins. A silent
// zero is worse than a crash: the plan looks ordinary.
static void test_every_operator_is_priced() {
    std::printf("test_every_operator_is_priced\n");
    const CalibrationProfile cal = default_calibration();
    CardinalityModel card;
    card.base_rows["t"] = 1000.0;
    const double in[2] = {1000.0, 500.0};
    for (const PhysicalOp op : kAllPhysicalOps) {
        const std::size_t arity = expected_arity(op);
        const std::span<const double> inputs{in, arity};
        const double rows = operator_rows(op, inputs, "t", card, LimitSpec{true, false, 10, 0},
                                          GroupingSpec{1, true}, SetOp::Union, 2.0);
        const double cost = operator_cost(op, inputs, rows, cal);
        if (rows <= 0.0 || cost <= 0.0) {
            std::printf("  unpriced operator: %s (rows=%.3f cost=%.3f)\n",
                        physical_op_to_string(op), rows, cost);
        }
        CHECK(rows > 0.0);
        CHECK(cost > 0.0);
    }
}

// 5. EVERY OPERATOR DERIVES AND RENDERS. An operator missing from derive_op falls
// through to a default-constructed PhysicalProperties, which claims Any format and
// Any freshness - a claim that a stale columnar subplan satisfies a Fresh row
// requirement. And one missing from the s-expr writer renders without its payload,
// which is how a join kind, an aggregate FILTER and a window's OVER clause each
// went missing from a golden.
static void test_every_operator_derives_and_renders() {
    std::printf("test_every_operator_derives_and_renders\n");
    const PhysicalProperties stale_column{{}, StorageFormat::Column, Freshness::Stale};
    const PhysicalProperties inputs[2] = {stale_column, stale_column};
    for (const PhysicalOp op : kAllPhysicalOps) {
        const std::size_t arity = expected_arity(op);
        const PhysicalProperties p =
            derive_op(op, {}, StorageFormat::Row, std::span<const PhysicalProperties>{inputs, arity},
                      Freshness::Fresh);
        // Staleness must propagate: nothing here manufactures freshness, so an
        // operator over a stale input cannot claim Fresh, and must not leave the
        // field at Any - which would satisfy a Fresh requirement by accident.
        // The write path is the ONE exception, and a real one rather than an
        // oversight: Insert / Update / Delete emit the rows they just WROTE, so
        // their output reflects all committed writes whatever their input read.
        // `INSERT INTO t SELECT ... FROM replica RETURNING *` returns rows that
        // now exist in t. It is the only place in this catalogue where freshness
        // is manufactured, and it is manufactured because the operator is what
        // makes the data current.
        const bool writes = op == PhysicalOp::Insert || op == PhysicalOp::Update ||
                            op == PhysicalOp::Delete;
        if (arity > 0 && !writes) {
            if (p.freshness == Freshness::Any) {
                std::printf("  operator does not derive freshness: %s\n",
                            physical_op_to_string(op));
            }
            CHECK(p.freshness != Freshness::Any);
            CHECK(p.freshness == Freshness::Stale);
        }
        if (writes) CHECK(p.freshness == Freshness::Fresh);
        // And it must have a NAME - an operator absent from the string table
        // renders as "?" and is unreadable in every golden it appears in.
        CHECK(std::string(physical_op_to_string(op)) != "?");
    }
}

int main() {
    test_every_operator_is_reachable();
    test_the_declared_preconditions_cover_every_operator();
    test_the_code_agrees_with_the_declared_preconditions();
    test_every_logical_operator_has_a_plan_in_every_context();
    test_every_operator_is_priced();
    test_every_operator_derives_and_renders();
    if (g_failures != 0) {
        std::printf("applicability sweep: %d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("applicability sweep: all passed\n");
    return 0;
}
