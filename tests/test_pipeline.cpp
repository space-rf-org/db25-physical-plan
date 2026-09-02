// Increment 4.1: pipelines - where a row stops.
//
// The claim this file exists to keep honest is that BREAKING IS A PROPERTY OF AN
// EDGE, not of an operator. A hash join breaks its build input and streams its
// probe input, and which one is which is decided per plan by the search. The
// spec labels each operator `pipeline` or `pipeline-breaker`, one label each,
// and cannot say that.
#include "db25/physical/lowering.hpp"
#include "db25/physical/pipeline.hpp"
#include "db25/physical/sexpr.hpp"
#include "db25/physical/spec.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdio>
#include <memory>
#include <set>
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

static bool contains(const std::string& hay, const char* needle) {
    return hay.find(needle) != std::string::npos;
}

// ---- builders -------------------------------------------------------------

static plan::ExprPtr col(std::uint32_t i) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    e->input_index = i;
    e->type = DataType::Integer;
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
static plan::LogicalNodePtr scan(const std::string& name) {
    auto s = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Scan);
    s->table_name = name;
    s->output = {{name + "0", DataType::Integer, false, 0, 0, name, false},
                 {name + "1", DataType::Integer, true, 0, 0, name, false}};
    return s;
}
// `a JOIN b ON a.a0 = b.b0`.
static plan::LogicalNodePtr join_ab() {
    auto sa = scan("a");
    auto sb = scan("b");
    auto j = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
    j->join_type = db25::ast::JoinType::Inner;
    j->output = sa->output;
    for (const auto& c : sb->output) j->output.push_back(c);
    j->predicate = binop(BinaryOp::Equal, col(0), col(2));
    j->add_child(std::move(sa));
    j->add_child(std::move(sb));
    return j;
}
static LoweringResult lower_with(const plan::LogicalNode& n, double a_rows, double b_rows,
                                 const PhysicalProperties& required = {}) {
    static CardinalityModel card;
    card = CardinalityModel{};
    card.base_rows["a"] = a_rows;
    card.base_rows["b"] = b_rows;
    LoweringContext ctx;
    ctx.cardinality = &card;
    ctx.required_output = required;
    return lower(n, ctx);
}

// The pipeline a given operator belongs to, by index; -1 if none.
static int pipeline_of(const std::vector<Pipeline>& ps, PhysicalOp op) {
    for (std::size_t i = 0; i < ps.size(); ++i) {
        for (const PhysicalNode* m : ps[i].members) {
            if (m->op == op) return static_cast<int>(i);
        }
    }
    return -1;
}
static const PhysicalNode* find_op(const PhysicalNode& n, PhysicalOp op) {
    if (n.op == op) return &n;
    for (const auto& c : n.children) {
        if (const PhysicalNode* r = find_op(*c, op)) return r;
    }
    return nullptr;
}

// ---- tests ----------------------------------------------------------------

// The headline, and the reason this is per-edge. A hash join's build side ends a
// pipeline and its probe side continues one - and when the search picks the OTHER
// build side, the two swap. An operator-level label cannot express a fact that
// changes with the plan.
static void test_a_hash_join_breaks_the_build_side_and_streams_the_probe() {
    std::printf("test_a_hash_join_breaks_the_build_side_and_streams_the_probe\n");
    const auto q = join_ab();

    // `b` tiny: the search builds from the right, so `a` streams into the join.
    const LoweringResult small_b = lower_with(*q, 100000.0, 10.0);
    CHECK(small_b.ok);
    if (!small_b.plan) return;
    CHECK(contains(physical_to_sexpr(*small_b.plan), "build=right"));
    const std::vector<Pipeline> pb = pipelines(*small_b.plan);
    const PhysicalNode* join_b = find_op(*small_b.plan, PhysicalOp::HashJoin);
    CHECK(join_b != nullptr);
    if (join_b == nullptr) return;
    // The join shares a pipeline with the scan it PROBES with (its left input),
    // and not with the one it builds from.
    const int jp = pipeline_of(pb, PhysicalOp::HashJoin);
    bool left_with_join = false, right_with_join = false;
    for (const PhysicalNode* m : pb[static_cast<std::size_t>(jp)].members) {
        if (m == join_b->children[0].get()) left_with_join = true;
        if (m == join_b->children[1].get()) right_with_join = true;
    }
    CHECK(left_with_join);
    CHECK(!right_with_join);

    // Now `a` tiny: the build side flips, and so does the answer.
    const LoweringResult small_a = lower_with(*q, 10.0, 100000.0);
    CHECK(small_a.ok);
    if (!small_a.plan) return;
    CHECK(contains(physical_to_sexpr(*small_a.plan), "build=left"));
    const std::vector<Pipeline> pa = pipelines(*small_a.plan);
    const PhysicalNode* join_a = find_op(*small_a.plan, PhysicalOp::HashJoin);
    CHECK(join_a != nullptr);
    if (join_a == nullptr) return;
    const int ja = pipeline_of(pa, PhysicalOp::HashJoin);
    bool left2 = false, right2 = false;
    for (const PhysicalNode* m : pa[static_cast<std::size_t>(ja)].members) {
        if (m == join_a->children[0].get()) left2 = true;
        if (m == join_a->children[1].get()) right2 = true;
    }
    // The mirror image. A per-OPERATOR breaker flag would have given the same
    // answer in both runs, and one of them would have been wrong.
    CHECK(!left2);
    CHECK(right2);
}

// Every node runs in exactly one loop. A node in no pipeline is an operator
// nothing executes; a node in two is one an executor would generate twice.
static void test_every_node_is_in_exactly_one_pipeline() {
    std::printf("test_every_node_is_in_exactly_one_pipeline\n");
    struct Counter {
        static void count(const PhysicalNode& n, std::size_t& total) {
            ++total;
            for (const auto& c : n.children) count(*c, total);
        }
    };
    const auto q = join_ab();
    PhysicalProperties sorted;
    sorted.sort = {SortKey{0, false, false, false}};
    for (const PhysicalProperties& req : {PhysicalProperties{}, sorted}) {
        const LoweringResult r = lower_with(*q, 1000.0, 1000.0, req);
        CHECK(r.ok);
        if (!r.plan) continue;
        std::size_t nodes = 0;
        Counter::count(*r.plan, nodes);
        std::multiset<const PhysicalNode*> seen;
        for (const Pipeline& p : pipelines(*r.plan)) {
            for (const PhysicalNode* m : p.members) seen.insert(m);
        }
        CHECK(seen.size() == nodes);
        for (const PhysicalNode* m : seen) CHECK(seen.count(m) == 1);
    }
}

// The numbering is a SCHEDULE. Every pipeline must come out after the ones whose
// output it consumes - otherwise a build pipeline could be listed after the
// probe that reads its hash table, and an executor following the order would
// probe an empty one.
static void test_pipelines_come_out_in_execution_order() {
    std::printf("test_pipelines_come_out_in_execution_order\n");
    const auto q = join_ab();
    PhysicalProperties sorted;
    sorted.sort = {SortKey{0, false, false, false}};
    const LoweringResult r = lower_with(*q, 100000.0, 10.0, sorted);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::vector<Pipeline> ps = pipelines(*r.plan);
    CHECK(ps.size() >= 3);
    // Where each node lives.
    const auto index_of = [&](const PhysicalNode* n) {
        for (std::size_t i = 0; i < ps.size(); ++i) {
            for (const PhysicalNode* m : ps[i].members) {
                if (m == n) return static_cast<int>(i);
            }
        }
        return -1;
    };
    for (std::size_t i = 0; i < ps.size(); ++i) {
        for (const PhysicalNode* m : ps[i].members) {
            for (std::size_t k = 0; k < m->children.size(); ++k) {
                if (edge_kind(m->op, k, m->build_right) == EdgeKind::Streaming) continue;
                const int dep = index_of(m->children[k].get());
                CHECK(dep >= 0);
                // Strictly earlier: the pipeline that fills a hash table must run
                // before the one that probes it.
                CHECK(dep < static_cast<int>(i));
            }
        }
    }
}

// The aggregate trade, made visible in the shape rather than only in the cost. A
// streaming aggregate emits a group the moment it closes, so the pipeline runs
// through it; a hash aggregate holds every row first, so it ends one. That is the
// same fact the cost model prices, seen from the other side.
static void test_the_two_aggregates_differ_in_where_rows_stop() {
    std::printf("test_the_two_aggregates_differ_in_where_rows_stop\n");
    CHECK(edge_kind(PhysicalOp::StreamingAggregate, 0, true) == EdgeKind::Streaming);
    CHECK(edge_kind(PhysicalOp::HashAggregate, 0, true) == EdgeKind::Materialized);
    CHECK(edge_kind(PhysicalOp::StreamingDistinct, 0, true) == EdgeKind::Streaming);
    CHECK(edge_kind(PhysicalOp::HashDistinct, 0, true) == EdgeKind::Materialized);
}

// Rescanned is not Materialized, and the difference is where the memory goes. A
// nested loop RE-EVALUATES its right input per left row - that is why it is
// quadratic, and why it is the only join that can serve a correlated LATERAL
// right side. Calling it materialized would tell an executor to allocate for it.
static void test_a_nested_loop_rescans_and_a_merge_join_allocates_nothing() {
    std::printf("test_a_nested_loop_rescans_and_a_merge_join_allocates_nothing\n");
    CHECK(edge_kind(PhysicalOp::NestedLoopJoin, 1, true) == EdgeKind::Rescanned);
    CHECK(edge_kind(PhysicalOp::NestedLoopSemiJoin, 1, true) == EdgeKind::Rescanned);
    CHECK(edge_kind(PhysicalOp::RecursiveFixpoint, 1, true) == EdgeKind::Rescanned);
    // A merge join buffers neither input - two loops advancing against each
    // other - so its right edge is Separate, not Materialized.
    CHECK(edge_kind(PhysicalOp::MergeJoin, 1, true) == EdgeKind::Separate);
    // And a concatenation allocates nothing either.
    CHECK(edge_kind(PhysicalOp::UnionAll, 0, true) == EdgeKind::Separate);
    CHECK(edge_kind(PhysicalOp::UnionAll, 1, true) == EdgeKind::Separate);
    // While the hash set operation does build from its right.
    CHECK(edge_kind(PhysicalOp::HashSetOp, 1, true) == EdgeKind::Materialized);
    CHECK(edge_kind(PhysicalOp::HashSetOp, 0, true) == EdgeKind::Streaming);
}

// The sweep. Every INPUT of every operator must have a declared edge kind, so an
// operator cannot be added without someone deciding whether rows stop in it -
// the same discipline as the declared preconditions and the render cases.
struct DeclaredEdge {
    PhysicalOp op;
    EdgeKind input0;
    EdgeKind input1;  // ignored for arity < 2
};

static const std::vector<DeclaredEdge>& declared_edges() {
    static const std::vector<DeclaredEdge> v{
        {PhysicalOp::SeqScan, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::ValuesScan, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::WorkingTableScan, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::Filter, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::Project, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::FormatConvert, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::Limit, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::Window, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::StreamingAggregate, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::StreamingDistinct, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::CreateTableAs, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::Insert, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::Update, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::Delete, EdgeKind::Streaming, EdgeKind::Streaming},
        {PhysicalOp::Sort, EdgeKind::Materialized, EdgeKind::Materialized},
        {PhysicalOp::HashAggregate, EdgeKind::Materialized, EdgeKind::Materialized},
        {PhysicalOp::HashGroupingSets, EdgeKind::Materialized, EdgeKind::Materialized},
        {PhysicalOp::HashDistinct, EdgeKind::Materialized, EdgeKind::Materialized},
        // Declared for build_right = true; the flip is checked separately, since
        // a table indexed by operator alone cannot express it - which is the
        // whole point of this unit.
        {PhysicalOp::HashJoin, EdgeKind::Streaming, EdgeKind::Materialized},
        {PhysicalOp::HashSemiJoin, EdgeKind::Streaming, EdgeKind::Materialized},
        {PhysicalOp::HashAntiJoin, EdgeKind::Streaming, EdgeKind::Materialized},
        {PhysicalOp::HashSetOp, EdgeKind::Streaming, EdgeKind::Materialized},
        {PhysicalOp::UnionAll, EdgeKind::Separate, EdgeKind::Separate},
        {PhysicalOp::MergeJoin, EdgeKind::Streaming, EdgeKind::Separate},
        {PhysicalOp::NestedLoopJoin, EdgeKind::Streaming, EdgeKind::Rescanned},
        {PhysicalOp::NestedLoopSemiJoin, EdgeKind::Streaming, EdgeKind::Rescanned},
        {PhysicalOp::NestedLoopAntiJoin, EdgeKind::Streaming, EdgeKind::Rescanned},
        {PhysicalOp::RecursiveFixpoint, EdgeKind::Streaming, EdgeKind::Rescanned},
    };
    return v;
}

static void test_every_operator_declares_where_rows_stop() {
    std::printf("test_every_operator_declares_where_rows_stop\n");
    CHECK(declared_edges().size() == kAllPhysicalOps.size());
    for (const PhysicalOp op : kAllPhysicalOps) {
        std::size_t found = 0;
        for (const DeclaredEdge& d : declared_edges()) found += (d.op == op) ? 1 : 0;
        if (found != 1) {
            std::printf("  operator with %zu declared edge kinds (want 1): %s - where "
                        "do rows stop in it?\n", found, physical_op_to_string(op));
        }
        CHECK(found == 1);
    }
    for (const DeclaredEdge& d : declared_edges()) {
        const std::size_t arity = expected_arity(d.op);
        if (arity >= 1) CHECK(edge_kind(d.op, 0, true) == d.input0);
        if (arity >= 2) CHECK(edge_kind(d.op, 1, true) == d.input1);
    }
    // And the one thing the table above cannot say: the hash-family build side
    // follows `build_right`, so the two inputs SWAP when the search chooses the
    // other side. Only HashJoin has that choice (is_build_side_choosable), so
    // only HashJoin flips.
    CHECK(edge_kind(PhysicalOp::HashJoin, 0, false) == EdgeKind::Materialized);
    CHECK(edge_kind(PhysicalOp::HashJoin, 1, false) == EdgeKind::Streaming);
    CHECK(edge_kind(PhysicalOp::HashSemiJoin, 1, false) == EdgeKind::Materialized);
}

// The spec now DECLARES where rows stop, one word per input, and
// check_conformance asserts the code agrees and that `kind` is the summary those
// words imply. This test is that the shipped spec is conformant - the same
// question test_spec asks, asked again here because this is the unit that made
// `edges` mean something and a silent regression would belong to it.
//
// Nine labels were wrong when the check was first written: a merge join, the
// three nested loops and the recursive fixpoint were called breakers though they
// buffer nothing, and the four write operators were called breakers though rows
// pass straight through them into a table. They END pipelines - a real fact,
// which `edges` records as `separate` / `rescanned` / `streaming` - but they do
// not BREAK them in the sense that costs memory, which is what a reader takes
// from the word.
static void test_the_shipped_spec_declares_where_rows_stop() {
    std::printf("test_the_shipped_spec_declares_where_rows_stop\n");
    std::string error;
    auto spec = db25::physical::load_spec(
        std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    CHECK(spec.has_value());
    if (!spec) { std::printf("  spec load failed: %s\n", error.c_str()); return; }
    for (const std::string& problem : db25::physical::check_conformance(*spec)) {
        std::printf("  %s\n", problem.c_str());
        ++g_failures;
    }
    // And every operator with inputs really did declare them - a spec that simply
    // omitted `edges` would otherwise conform by saying nothing.
    for (const PhysicalOp op : kAllPhysicalOps) {
        const auto* os = spec->find_operator(physical_op_to_string(op));
        CHECK(os != nullptr);
        if (os != nullptr) CHECK(os->edges.size() == expected_arity(op));
    }
}

int main() {
    test_the_shipped_spec_declares_where_rows_stop();
    test_a_hash_join_breaks_the_build_side_and_streams_the_probe();
    test_every_node_is_in_exactly_one_pipeline();
    test_pipelines_come_out_in_execution_order();
    test_the_two_aggregates_differ_in_where_rows_stop();
    test_a_nested_loop_rescans_and_a_merge_join_allocates_nothing();
    test_every_operator_declares_where_rows_stop();

    if (g_failures == 0) {
        std::printf("pipeline tests: all passed\n");
        return 0;
    }
    std::printf("pipeline tests: %d failure(s)\n", g_failures);
    return 1;
}
