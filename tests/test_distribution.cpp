// Increment 4.2: distribution as a physical property.
//
// The same machinery as sort order and storage format, one dimension over: the
// planner DERIVES where a subplan's rows are, an operator REQUIRES them
// somewhere, and an Exchange ENFORCES the difference at a price. What is new is
// that the price is a NETWORK price, and it is the first cost in this model that
// an operator can incur without doing any work of its own.
//
// DB25 is single-node today, so a table nobody declares is `Single` and no plan
// contains an Exchange. Every test here that wants one says so.
#include "db25/physical/cost.hpp"
#include "db25/physical/distribution_catalog.hpp"
#include "db25/physical/lowering.hpp"
#include "db25/physical/pipeline.hpp"
#include "db25/physical/properties.hpp"
#include "db25/physical/sexpr.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdio>
#include <memory>
#include <string>

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
static std::size_t count(const std::string& hay, const std::string& needle) {
    std::size_t n = 0;
    for (std::size_t i = hay.find(needle); i != std::string::npos;
         i = hay.find(needle, i + needle.size())) {
        ++n;
    }
    return n;
}

static plan::ExprPtr col(std::uint32_t i) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ColumnRef);
    e->input_index = i;
    e->type = DataType::Integer;
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

// ---- the rule that runs the direction people guess wrong -------------------

// Rows partitioned by hash(K') are co-located by K when K' is a SUBSET of K: two
// rows agreeing on K also agree on K', so they hash to the same node.
// Partitioning on FEWER columns is the STRONGER guarantee, and getting this
// backwards would let a join read rows that are on another machine.
static void test_hashing_on_fewer_columns_is_the_stronger_guarantee() {
    std::printf("test_hashing_on_fewer_columns_is_the_stronger_guarantee\n");
    const Distribution on_a{DistributionKind::Hashed, {0}};
    const Distribution on_ab{DistributionKind::Hashed, {0, 1}};
    // Partitioned on (a) alone, asked to be co-located by (a, b): satisfied.
    CHECK(distribution_satisfies(on_a, on_ab));
    // Partitioned on (a, b), asked to be co-located by (a) alone: NOT satisfied -
    // two rows with the same `a` and different `b` are on different nodes.
    CHECK(!distribution_satisfies(on_ab, on_a));

    // Everything-in-one-place satisfies any co-location requirement.
    const Distribution single{DistributionKind::Single, {}};
    const Distribution broadcast{DistributionKind::Broadcast, {}};
    CHECK(distribution_satisfies(single, on_ab));
    CHECK(distribution_satisfies(broadcast, on_ab));
    // But a broadcast is not a gather and a gather is not a broadcast: one node
    // holding everything is not every node holding everything.
    CHECK(!distribution_satisfies(single, broadcast));
    CHECK(!distribution_satisfies(broadcast, single));
    // And an unspecified requirement is satisfied by anything.
    CHECK(distribution_satisfies(on_ab, Distribution{}));
}

// The default is single-node, and with it NOTHING changes: no Exchange, and the
// same plan the planner produced before distribution existed. A property that
// quietly rewrote every plan the day it was added would not be a property, it
// would be a regression.
static void test_the_single_node_default_inserts_no_exchange() {
    std::printf("test_the_single_node_default_inserts_no_exchange\n");
    const auto q = join_ab();
    CardinalityModel card;
    card.base_rows["a"] = 1000.0;
    card.base_rows["b"] = 1000.0;
    LoweringContext ctx;
    ctx.cardinality = &card;
    const LoweringResult r = lower(*q, ctx);
    CHECK(r.ok);
    if (!r.plan) return;
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(!contains(s, "Exchange"));

    // And a plan lowered with an explicitly single-node catalog is byte-identical
    // to one lowered with no catalog at all.
    DistributionCatalog single;
    single.tables["a"] = Distribution{DistributionKind::Single, {}};
    single.tables["b"] = Distribution{DistributionKind::Single, {}};
    LoweringContext with_cat = ctx;
    with_cat.distribution = &single;
    const LoweringResult r2 = lower(*q, with_cat);
    CHECK(r2.ok);
    if (r2.plan) CHECK(physical_to_sexpr(*r2.plan) == s);
}

// Two tables partitioned on their JOIN KEYS are already co-located, so the join
// runs where the rows are and no Exchange is inserted. This is the case a
// distributed planner exists to find, and finding it is what makes the property
// worth deriving rather than assuming.
static void test_co_located_tables_join_without_moving_a_row() {
    std::printf("test_co_located_tables_join_without_moving_a_row\n");
    const auto q = join_ab();
    CardinalityModel card;
    card.base_rows["a"] = 1000.0;
    card.base_rows["b"] = 1000.0;
    DistributionCatalog cat;
    cat.tables["a"] = Distribution{DistributionKind::Hashed, {0}};  // on a.a0
    cat.tables["b"] = Distribution{DistributionKind::Hashed, {0}};  // on b.b0
    LoweringContext ctx;
    ctx.cardinality = &card;
    ctx.distribution = &cat;
    const LoweringResult r = lower(*q, ctx);
    CHECK(r.ok);
    if (!r.plan) { std::printf("  error: %s\n", r.error.c_str()); return; }
    CHECK(!contains(physical_to_sexpr(*r.plan), "Exchange"));
}

// Partitioned on the WRONG column, the same join has to move rows - and the plan
// says which columns it repartitions on, because a repartition that did not
// would read the same as one that sends rows to different nodes.
static void test_a_join_on_the_wrong_partition_key_repartitions() {
    std::printf("test_a_join_on_the_wrong_partition_key_repartitions\n");
    const auto q = join_ab();
    CardinalityModel card;
    card.base_rows["a"] = 1000.0;
    card.base_rows["b"] = 1000.0;
    DistributionCatalog cat;
    // `b` is partitioned on b1, but the join is on b0.
    cat.tables["a"] = Distribution{DistributionKind::Hashed, {0}};
    cat.tables["b"] = Distribution{DistributionKind::Hashed, {1}};
    LoweringContext ctx;
    ctx.cardinality = &card;
    ctx.distribution = &cat;
    const LoweringResult r = lower(*q, ctx);
    CHECK(r.ok);
    if (!r.plan) { std::printf("  error: %s\n", r.error.c_str()); return; }
    const std::string s = physical_to_sexpr(*r.plan);
    CHECK(contains(s, "Exchange to=hashed"));
    // Exactly ONE side moves. `a` is already where it needs to be.
    CHECK(count(s, "Exchange") == 1);
    // And it says what it hashes on - the right input's own key column.
    CHECK(contains(s, "Exchange to=hashed on=[#0]"));
}

// The network is PRICED, and priced far above the CPU. A plan that has to move
// rows must cost more than the same plan that does not, or the search would
// scatter data for free.
static void test_moving_rows_costs_more_than_not_moving_them() {
    std::printf("test_moving_rows_costs_more_than_not_moving_them\n");
    const auto q = join_ab();
    CardinalityModel card;
    card.base_rows["a"] = 1000.0;
    card.base_rows["b"] = 1000.0;
    const auto cost_with = [&](const Distribution& b_dist) {
        DistributionCatalog cat;
        cat.tables["a"] = Distribution{DistributionKind::Hashed, {0}};
        cat.tables["b"] = b_dist;
        LoweringContext ctx;
        ctx.cardinality = &card;
        ctx.distribution = &cat;
        const LoweringResult r = lower(*q, ctx);
        CHECK(r.ok);
        return r.plan ? cost_of(*r.plan, default_calibration(), card) : -1.0;
    };
    const double co_located = cost_with(Distribution{DistributionKind::Hashed, {0}});
    const double must_move = cost_with(Distribution{DistributionKind::Hashed, {1}});
    CHECK(co_located > 0.0 && must_move > 0.0);
    CHECK(must_move > co_located);
}

// A scalar aggregate is one row, and one row is in one place - so its input has
// to be gathered. A partial sum per node is not a sum, and this is the
// requirement that says so.
static void test_a_scalar_aggregate_gathers_its_input() {
    std::printf("test_a_scalar_aggregate_gathers_its_input\n");
    auto s = scan("a");
    auto agg = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Aggregate);
    agg->output = {{"SUM", DataType::Integer, true}};
    auto sum = std::make_unique<plan::Expr>(plan::ExprKind::Aggregate);
    sum->func_name = "SUM";
    sum->children.push_back(col(0));
    agg->aggregates.push_back(std::move(sum));
    agg->add_child(std::move(s));

    CardinalityModel card;
    card.base_rows["a"] = 1000.0;
    DistributionCatalog cat;
    cat.tables["a"] = Distribution{DistributionKind::Hashed, {0}};
    LoweringContext ctx;
    ctx.cardinality = &card;
    ctx.distribution = &cat;
    const LoweringResult r = lower(*agg, ctx);
    CHECK(r.ok);
    if (!r.plan) { std::printf("  error: %s\n", r.error.c_str()); return; }
    CHECK(contains(physical_to_sexpr(*r.plan), "Exchange to=single"));
}

// A GROUPED aggregate does not gather: it needs each group on one node, which a
// table already partitioned on the grouping key satisfies. The distinction
// between the two is the whole reason distribution is derived per operator
// rather than assumed.
static void test_a_grouped_aggregate_does_not_gather_when_already_partitioned() {
    std::printf("test_a_grouped_aggregate_does_not_gather_when_already_partitioned\n");
    auto s = scan("a");
    auto agg = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Aggregate);
    agg->output = {{"a0", DataType::Integer, false}, {"SUM", DataType::Integer, true}};
    agg->group_keys.push_back(col(0));
    auto sum = std::make_unique<plan::Expr>(plan::ExprKind::Aggregate);
    sum->func_name = "SUM";
    sum->children.push_back(col(1));
    agg->aggregates.push_back(std::move(sum));
    agg->add_child(std::move(s));

    CardinalityModel card;
    card.base_rows["a"] = 1000.0;
    DistributionCatalog cat;
    cat.tables["a"] = Distribution{DistributionKind::Hashed, {0}};  // the grouping key
    LoweringContext ctx;
    ctx.cardinality = &card;
    ctx.distribution = &cat;
    const LoweringResult r = lower(*agg, ctx);
    CHECK(r.ok);
    if (!r.plan) { std::printf("  error: %s\n", r.error.c_str()); return; }
    CHECK(!contains(physical_to_sexpr(*r.plan), "Exchange"));

    // Partitioned on the OTHER column, the same aggregate has to repartition.
    DistributionCatalog wrong;
    wrong.tables["a"] = Distribution{DistributionKind::Hashed, {1}};
    LoweringContext ctx2 = ctx;
    ctx2.distribution = &wrong;
    const LoweringResult r2 = lower(*agg, ctx2);
    CHECK(r2.ok);
    if (r2.plan) CHECK(contains(physical_to_sexpr(*r2.plan), "Exchange to=hashed"));
}

// A repartition scatters rows across nodes, and no order survives that. An
// enforcer that established an order and then moved the rows would produce a
// plan whose stated property is not true of its output - the defect this planner
// keeps finding, one property over.
static void test_a_repartition_destroys_the_order_it_moves() {
    std::printf("test_a_repartition_destroys_the_order_it_moves\n");
    auto s = scan("a");
    // The input must ALREADY be partitioned, and on the wrong column, or there is
    // nothing to move: rows all in one place are trivially co-located by
    // anything, so a `Single` input satisfies every hashed requirement and no
    // exchange is inserted. (The first draft of this test missed that and was
    // asserting against a plan with no Exchange in it at all.)
    const auto partitioned_scan = [&] {
        PhysicalNodePtr n = make_seq_scan("a", s->output);
        n->scan_distribution = Distribution{DistributionKind::Hashed, {0}};
        return n;
    };
    PhysicalNodePtr sorted = make_sort(partitioned_scan(), {SortKey{0, false, false, false}});
    CHECK(!derive(*sorted).sort.empty());
    PhysicalProperties want;
    want.distribution = Distribution{DistributionKind::Hashed, {1}};
    PhysicalNodePtr moved = enforce(std::move(sorted), want);
    CHECK(moved != nullptr);
    if (moved == nullptr) return;
    CHECK(moved->op == PhysicalOp::Exchange);
    CHECK(derive(*moved).sort.empty());

    // And asking for BOTH puts the sort ABOVE the exchange, not below it - so the
    // plan that comes out actually has the order it claims. Enforcing the other
    // way round would establish an order and then destroy it.
    PhysicalProperties both;
    both.distribution = Distribution{DistributionKind::Hashed, {1}};
    both.sort = {SortKey{0, false, false, false}};
    PhysicalNodePtr fixed = enforce(partitioned_scan(), both);
    CHECK(fixed != nullptr);
    if (fixed == nullptr) return;
    CHECK(satisfies(derive(*fixed), both));
    CHECK(fixed->op == PhysicalOp::Sort);
    CHECK(!fixed->children.empty() && fixed->children[0]->op == PhysicalOp::Exchange);
}

// An Exchange ends a pipeline without buffering anything: the sending side is one
// loop and the receiving side another, on a different node. `separate`, not
// `materialized` - telling a reader that moving rows allocates would be a lie
// about where the memory goes.
static void test_an_exchange_ends_a_pipeline_without_buffering() {
    std::printf("test_an_exchange_ends_a_pipeline_without_buffering\n");
    CHECK(edge_kind(PhysicalOp::Exchange, 0, true) == EdgeKind::Separate);
    auto s = scan("a");
    PhysicalProperties want;
    want.distribution = Distribution{DistributionKind::Single, {}};
    PhysicalNodePtr p = make_exchange(make_seq_scan("a", s->output),
                                      Distribution{DistributionKind::Single, {}});
    const auto ps = pipelines(*p);
    CHECK(ps.size() == 2);  // the scan's loop, then the exchange's
}

int main() {
    test_hashing_on_fewer_columns_is_the_stronger_guarantee();
    test_the_single_node_default_inserts_no_exchange();
    test_co_located_tables_join_without_moving_a_row();
    test_a_join_on_the_wrong_partition_key_repartitions();
    test_moving_rows_costs_more_than_not_moving_them();
    test_a_scalar_aggregate_gathers_its_input();
    test_a_grouped_aggregate_does_not_gather_when_already_partitioned();
    test_a_repartition_destroys_the_order_it_moves();
    test_an_exchange_ends_a_pipeline_without_buffering();

    if (g_failures == 0) {
        std::printf("distribution tests: all passed\n");
        return 0;
    }
    std::printf("distribution tests: %d failure(s)\n", g_failures);
    return 1;
}
