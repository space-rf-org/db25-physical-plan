// Increment 5.1: the plan cache, and the key that makes one safe.
//
// The interesting tests here are the MISSES. A cache that hits when it should is
// a speedup; a cache that hits when it should not returns a plan chosen under a
// different cost model, a different catalog, or for a different query, and calls
// it correct. So the sweep below changes one declared input at a time and
// requires the key to move - the same discipline as the rendering sweep, asking
// the same question one layer up.
#include "db25/physical/distribution_catalog.hpp"
#include "db25/physical/lowering.hpp"
#include "db25/physical/plan_cache.hpp"
#include "db25/physical/sexpr.hpp"
#include "db25/physical/spec.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <utility>
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

#ifndef DB25_PHYSICAL_SPEC_DIR
#define DB25_PHYSICAL_SPEC_DIR "."
#endif
#ifndef DB25_PLAN_EXPR_HEADER
#define DB25_PLAN_EXPR_HEADER "expr_ir.hpp"
#endif
#ifndef DB25_PLAN_NODE_HEADER
#define DB25_PLAN_NODE_HEADER "logical_plan.hpp"
#endif

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
static plan::ExprPtr param(std::uint32_t i) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::Parameter);
    e->type = DataType::Integer;
    e->param_index = i;
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
// `SELECT * FROM a JOIN b ON a.a0 = b.b0 WHERE a.a1 > <bound>`.
static plan::LogicalNodePtr query(plan::ExprPtr bound) {
    auto sa = scan("a");
    auto sb = scan("b");
    auto j = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Join);
    j->join_type = db25::ast::JoinType::Inner;
    j->output = sa->output;
    for (const auto& c : sb->output) j->output.push_back(c);
    j->predicate = binop(BinaryOp::Equal, col(0), col(2));
    j->add_child(std::move(sa));
    j->add_child(std::move(sb));
    auto f = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
    f->output = j->output;
    f->predicate = binop(BinaryOp::GreaterThan, col(1), std::move(bound));
    f->add_child(std::move(j));
    return f;
}

// ---- hits --------------------------------------------------------------

// The point of the thing: plan once, answer twice. And the second answer is the
// SAME plan, not merely an equal one - the cache hands back what it owns.
static void test_the_same_query_hits_and_returns_the_same_plan() {
    std::printf("test_the_same_query_hits_and_returns_the_same_plan\n");
    PlanCache cache;
    LoweringContext ctx;
    auto q1 = query(int_lit(10));
    const PlanCacheKey k1 = plan_cache_key(*q1, ctx);
    CHECK(cache.get(k1) == nullptr);  // cold

    LoweringResult r = lower(*q1, ctx);
    CHECK(r.ok);
    const std::string first = r.plan ? physical_to_sexpr(*r.plan) : std::string{};
    cache.put(k1, std::move(q1), std::move(r));

    // A second, INDEPENDENTLY BUILT copy of the same query.
    auto q2 = query(int_lit(10));
    const PlanCacheKey k2 = plan_cache_key(*q2, ctx);
    CHECK(k1 == k2);
    const CachedPlan* hit = cache.get(k2);
    CHECK(hit != nullptr);
    if (hit == nullptr) return;
    CHECK(hit->result.plan != nullptr);
    if (hit->result.plan) CHECK(physical_to_sexpr(*hit->result.plan) == first);
    CHECK(cache.hits() == 1);
}

// THE OWNERSHIP CONSTRAINT, tested rather than trusted. A physical plan borrows
// its expressions from the logical plan, so an entry has to own both - otherwise
// a hit hands back a plan pointing at freed memory. Here the caller's own logical
// plan is destroyed before the hit is read, which is exactly the situation a
// cache creates and the reason CachedPlan holds two things.
static void test_a_hit_survives_the_callers_logical_plan() {
    std::printf("test_a_hit_survives_the_callers_logical_plan\n");
    PlanCache cache;
    LoweringContext ctx;
    PlanCacheKey key;
    std::string expected;
    {
        auto q = query(int_lit(10));
        key = plan_cache_key(*q, ctx);
        LoweringResult r = lower(*q, ctx);
        CHECK(r.ok);
        expected = r.plan ? physical_to_sexpr(*r.plan) : std::string{};
        cache.put(key, std::move(q), std::move(r));
        // `q` is now null - the cache owns it. Anything the caller still held
        // would be the thing that dangles, and it is deliberately not held.
    }
    const CachedPlan* hit = cache.get(key);
    CHECK(hit != nullptr);
    // Rendering walks every borrowed expression. Under ASan this reads freed
    // memory if the entry did not own the logical plan.
    if (hit != nullptr && hit->result.plan != nullptr) {
        CHECK(physical_to_sexpr(*hit->result.plan) == expected);
    }
}

// A PARAMETER carries no value, so two prepared statements that differ only in
// what will be bound to them are the same query - which is what makes a prepared
// statement's plan reusable. Nothing has to be abstracted for this: the IR
// already distinguishes a parameter from a literal.
static void test_parameters_are_already_separate() {
    std::printf("test_parameters_are_already_separate\n");
    LoweringContext ctx;
    const auto a = query(param(0));
    const auto b = query(param(0));
    CHECK(plan_cache_key(*a, ctx) == plan_cache_key(*b, ctx));
    // A different parameter SLOT is a different query - it reads a different
    // value at execution.
    const auto c = query(param(1));
    CHECK(!(plan_cache_key(*a, ctx) == plan_cache_key(*c, ctx)));
    // And a parameter is not a literal, whatever the literal's value.
    const auto d = query(int_lit(0));
    CHECK(!(plan_cache_key(*a, ctx) == plan_cache_key(*d, ctx)));
}

// A LITERAL is part of the key, and must be. The physical plan RENDERS it, so
// returning a plan built for `> 10` to a query saying `> 20` would hand back a
// plan that filters on the wrong constant - the plan is not a shape, it is a
// plan.
static void test_a_different_literal_is_a_different_query() {
    std::printf("test_a_different_literal_is_a_different_query\n");
    LoweringContext ctx;
    const auto ten = query(int_lit(10));
    const auto twenty = query(int_lit(20));
    CHECK(!(plan_cache_key(*ten, ctx) == plan_cache_key(*twenty, ctx)));

    // And the plans really do differ, which is why the key has to.
    const LoweringResult a = lower(*ten, ctx);
    const LoweringResult b = lower(*twenty, ctx);
    CHECK(a.ok && b.ok);
    if (a.plan && b.plan) {
        CHECK(physical_to_sexpr(*a.plan) != physical_to_sexpr(*b.plan));
    }
}

// ---- the misses that matter --------------------------------------------

// One declared input at a time. Each must move the key, because each can change
// what the planner chooses - and a key that missed one would return a plan
// optimized for something else and report a hit.
struct InputCase {
    const char* what;
    std::function<void(LoweringContext&, CalibrationProfile&, CardinalityModel&,
                       StorageCatalog&, DistributionCatalog&)> change;
};

static void test_every_planning_input_moves_the_key() {
    std::printf("test_every_planning_input_moves_the_key\n");
    const auto q = query(int_lit(10));
    std::string spec_error;
    auto spec = load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr",
                          spec_error);
    CHECK(spec.has_value());

    const std::vector<InputCase> cases{
        {"calibration coefficient",
         [](LoweringContext&, CalibrationProfile& cal, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) { cal.hash_probe_row += 0.1; }},
        {"cluster size",
         [](LoweringContext&, CalibrationProfile& cal, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) { cal.cluster_nodes = 4; }},
        {"a table's row count",
         [](LoweringContext&, CalibrationProfile&, CardinalityModel& card, StorageCatalog&,
            DistributionCatalog&) { card.base_rows["a"] = 12345.0; }},
        {"a default selectivity",
         [](LoweringContext&, CalibrationProfile&, CardinalityModel& card, StorageCatalog&,
            DistributionCatalog&) { card.join_selectivity = 0.42; }},
        {"the recursion assumption",
         [](LoweringContext&, CalibrationProfile&, CardinalityModel& card, StorageCatalog&,
            DistributionCatalog&) { card.recursive_iterations = 25.0; }},
        {"a table's storage formats",
         [](LoweringContext&, CalibrationProfile&, CardinalityModel&, StorageCatalog& st,
            DistributionCatalog&) {
             st.formats["a"] = {FormatAvailability{StorageFormat::Column}};
         }},
        {"a table's distribution",
         [](LoweringContext&, CalibrationProfile&, CardinalityModel&, StorageCatalog&,
            DistributionCatalog& d) {
             d.tables["a"] = Distribution{DistributionKind::Hashed, {0}};
         }},
        {"the required freshness",
         [](LoweringContext& c, CalibrationProfile&, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) { c.required_freshness = Freshness::Fresh; }},
        {"the required output order",
         [](LoweringContext& c, CalibrationProfile&, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) {
             c.required_output.sort = {SortKey{0, false, false, false}};
         }},
        {"the required output distribution",
         [](LoweringContext& c, CalibrationProfile&, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) {
             c.required_output.distribution = Distribution{DistributionKind::Broadcast, {}};
         }},
        {"branch-and-bound pruning",
         [](LoweringContext& c, CalibrationProfile&, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) { c.prune = false; }},
        {"memo dedup",
         [](LoweringContext& c, CalibrationProfile&, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) { c.dedup = true; }},
        {"join reordering",
         [](LoweringContext& c, CalibrationProfile&, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) { c.reorder_joins = false; }},
        {"the search budget override",
         [](LoweringContext& c, CalibrationProfile&, CardinalityModel&, StorageCatalog&,
            DistributionCatalog&) { c.max_join_count_override = 2; }},
    };

    // The baseline: nothing changed.
    CalibrationProfile cal0 = default_calibration();
    CardinalityModel card0;
    StorageCatalog st0;
    DistributionCatalog d0;
    LoweringContext base;
    base.calibration = &cal0;
    base.cardinality = &card0;
    base.storage = &st0;
    base.distribution = &d0;
    const PlanCacheKey baseline = plan_cache_key(*q, base);

    for (const InputCase& c : cases) {
        CalibrationProfile cal = default_calibration();
        CardinalityModel card;
        StorageCatalog st;
        DistributionCatalog d;
        LoweringContext ctx;
        ctx.calibration = &cal;
        ctx.cardinality = &card;
        ctx.storage = &st;
        ctx.distribution = &d;
        c.change(ctx, cal, card, st, d);
        const PlanCacheKey moved = plan_cache_key(*q, ctx);
        if (moved == baseline) {
            std::printf("  %s does not move the cache key - a plan chosen under a "
                        "different one would be returned as a hit\n", c.what);
        }
        CHECK(!(moved == baseline));
    }

    // The SPEC too: a different rule set is a different planner.
    if (spec) {
        LoweringContext with_spec = base;
        with_spec.spec = &*spec;
        CHECK(!(plan_cache_key(*q, with_spec) == baseline));
    }
}

// And the structure half: changing the query moves the structure digest, not the
// inputs digest. Two digests rather than one exactly so a diagnostic can say
// which of them moved.
static void test_the_two_halves_of_the_key_move_independently() {
    std::printf("test_the_two_halves_of_the_key_move_independently\n");
    CalibrationProfile cal = default_calibration();
    LoweringContext ctx;
    ctx.calibration = &cal;
    const auto a = query(int_lit(10));
    const auto b = query(int_lit(20));
    const PlanCacheKey ka = plan_cache_key(*a, ctx);
    const PlanCacheKey kb = plan_cache_key(*b, ctx);
    CHECK(ka.structure != kb.structure);
    CHECK(ka.inputs == kb.inputs);  // the query changed, not the inputs

    CalibrationProfile other = default_calibration();
    other.sort_row += 1.0;
    LoweringContext ctx2;
    ctx2.calibration = &other;
    const PlanCacheKey kc = plan_cache_key(*a, ctx2);
    CHECK(kc.structure == ka.structure);  // the inputs changed, not the query
    CHECK(kc.inputs != ka.inputs);
}

// A key must not depend on the ITERATION ORDER of an unordered_map. Two
// catalogs with the same contents built by inserting in different orders are the
// same catalog, and a key that disagreed would miss on every second run for no
// reason a reader could see.
static void test_the_key_does_not_depend_on_hash_map_order() {
    std::printf("test_the_key_does_not_depend_on_hash_map_order\n");
    const auto q = query(int_lit(10));
    CardinalityModel one;
    one.base_rows["a"] = 10.0;
    one.base_rows["b"] = 20.0;
    one.base_rows["c"] = 30.0;
    CardinalityModel two;
    two.base_rows["c"] = 30.0;
    two.base_rows["b"] = 20.0;
    two.base_rows["a"] = 10.0;
    LoweringContext c1;
    c1.cardinality = &one;
    LoweringContext c2;
    c2.cardinality = &two;
    CHECK(plan_cache_key(*q, c1) == plan_cache_key(*q, c2));

    // But the CONTENTS still matter.
    two.base_rows["a"] = 11.0;
    CHECK(!(plan_cache_key(*q, c1) == plan_cache_key(*q, c2)));

    // The same for the storage catalog. Three maps are folded order-independently
    // and each needs its own case: a sweep that checked one of them would pass
    // while the other two folded in sequence.
    StorageCatalog s1;
    s1.formats["a"] = {FormatAvailability{StorageFormat::Row}};
    s1.formats["b"] = {FormatAvailability{StorageFormat::Column}};
    s1.formats["c"] = {FormatAvailability{StorageFormat::Row}};
    StorageCatalog s2;
    s2.formats["c"] = {FormatAvailability{StorageFormat::Row}};
    s2.formats["b"] = {FormatAvailability{StorageFormat::Column}};
    s2.formats["a"] = {FormatAvailability{StorageFormat::Row}};
    LoweringContext s_one;
    s_one.storage = &s1;
    LoweringContext s_two;
    s_two.storage = &s2;
    CHECK(plan_cache_key(*q, s_one) == plan_cache_key(*q, s_two));
    s2.formats["a"] = {FormatAvailability{StorageFormat::Column}};
    CHECK(!(plan_cache_key(*q, s_one) == plan_cache_key(*q, s_two)));

    // And the distribution catalog.
    DistributionCatalog d1;
    d1.tables["a"] = Distribution{DistributionKind::Hashed, {0}};
    d1.tables["b"] = Distribution{DistributionKind::Broadcast, {}};
    d1.tables["c"] = Distribution{DistributionKind::Single, {}};
    DistributionCatalog d2;
    d2.tables["c"] = Distribution{DistributionKind::Single, {}};
    d2.tables["b"] = Distribution{DistributionKind::Broadcast, {}};
    d2.tables["a"] = Distribution{DistributionKind::Hashed, {0}};
    LoweringContext d_one;
    d_one.distribution = &d1;
    LoweringContext d_two;
    d_two.distribution = &d2;
    CHECK(plan_cache_key(*q, d_one) == plan_cache_key(*q, d_two));
    d2.tables["a"] = Distribution{DistributionKind::Hashed, {1}};
    CHECK(!(plan_cache_key(*q, d_one) == plan_cache_key(*q, d_two)));
}

// A subquery's inner plan is part of the query. The memo's structural key
// declines to compare one and simply never shares such a group; the cache has no
// such escape - refusing to key a query with a subquery would mean never caching
// one - so it digests the inner plan inline.
static void test_a_subquery_is_part_of_the_key() {
    std::printf("test_a_subquery_is_part_of_the_key\n");
    const auto with_sub = [](const std::string& inner_table) {
        auto sq = std::make_unique<plan::Expr>(plan::ExprKind::Subquery);
        sq->subquery_kind = plan::SubqueryKind::Exists;
        sq->type = DataType::Boolean;
        sq->sub_plan = scan(inner_table);
        auto f = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
        auto s = scan("a");
        f->output = s->output;
        f->predicate = std::move(sq);
        f->add_child(std::move(s));
        return f;
    };
    LoweringContext ctx;
    const auto over_b = with_sub("b");
    const auto over_b2 = with_sub("b");
    const auto over_c = with_sub("c");
    CHECK(plan_cache_key(*over_b, ctx) == plan_cache_key(*over_b2, ctx));
    // Two EXISTS subqueries over different tables are different queries, and a
    // key that stopped at the Subquery node would say they are the same.
    CHECK(!(plan_cache_key(*over_b, ctx) == plan_cache_key(*over_c, ctx)));
}

// ---- the field sweeps -------------------------------------------------

// Reads the member names of `struct <name>` out of a header. Crude parsing, and
// crude is enough: this is a COVERAGE check, and its job is to fail the day
// someone adds a field to plan::Expr, plan::LogicalNode or plan::ColumnSchema
// without deciding whether the cache key digests it. A field that is added and
// not keyed makes the cache return one query's plan for a different query; a
// field that is added and not swept is a field nobody ever asked about.
static std::vector<std::string> struct_members(const char* path, const std::string& name) {
    std::vector<std::string> out;
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
        std::printf("  FAIL: cannot open %s\n", path);
        ++g_failures;
        return out;
    }
    std::string text;
    char buf[4096];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    std::fclose(f);

    const std::string open = "struct " + name + " {";
    std::size_t p = text.find(open);
    if (p == std::string::npos) {
        std::printf("  FAIL: no `%s` in %s\n", open.c_str(), path);
        ++g_failures;
        return out;
    }
    p += open.size();
    while (p < text.size()) {
        std::size_t e = text.find('\n', p);
        if (e == std::string::npos) e = text.size();
        std::string line = text.substr(p, e - p);
        p = e + 1;
        const std::size_t c = line.find("//");
        if (c != std::string::npos) line = line.substr(0, c);
        // Trim.
        std::size_t b = line.find_first_not_of(" \t");
        if (b == std::string::npos) continue;
        line = line.substr(b, line.find_last_not_of(" \t\r") - b + 1);
        if (line.empty()) continue;
        if (line == "};") break;                          // end of the struct
        if (line.find('(') != std::string::npos) break;   // first member function
        if (line.back() != ';') continue;                 // continuation / brace line
        line.pop_back();
        if (line.size() >= 2 && line.compare(line.size() - 2, 2, "{}") == 0)
            line.resize(line.size() - 2);
        const std::size_t eq = line.find('=');
        if (eq != std::string::npos) line = line.substr(0, eq);
        const std::size_t last = line.find_last_not_of(" \t");
        if (last == std::string::npos) continue;
        line = line.substr(0, last + 1);
        const std::size_t sp = line.find_last_of(" \t*&>");
        const std::string member = sp == std::string::npos ? line : line.substr(sp + 1);
        if (member.empty()) continue;
        bool ident = (std::isalpha(static_cast<unsigned char>(member[0])) != 0 || member[0] == '_');
        for (const char ch : member)
            if (std::isalnum(static_cast<unsigned char>(ch)) == 0 && ch != '_') ident = false;
        if (ident) out.push_back(member);
    }
    return out;
}

// Every case names the field it changes; the coverage check below requires the
// set of names to be exactly the struct's members, so the sweep cannot quietly
// fall behind the IR it is sweeping.
struct FieldCase {
    const char* field;
    bool keyed;  // false = deliberately NOT part of the key, and the sweep says why
    std::function<void(plan::Expr&)> change;
};

static void check_coverage(const char* what, const char* header, const std::string& sname,
                           const std::vector<std::string>& named) {
    const std::vector<std::string> members = struct_members(header, sname);
    CHECK(!members.empty());
    for (const std::string& m : members) {
        bool found = false;
        for (const std::string& c : named)
            if (c == m) found = true;
        if (!found) {
            std::printf("  %s field `%s` is swept by no case: nobody has decided whether "
                        "the plan cache key digests it\n", what, m.c_str());
            ++g_failures;
        }
    }
    for (const std::string& c : named) {
        bool found = false;
        for (const std::string& m : members)
            if (c == m) found = true;
        if (!found) {
            std::printf("  %s case names `%s`, which is not a member of %s\n", what,
                        c.c_str(), sname.c_str());
            ++g_failures;
        }
    }
}

// The Expr under test, wrapped as a Filter predicate so it is reached through a
// real plan rather than digested in isolation.
static PlanCacheKey expr_key(const std::function<void(plan::Expr&)>& tweak) {
    auto e = std::make_unique<plan::Expr>(plan::ExprKind::ScalarFunction);
    e->type = DataType::Integer;
    e->func_name = "F";
    e->children.push_back(col(0));
    tweak(*e);
    auto s = scan("a");
    auto f = std::make_unique<plan::LogicalNode>(plan::LogicalOp::Filter);
    f->output = s->output;
    f->predicate = std::move(e);
    f->add_child(std::move(s));
    LoweringContext ctx;
    return plan_cache_key(*f, ctx);
}

// One field of plan::Expr at a time. `source` is the one deliberate exclusion:
// it is a borrowed pointer into the AST, so two independently parsed copies of
// the SAME query carry different values and keying it would turn every lookup
// into a miss. The consequence is stated rather than hidden - a hit hands back
// expressions whose `source` points at the AST of the query planned FIRST, which
// is fine for a diagnostic echo and would not be fine for anything semantic.
static void test_every_expr_field_moves_the_key() {
    std::printf("test_every_expr_field_moves_the_key\n");
    const auto& key_with = expr_key;
    static const db25::ast::ASTNode* const kSomeNode =
        reinterpret_cast<const db25::ast::ASTNode*>(0x10);

    const std::vector<FieldCase> cases{
        {"kind", true, [](plan::Expr& e) { e.kind = plan::ExprKind::Aggregate; }},
        {"type", true, [](plan::Expr& e) { e.type = DataType::BigInt; }},
        {"nullability", true, [](plan::Expr& e) { e.nullability = 2; }},
        {"children", true, [](plan::Expr& e) { e.children.push_back(col(1)); }},
        {"source", false, [](plan::Expr& e) { e.source = kSomeNode; }},
        {"input_index", true, [](plan::Expr& e) { e.input_index = 7; }},
        {"ref_table_id", true, [](plan::Expr& e) { e.ref_table_id = 3; }},
        {"ref_column_id", true, [](plan::Expr& e) { e.ref_column_id = 4; }},
        {"outer_depth", true, [](plan::Expr& e) { e.outer_depth = 1; }},
        {"value", true, [](plan::Expr& e) { e.value.value = std::int64_t{5}; }},
        {"bin_op", true, [](plan::Expr& e) { e.bin_op = BinaryOp::Subtract; }},
        {"un_op", true, [](plan::Expr& e) { e.un_op = db25::ast::UnaryOp::Negate; }},
        {"func_name", true, [](plan::Expr& e) { e.func_name = "G"; }},
        {"distinct", true, [](plan::Expr& e) { e.distinct = true; }},
        {"window", true, [](plan::Expr& e) { e.window.partition_by.push_back(col(1)); }},
        {"filter", true, [](plan::Expr& e) { e.filter = col(1); }},
        {"agg_order_by", true,
         [](plan::Expr& e) {
             plan::SortKeyIR k;
             k.expr = col(1);
             e.agg_order_by.push_back(std::move(k));
         }},
        {"target_type", true, [](plan::Expr& e) { e.target_type = DataType::Double; }},
        {"type_precision", true, [](plan::Expr& e) { e.type_precision = 10; }},
        {"type_scale", true, [](plan::Expr& e) { e.type_scale = 2; }},
        {"type_length", true, [](plan::Expr& e) { e.type_length = 32; }},
        {"expr_flags", true, [](plan::Expr& e) { e.expr_flags = plan::ExprFlagNegated; }},
        {"bool_test", true, [](plan::Expr& e) { e.bool_test = plan::BoolTest::Unknown; }},
        {"subquery_kind", true,
         [](plan::Expr& e) { e.subquery_kind = plan::SubqueryKind::Exists; }},
        {"correlated", true, [](plan::Expr& e) { e.correlated = true; }},
        {"sub_plan", true, [](plan::Expr& e) { e.sub_plan = scan("z"); }},
        {"param_index", true, [](plan::Expr& e) { e.param_index = 3; }},
    };

    const PlanCacheKey baseline = key_with([](plan::Expr&) {});
    std::vector<std::string> named;
    for (const FieldCase& c : cases) {
        named.emplace_back(c.field);
        const PlanCacheKey moved = key_with(c.change);
        if (c.keyed && moved == baseline) {
            std::printf("  Expr::%s does not move the cache key: two queries differing "
                        "only in it would share a plan\n", c.field);
            ++g_failures;
        }
        if (!c.keyed && !(moved == baseline)) {
            std::printf("  Expr::%s moves the cache key, but is declared not keyed\n",
                        c.field);
            ++g_failures;
        }
    }
    check_coverage("plan::Expr", DB25_PLAN_EXPR_HEADER, "Expr", named);
}

struct NodeCase {
    const char* field;
    std::function<void(plan::LogicalNode&)> change;
};

// And one field of plan::LogicalNode at a time. LIMIT is the illustrative one:
// it looks like an incidental constant and is not - the cardinality model reads
// it, so `LIMIT 10` and `LIMIT 1000000` can get different plans, and a key that
// dropped it would hand the first query's plan to the second.
static void test_every_logical_node_field_moves_the_key() {
    std::printf("test_every_logical_node_field_moves_the_key\n");
    const auto key_with = [](const std::function<void(plan::LogicalNode&)>& tweak) {
        auto n = scan("a");
        tweak(*n);
        LoweringContext ctx;
        return plan_cache_key(*n, ctx);
    };

    const std::vector<NodeCase> cases{
        {"op", [](plan::LogicalNode& n) { n.op = plan::LogicalOp::Filter; }},
        {"children", [](plan::LogicalNode& n) { n.add_child(scan("b")); }},
        {"output",
         [](plan::LogicalNode& n) {
             n.output.push_back({"extra", DataType::Integer, true, 0, 0, "a", false});
         }},
        {"table_name", [](plan::LogicalNode& n) { n.table_name = "z"; }},
        {"alias", [](plan::LogicalNode& n) { n.alias = "x"; }},
        {"predicate",
         [](plan::LogicalNode& n) { n.predicate = binop(BinaryOp::Equal, col(0), int_lit(1)); }},
        {"join_type", [](plan::LogicalNode& n) { n.join_type = db25::ast::JoinType::Left; }},
        {"exprs", [](plan::LogicalNode& n) { n.exprs.push_back(col(0)); }},
        {"group_keys", [](plan::LogicalNode& n) { n.group_keys.push_back(col(0)); }},
        {"aggregates", [](plan::LogicalNode& n) { n.aggregates.push_back(col(0)); }},
        {"grouping_sets", [](plan::LogicalNode& n) { n.grouping_sets.push_back({0}); }},
        {"window_functions", [](plan::LogicalNode& n) { n.window_functions.push_back(col(0)); }},
        {"sort_keys",
         [](plan::LogicalNode& n) {
             plan::SortKeyIR k;
             k.expr = col(0);
             n.sort_keys.push_back(std::move(k));
         }},
        {"value_rows",
         [](plan::LogicalNode& n) {
             std::vector<plan::ExprPtr> row;
             row.push_back(int_lit(1));
             n.value_rows.push_back(std::move(row));
         }},
        {"has_limit", [](plan::LogicalNode& n) { n.has_limit = true; }},
        {"limit", [](plan::LogicalNode& n) { n.limit = 5; }},
        {"has_offset", [](plan::LogicalNode& n) { n.has_offset = true; }},
        {"offset", [](plan::LogicalNode& n) { n.offset = 3; }},
        {"set_op", [](plan::LogicalNode& n) { n.set_op = db25::ast::SetOp::Except; }},
        {"target_columns", [](plan::LogicalNode& n) { n.target_columns.emplace_back("c"); }},
        {"assignments",
         [](plan::LogicalNode& n) {
             plan::Assignment a;
             a.target_column_id = 1;
             a.value = int_lit(1);
             n.assignments.push_back(std::move(a));
         }},
        {"conflict_action",
         [](plan::LogicalNode& n) { n.conflict_action = plan::ConflictAction::DoNothing; }},
        {"conflict_columns", [](plan::LogicalNode& n) { n.conflict_columns.emplace_back("c"); }},
    };

    const PlanCacheKey baseline = key_with([](plan::LogicalNode&) {});
    std::vector<std::string> named;
    for (const NodeCase& c : cases) {
        named.emplace_back(c.field);
        if (key_with(c.change) == baseline) {
            std::printf("  LogicalNode::%s does not move the cache key: two queries "
                        "differing only in it would share a plan\n", c.field);
            ++g_failures;
        }
    }
    check_coverage("plan::LogicalNode", DB25_PLAN_NODE_HEADER, "LogicalNode", named);
}

// The output SCHEMA is carried into the physical plan and handed back on a hit,
// so every column attribute has to be in the key - including `hidden`, which
// looks like a bind-time detail and is not: a hit returns the schema the CACHE
// owns, so a difference the key ignored would be a difference the caller sees.
static void test_every_column_schema_field_moves_the_key() {
    std::printf("test_every_column_schema_field_moves_the_key\n");
    const auto key_with = [](const std::function<void(plan::ColumnSchema&)>& tweak) {
        auto n = scan("a");
        tweak(n->output[0]);
        LoweringContext ctx;
        return plan_cache_key(*n, ctx);
    };
    const std::vector<std::pair<const char*, std::function<void(plan::ColumnSchema&)>>> cases{
        {"name", [](plan::ColumnSchema& c) { c.name = "renamed"; }},
        {"type", [](plan::ColumnSchema& c) { c.type = DataType::Double; }},
        {"nullable", [](plan::ColumnSchema& c) { c.nullable = !c.nullable; }},
        {"table_id", [](plan::ColumnSchema& c) { c.table_id = 9; }},
        {"column_id", [](plan::ColumnSchema& c) { c.column_id = 9; }},
        {"alias", [](plan::ColumnSchema& c) { c.alias = "other"; }},
        {"hidden", [](plan::ColumnSchema& c) { c.hidden = true; }},
    };
    const PlanCacheKey baseline = key_with([](plan::ColumnSchema&) {});
    std::vector<std::string> named;
    for (const auto& [field, change] : cases) {
        named.emplace_back(field);
        if (key_with(change) == baseline) {
            std::printf("  ColumnSchema::%s does not move the cache key\n", field);
            ++g_failures;
        }
    }
    check_coverage("plan::ColumnSchema", DB25_PLAN_NODE_HEADER, "ColumnSchema", named);
}


// A field whose type is a STRUCT is one sweep case and several fields. The Expr
// sweep changing `window` proves the window spec is reached; it does not prove
// that the frame text inside it is - and a plan cache that ignored the frame
// would hand a ROWS plan to a RANGE query. So each nested IR struct gets its own
// sweep, with its own coverage check against its own declaration.
static void test_every_nested_ir_field_moves_the_key() {
    std::printf("test_every_nested_ir_field_moves_the_key\n");

    const auto sweep = [](const char* what, const char* header, const char* sname,
                          const std::function<void(plan::Expr&)>& base,
                          const std::vector<std::pair<const char*,
                                                      std::function<void(plan::Expr&)>>>& cases) {
        const PlanCacheKey baseline = expr_key(base);
        std::vector<std::string> named;
        for (const auto& [field, change] : cases) {
            named.emplace_back(field);
            const auto both = [&](plan::Expr& e) { base(e); change(e); };
            if (expr_key(both) == baseline) {
                std::printf("  %s::%s does not move the cache key\n", sname, field);
                ++g_failures;
            }
        }
        check_coverage(what, header, sname, named);
    };

    // WindowSpecIR: the OVER clause.
    sweep("plan::WindowSpecIR", DB25_PLAN_EXPR_HEADER, "WindowSpecIR",
          [](plan::Expr& e) {
              e.kind = plan::ExprKind::WindowFunction;
              e.window.partition_by.push_back(col(0));
              plan::SortKeyIR k;
              k.expr = col(1);
              e.window.order_by.push_back(std::move(k));
          },
          {{"partition_by", [](plan::Expr& e) { e.window.partition_by.push_back(col(1)); }},
           {"order_by",
            [](plan::Expr& e) {
                plan::SortKeyIR k;
                k.expr = col(0);
                e.window.order_by.push_back(std::move(k));
            }},
           {"frame", [](plan::Expr& e) { e.window.frame.present = true; }}});

    // FrameIR: `present` and the raw frame text. Two windows differing only in
    // `ROWS` versus `RANGE` are different windows.
    sweep("plan::FrameIR", DB25_PLAN_EXPR_HEADER, "FrameIR",
          [](plan::Expr& e) {
              e.kind = plan::ExprKind::WindowFunction;
              e.window.frame.present = true;
              e.window.frame.spec = "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW";
          },
          {{"present", [](plan::Expr& e) { e.window.frame.present = false; }},
           {"spec",
            [](plan::Expr& e) {
                e.window.frame.spec = "RANGE BETWEEN 1 PRECEDING AND CURRENT ROW";
            }}});

    // SortKeyIR: reached through an ordered aggregate, which exercises the same
    // hash_sort_key() the window ORDER BY and the node's sort keys use.
    sweep("plan::SortKeyIR", DB25_PLAN_NODE_HEADER, "SortKeyIR",
          [](plan::Expr& e) {
              e.kind = plan::ExprKind::Aggregate;
              plan::SortKeyIR k;
              k.expr = col(0);
              e.agg_order_by.push_back(std::move(k));
          },
          {{"expr", [](plan::Expr& e) { e.agg_order_by[0].expr = col(1); }},
           {"descending", [](plan::Expr& e) { e.agg_order_by[0].descending = true; }},
           {"nulls_order_explicit",
            [](plan::Expr& e) { e.agg_order_by[0].nulls_order_explicit = true; }},
           {"nulls_first", [](plan::Expr& e) { e.agg_order_by[0].nulls_first = true; }}});

    // LiteralValue holds one member, a variant - so the sweep that matters is over
    // its ARMS. The arm is keyed as well as the value: the integer 1 and the
    // boolean true are different literals and a plan renders them differently.
    sweep("plan::LiteralValue", DB25_PLAN_EXPR_HEADER, "LiteralValue",
          [](plan::Expr& e) {
              e.kind = plan::ExprKind::Literal;
              e.value.value = std::int64_t{1};
          },
          {{"value", [](plan::Expr& e) { e.value.value = std::int64_t{2}; }}});
    const auto lit = [](const plan::LiteralValue& v) {
        return expr_key([&](plan::Expr& e) {
            e.kind = plan::ExprKind::Literal;
            e.value = v;
        });
    };
    plan::LiteralValue null_v;
    plan::LiteralValue int_v;
    int_v.value = std::int64_t{1};
    plan::LiteralValue dbl_v;
    dbl_v.value = 1.0;
    plan::LiteralValue str_v;
    str_v.value = std::string("1");
    plan::LiteralValue bool_v;
    bool_v.value = true;
    const std::vector<PlanCacheKey> arms{lit(null_v), lit(int_v), lit(dbl_v), lit(str_v),
                                         lit(bool_v)};
    for (std::size_t i = 0; i < arms.size(); ++i)
        for (std::size_t j = i + 1; j < arms.size(); ++j) CHECK(!(arms[i] == arms[j]));
    // And within an arm: two doubles a rounding apart are two different plans.
    plan::LiteralValue dbl2;
    dbl2.value = std::nextafter(1.0, 2.0);
    CHECK(!(lit(dbl_v) == lit(dbl2)));
    plan::LiteralValue false_v;
    false_v.value = false;
    CHECK(!(lit(bool_v) == lit(false_v)));

    // Assignment: an UPDATE's SET list, reached through the node.
    const auto assign_key = [](const std::function<void(plan::Assignment&)>& tweak) {
        auto n = scan("a");
        n->op = plan::LogicalOp::Update;
        plan::Assignment a;
        a.target_column_id = 1;
        a.value = int_lit(1);
        tweak(a);
        n->assignments.push_back(std::move(a));
        LoweringContext ctx;
        return plan_cache_key(*n, ctx);
    };
    const PlanCacheKey a_base = assign_key([](plan::Assignment&) {});
    std::vector<std::string> a_named;
    const std::vector<std::pair<const char*, std::function<void(plan::Assignment&)>>> a_cases{
        {"target_column_id", [](plan::Assignment& a) { a.target_column_id = 2; }},
        {"value", [](plan::Assignment& a) { a.value = int_lit(2); }},
    };
    for (const auto& [field, change] : a_cases) {
        a_named.emplace_back(field);
        if (assign_key(change) == a_base) {
            std::printf("  plan::Assignment::%s does not move the cache key\n", field);
            ++g_failures;
        }
    }
    check_coverage("plan::Assignment", DB25_PLAN_NODE_HEADER, "Assignment", a_named);
}

int main() {
    test_the_same_query_hits_and_returns_the_same_plan();
    test_a_hit_survives_the_callers_logical_plan();
    test_parameters_are_already_separate();
    test_a_different_literal_is_a_different_query();
    test_every_planning_input_moves_the_key();
    test_the_two_halves_of_the_key_move_independently();
    test_the_key_does_not_depend_on_hash_map_order();
    test_a_subquery_is_part_of_the_key();
    test_every_expr_field_moves_the_key();
    test_every_logical_node_field_moves_the_key();
    test_every_column_schema_field_moves_the_key();
    test_every_nested_ir_field_moves_the_key();

    if (g_failures == 0) {
        std::printf("plan cache tests: all passed\n");
        return 0;
    }
    std::printf("plan cache tests: %d failure(s)\n", g_failures);
    return 1;
}
