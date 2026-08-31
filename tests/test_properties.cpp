// Unit 0.5: physical properties and the enforcer framework (design D4).
//
// Properties (sort order + storage format) are DERIVED bottom-up; where a
// property is REQUIRED but not provided, the minimal ENFORCER (Sort,
// FormatConvert) is inserted. Storage format is the HTAP substrate handled by the
// same machinery as sort order. The tests check derivation, the satisfies
// relation, that enforcement inserts exactly the enforcers needed (and nothing
// when already satisfied), that enforcement costs more, and that the enforcers
// render.
#include "db25/physical/cost.hpp"
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/properties.hpp"
#include "db25/physical/sexpr.hpp"

#include <limits>

#include <cstdio>
#include <string>

using namespace db25::physical;

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

static void test_derive() {
    std::printf("test_derive\n");
    {  // A column-store scan derives {unsorted, column}.
        auto scan = make_seq_scan("t", {});
        scan->scan_format = StorageFormat::Column;
        const PhysicalProperties p = derive(*scan);
        CHECK(p.format == StorageFormat::Column);
        CHECK(p.sort.empty());
    }
    {  // Sort establishes its order, preserves the child's (row) format.
        auto sorted = make_sort(make_seq_scan("t", {}), {{1, false}});
        const PhysicalProperties p = derive(*sorted);
        CHECK(p.sort.size() == 1);
        CHECK(p.sort[0].column == 1);
        CHECK(p.format == StorageFormat::Row);
    }
    {  // FormatConvert changes format, preserves order (child unsorted here).
        auto conv = make_format_convert(make_seq_scan("t", {}), StorageFormat::Column);
        const PhysicalProperties p = derive(*conv);
        CHECK(p.format == StorageFormat::Column);
        CHECK(p.sort.empty());
    }
    {  // A hash join outputs unsorted row-format rows.
        auto join = make_hash_join(make_seq_scan("a", {}), make_seq_scan("b", {}),
                                   {{0, 0}}, {});
        const PhysicalProperties p = derive(*join);
        CHECK(p.sort.empty());
        CHECK(p.format == StorageFormat::Row);
    }
}

static void test_satisfies() {
    std::printf("test_satisfies\n");
    PhysicalProperties provided;
    provided.format = StorageFormat::Row;
    provided.sort = {{1, false}, {2, false}};

    CHECK(satisfies(provided, PhysicalProperties{}));  // no requirement

    PhysicalProperties req_prefix;
    req_prefix.sort = {{1, false}};
    CHECK(satisfies(provided, req_prefix));  // prefix of provided order

    PhysicalProperties req_fmt;
    req_fmt.format = StorageFormat::Column;
    CHECK(!satisfies(provided, req_fmt));  // wrong format

    PhysicalProperties req_longer;
    req_longer.sort = {{1, false}, {2, false}, {3, false}};
    CHECK(!satisfies(provided, req_longer));  // more specific than provided

    PhysicalProperties req_wrong_dir;
    req_wrong_dir.sort = {{1, true}};
    CHECK(!satisfies(provided, req_wrong_dir));  // wrong direction
}

static void test_enforce_inserts_minimal_enforcers() {
    std::printf("test_enforce_inserts_minimal_enforcers\n");
    {  // Sort required, not provided -> a Sort is inserted.
        PhysicalProperties req;
        req.sort = {{0, false}};
        auto enforced = enforce(make_seq_scan("t", {}), req);
        CHECK(enforced->op == PhysicalOp::Sort);
        CHECK(satisfies(derive(*enforced), req));
    }
    {  // Format required, not provided -> a FormatConvert is inserted.
        PhysicalProperties req;
        req.format = StorageFormat::Column;
        auto enforced = enforce(make_seq_scan("t", {}), req);
        CHECK(enforced->op == PhysicalOp::FormatConvert);
        CHECK(satisfies(derive(*enforced), req));
    }
    {  // Both required -> FormatConvert then Sort (sort runs on the target format).
        PhysicalProperties req;
        req.format = StorageFormat::Column;
        req.sort = {{0, false}};
        auto enforced = enforce(make_seq_scan("t", {}), req);
        CHECK(enforced->op == PhysicalOp::Sort);
        CHECK(enforced->children.size() == 1);
        CHECK(enforced->children[0]->op == PhysicalOp::FormatConvert);
        CHECK(satisfies(derive(*enforced), req));
    }
}

static void test_enforce_is_noop_when_satisfied() {
    std::printf("test_enforce_is_noop_when_satisfied\n");
    auto scan = make_seq_scan("t", {});
    scan->scan_format = StorageFormat::Column;
    const PhysicalNode* raw = scan.get();

    PhysicalProperties req;
    req.format = StorageFormat::Column;  // already satisfied, no sort required
    auto enforced = enforce(std::move(scan), req);
    CHECK(enforced.get() == raw);  // same node, unchanged
    CHECK(enforced->op == PhysicalOp::SeqScan);
}

static void test_enforcement_adds_cost() {
    std::printf("test_enforcement_adds_cost\n");
    const CalibrationProfile cal = default_calibration();
    CardinalityModel card;
    card.base_rows["t"] = 1000.0;

    const double base = cost_of(*make_seq_scan("t", {}), cal, card);

    PhysicalProperties req;
    req.sort = {{0, false}};
    auto enforced = enforce(make_seq_scan("t", {}), req);
    CHECK(cost_of(*enforced, cal, card) > base);  // the Sort adds cost
}

// enforce() used to promise "a plan whose output satisfies `required`" and then
// break that promise silently: for an unmet Fresh requirement neither enforcer
// applies, so it returned the STALE input unchanged and the caller emitted a plan
// reading lagging data for a query that must not see it.
//
// Unreachable at the time - nothing set a freshness REQUIREMENT on an input - but
// property-directed search (Increment 2) is exactly what makes consumer
// requirements flow down into enforce(). Pinned now, while it is cheap.
static void test_enforce_reports_an_unenforceable_requirement() {
    std::printf("test_enforce_reports_an_unenforceable_requirement\n");
    PhysicalProperties fresh_required;
    fresh_required.freshness = Freshness::Fresh;

    auto stale = make_seq_scan("t", {});
    stale->scan_freshness = Freshness::Stale;
    CHECK(enforce(std::move(stale), fresh_required) == nullptr);  // no silent pass-through

    // A fresh input needs no enforcer and comes back unchanged.
    auto fresh = make_seq_scan("t", {});
    fresh->scan_freshness = Freshness::Fresh;
    auto ok = enforce(std::move(fresh), fresh_required);
    CHECK(ok != nullptr);
    if (ok) CHECK(satisfies(derive(*ok), fresh_required));

    // The enforceable properties still work over a stale input: staleness is not
    // a requirement here, so it is simply carried through.
    PhysicalProperties order_required;
    order_required.sort = {{0, false}};
    auto stale2 = make_seq_scan("t", {});
    stale2->scan_freshness = Freshness::Stale;
    auto sorted = enforce(std::move(stale2), order_required);
    CHECK(sorted != nullptr);
    if (sorted) CHECK(satisfies(derive(*sorted), order_required));
}

static void test_enforcers_render() {
    std::printf("test_enforcers_render\n");
    auto conv = make_format_convert(make_sort(make_seq_scan("t", {}), {{0, true}}),
                                    StorageFormat::Column);
    const std::string s = physical_to_sexpr(*conv);
    CHECK(contains(s, "(FormatConvert to=column"));
    CHECK(contains(s, "(Sort by=[#0 desc]"));
    CHECK(contains(s, "fmt=row"));  // the underlying scan's storage format
}

// MergeJoin is the first operator with a real REQUIREMENT on its inputs, and the
// first join that gives an order back. Both halves matter: the requirement is
// what enforcement costs, the derivation is what can save a later Sort.
static void test_mergejoin_requires_and_derives_order() {
    std::printf("test_mergejoin_requires_and_derives_order\n");
    const std::vector<HashKey> keys = {{0, 1}};

    // Requires: left sorted on its key column, right sorted on its own.
    const auto reqs = required_input_properties(PhysicalOp::MergeJoin, keys);
    CHECK(reqs.size() == 2);
    CHECK(reqs[0].sort.size() == 1);
    CHECK(reqs[1].sort.size() == 1);
    if (reqs[0].sort.size() == 1) CHECK(reqs[0].sort[0].column == 0);
    if (reqs[1].sort.size() == 1) CHECK(reqs[1].sort[0].column == 1);

    // A HashJoin asks nothing of its inputs - that is its whole advantage.
    const auto none = required_input_properties(PhysicalOp::HashJoin, keys);
    CHECK(none.size() == 2);
    CHECK(none[0].sort.empty());
    CHECK(none[1].sort.empty());

    // Derives: MergeJoin keeps the key order on its output; HashJoin does not.
    auto mj = make_merge_join(make_seq_scan("a", {}), make_seq_scan("b", {}), keys, {});
    CHECK(derive(*mj).sort.size() == 1);
    auto hj = make_hash_join(make_seq_scan("a", {}), make_seq_scan("b", {}), keys, {});
    CHECK(derive(*hj).sort.empty());
}

// The structural difference between freshness and every other property: a Sort
// can establish an order and a FormatConvert a layout, but NOTHING makes stale
// rows fresh. So an unmet freshness requirement is priced as impossible, which is
// what stops such a candidate from ever being chosen.
static void test_freshness_is_unenforceable() {
    std::printf("test_freshness_is_unenforceable\n");
    PhysicalProperties stale;
    stale.format = StorageFormat::Row;
    stale.freshness = Freshness::Stale;

    PhysicalProperties needs_fresh;
    needs_fresh.freshness = Freshness::Fresh;

    CHECK(!satisfies(stale, needs_fresh));

    // An order or a format CAN be enforced, so those cost something finite.
    PhysicalProperties unsorted_fresh;
    unsorted_fresh.format = StorageFormat::Row;
    unsorted_fresh.freshness = Freshness::Fresh;
    PhysicalProperties needs_order;
    needs_order.sort = {{0, false}};
    const double order_cost =
        enforcement_cost(unsorted_fresh, needs_order, 1000.0, default_calibration());
    CHECK(order_cost > 0.0);
    CHECK(order_cost < std::numeric_limits<double>::infinity());

    // Freshness cannot: priced as impossible.
    const double fresh_cost =
        enforcement_cost(stale, needs_fresh, 1000.0, default_calibration());
    CHECK(fresh_cost == std::numeric_limits<double>::infinity());
}

// Staleness propagates upward and never washes out.
static void test_staleness_propagates() {
    std::printf("test_staleness_propagates\n");
    auto scan = make_seq_scan("a", {});
    scan->scan_freshness = Freshness::Stale;
    CHECK(derive(*scan).freshness == Freshness::Stale);

    auto filtered = make_filter(std::move(scan), nullptr);
    CHECK(derive(*filtered).freshness == Freshness::Stale);

    // A join with one lagging side is itself lagging.
    auto fresh_scan = make_seq_scan("b", {});
    auto joined = make_hash_join(std::move(filtered), std::move(fresh_scan), {{0, 0}}, {});
    CHECK(derive(*joined).freshness == Freshness::Stale);
}

int main() {
    test_derive();
    test_freshness_is_unenforceable();
    test_staleness_propagates();
    test_mergejoin_requires_and_derives_order();
    test_satisfies();
    test_enforce_inserts_minimal_enforcers();
    test_enforce_is_noop_when_satisfied();
    test_enforcement_adds_cost();
    test_enforce_reports_an_unenforceable_requirement();
    test_enforcers_render();

    if (g_failures == 0) {
        std::printf("property/enforcer tests: all passed\n");
        return 0;
    }
    std::printf("property/enforcer tests: %d failure(s)\n", g_failures);
    return 1;
}
