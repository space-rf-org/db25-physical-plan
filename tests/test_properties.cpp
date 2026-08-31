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

int main() {
    test_derive();
    test_mergejoin_requires_and_derives_order();
    test_satisfies();
    test_enforce_inserts_minimal_enforcers();
    test_enforce_is_noop_when_satisfied();
    test_enforcement_adds_cost();
    test_enforcers_render();

    if (g_failures == 0) {
        std::printf("property/enforcer tests: all passed\n");
        return 0;
    }
    std::printf("property/enforcer tests: %d failure(s)\n", g_failures);
    return 1;
}
