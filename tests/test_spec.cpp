// Unit 0.3: the versioned s-expr IDL and the spec<->code conformance check.
//
// Loads the shipped spec (spec/physical.spec.sexpr) and asserts it conforms to
// the code, then exercises the check adversarially: a spec that omits an
// operator, gives a wrong arity, declares a phantom operator, or withholds
// executability must each be REPORTED - that is what makes "spec-driven" a
// binding contract rather than a comment.
#include "db25/physical/spec.hpp"

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

static bool any_contains(const std::vector<std::string>& v, const char* needle) {
    for (const std::string& s : v) {
        if (s.find(needle) != std::string::npos) return true;
    }
    return false;
}

#ifndef DB25_PHYSICAL_SPEC_DIR
#define DB25_PHYSICAL_SPEC_DIR "."
#endif

static void test_shipped_spec_loads_and_conforms() {
    std::printf("test_shipped_spec_loads_and_conforms\n");
    std::string error;
    auto spec = load_spec(std::string(DB25_PHYSICAL_SPEC_DIR) + "/physical.spec.sexpr", error);
    CHECK(spec.has_value());
    if (!spec) {
        std::printf("  load error: %s\n", error.c_str());
        return;
    }
    CHECK(spec->version == 0);
    CHECK(spec->operators.size() == 11);  // + MergeJoin, NestedLoopJoin, Limit, 2 aggregates
    CHECK(spec->profile.name == "reference");

    const auto problems = check_conformance(*spec);
    if (!problems.empty()) {
        for (const std::string& p : problems) std::printf("  unexpected: %s\n", p.c_str());
    }
    CHECK(problems.empty());
}

static void test_missing_operator_is_reported() {
    std::printf("test_missing_operator_is_reported\n");
    // HashJoin omitted from both the catalog and the profile.
    const std::string text =
        "(physical-spec (version 0)"
        "  (operators"
        "    (operator (name SeqScan) (arity 0) (kind access-path))"
        "    (operator (name Filter)  (arity 1) (kind pipeline))"
        "    (operator (name Project) (arity 1) (kind pipeline)))"
        "  (capability-profile (name reference) (executes SeqScan Filter Project)))";
    std::string error;
    auto spec = parse_spec(text, error);
    CHECK(spec.has_value());
    if (!spec) return;
    const auto problems = check_conformance(*spec);
    CHECK(!problems.empty());
    CHECK(any_contains(problems, "HashJoin"));
}

static void test_arity_mismatch_is_reported() {
    std::printf("test_arity_mismatch_is_reported\n");
    // Filter declared with arity 2 (the IR expects 1).
    const std::string text =
        "(physical-spec (version 0)"
        "  (operators"
        "    (operator (name SeqScan)  (arity 0) (kind access-path))"
        "    (operator (name Filter)   (arity 2) (kind pipeline))"
        "    (operator (name Project)  (arity 1) (kind pipeline))"
        "    (operator (name HashJoin) (arity 2) (kind pipeline-breaker)))"
        "  (capability-profile (name reference) (executes SeqScan Filter Project HashJoin)))";
    std::string error;
    auto spec = parse_spec(text, error);
    CHECK(spec.has_value());
    if (!spec) return;
    const auto problems = check_conformance(*spec);
    CHECK(any_contains(problems, "arity mismatch"));
}

static void test_phantom_operator_is_reported() {
    std::printf("test_phantom_operator_is_reported\n");
    // A deliberately fictional name: this case previously used MergeJoin, which
    // Unit 1.2 made real - at which point conformance rightly stopped flagging it
    // and this test failed. Use a name no increment will ever implement.
    const std::string text =
        "(physical-spec (version 0)"
        "  (operators"
        "    (operator (name SeqScan)  (arity 0) (kind access-path))"
        "    (operator (name Filter)   (arity 1) (kind pipeline))"
        "    (operator (name Project)  (arity 1) (kind pipeline))"
        "    (operator (name HashJoin) (arity 2) (kind pipeline-breaker))"
        "    (operator (name PhantomScan)(arity 0) (kind access-path)))"
        "  (capability-profile (name reference)"
        "    (executes SeqScan Filter Project HashJoin PhantomScan)))";
    std::string error;
    auto spec = parse_spec(text, error);
    CHECK(spec.has_value());
    if (!spec) return;
    const auto problems = check_conformance(*spec);
    CHECK(any_contains(problems, "PhantomScan"));
}

static void test_not_executable_is_reported() {
    std::printf("test_not_executable_is_reported\n");
    // HashJoin declared but the reference profile cannot execute it.
    const std::string text =
        "(physical-spec (version 0)"
        "  (operators"
        "    (operator (name SeqScan)  (arity 0) (kind access-path))"
        "    (operator (name Filter)   (arity 1) (kind pipeline))"
        "    (operator (name Project)  (arity 1) (kind pipeline))"
        "    (operator (name HashJoin) (arity 2) (kind pipeline-breaker)))"
        "  (capability-profile (name reference) (executes SeqScan Filter Project)))";
    std::string error;
    auto spec = parse_spec(text, error);
    CHECK(spec.has_value());
    if (!spec) return;
    const auto problems = check_conformance(*spec);
    CHECK(any_contains(problems, "not marked executable"));
}

static void test_malformed_sexpr_is_rejected() {
    std::printf("test_malformed_sexpr_is_rejected\n");
    std::string error;
    auto spec = parse_spec("(physical-spec (version 0) (operators", error);  // unterminated
    CHECK(!spec.has_value());
    CHECK(!error.empty());

    std::string error2;
    auto spec2 = parse_spec("(not-a-spec (version 0))", error2);  // wrong head
    CHECK(!spec2.has_value());
}

int main() {
    test_shipped_spec_loads_and_conforms();
    test_missing_operator_is_reported();
    test_arity_mismatch_is_reported();
    test_phantom_operator_is_reported();
    test_not_executable_is_reported();
    test_malformed_sexpr_is_rejected();

    if (g_failures == 0) {
        std::printf("spec/conformance tests: all passed\n");
        return 0;
    }
    std::printf("spec/conformance tests: %d failure(s)\n", g_failures);
    return 1;
}
