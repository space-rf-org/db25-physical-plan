#pragma once
// The physical planner spec - the versioned IDL (design D1/D7) loaded from an
// s-expression file, plus the conformance check that keeps the spec and the code
// in agreement.
//
// "Spec-driven" means the operator catalog and the engine's capability profile
// live in data (spec/physical.spec.sexpr), not in control flow, and a
// conformance check - generated from the spec, walking it against the code -
// fails if the two ever diverge (an operator the planner can emit but the spec
// has not declared, an arity mismatch, an operator the reference engine cannot
// execute). Adding an operator is a spec edit the conformance check then binds.
#include "db25/physical/physical_plan.hpp"

#include <optional>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace db25::physical {

// One entry in the physical operator catalog.
struct OperatorSpec {
    std::string name;
    std::size_t arity = 0;
    std::string kind;
    // What happens to rows arriving on each INPUT, one word per input:
    // streaming / materialized / rescanned / separate. This is the part a single
    // `kind` cannot carry - a hash join materializes one input and streams the
    // other - and conformance asserts `edge_kind()` in the code agrees with it,
    // and that `kind` is the summary these imply.
    std::vector<std::string> edges;
};

// The reference execution-capability profile: which operators the (future)
// engine can execute. Declared in the spec so the planner is functional and
// testable before any engine exists.
struct CapabilityProfile {
    std::string name;
    std::vector<std::string> executes;  // operator names the engine can run

    [[nodiscard]] bool can_execute(const std::string& op) const;
};

// One implementation rule: the physical operator a logical operator lowers to.
// Increment 0 is single-candidate (one rule per logical op); alternatives are
// what the Cascades search explores later.
struct ImplRule {
    std::string logical;   // logical operator name (as the lowering spells it)
    std::string physical;  // physical operator name (must be a declared operator)
};

// A loaded physical spec.
// When exhaustive search stops being affordable (design D5). Past this many
// joins the planner falls back to a bounded enumeration rather than let planning
// time grow with the search space.
//
// The VALUE is a spec input rather than a constant in the code deliberately: what
// is affordable depends on the planning-time budget and the host, neither of which
// belongs in a source file. Deriving it from the budget is still open question 5;
// what this unit fixes is that it is data, and that crossing it is a decision the
// planner reports rather than a silent change of behaviour.
struct SearchBudget {
    std::uint32_t max_join_count = 8;  // a placeholder value, not a derived one
};

struct PhysicalSpec {
    int version = -1;
    std::vector<OperatorSpec> operators;
    std::vector<ImplRule> impl_rules;
    CapabilityProfile profile;
    SearchBudget budget;

    [[nodiscard]] const OperatorSpec* find_operator(const std::string& name) const;
    // Every physical operator a logical operator may lower to, in spec order.
    // More than one makes the choice cost-based: the planner enumerates them all
    // as candidates and the memo keeps the cheapest.
    [[nodiscard]] std::vector<std::string> physicals_for_logical(const std::string& logical) const;

    // The same rules, RESOLVED to operators once at load time. Lowering asks this
    // per logical node, and answering from the rule list meant building a vector
    // of strings and then mapping each name back to its enum by comparing it
    // against every operator name - per node, on the hot path, to compute
    // something that cannot change after the spec is parsed. Empty span for a
    // logical operator with no rule.
    [[nodiscard]] std::span<const PhysicalOp> ops_for_logical(const std::string& logical) const;

    // Built by parse_spec from impl_rules; not part of the spec's text.
    std::unordered_map<std::string, std::vector<PhysicalOp>> resolved_rules;
};

// Parse a spec from s-expression text. Returns the spec on success; on a parse
// error returns nullopt and writes a message to `error`.
[[nodiscard]] std::optional<PhysicalSpec> parse_spec(const std::string& text,
                                                     std::string& error);

// Load and parse a spec from a file path.
[[nodiscard]] std::optional<PhysicalSpec> load_spec(const std::string& path,
                                                    std::string& error);

// Conformance: assert the spec and the code agree. Returns a list of
// human-readable problems; empty means conformant. Checks that every operator
// the planner can emit (kAllPhysicalOps) is declared with the arity the IR
// expects and is executable per the reference profile, that the spec declares no
// operator the IR does not know, that the profile executes only declared
// operators, and that the spec is versioned.
[[nodiscard]] std::vector<std::string> check_conformance(const PhysicalSpec& spec);

}  // namespace db25::physical
