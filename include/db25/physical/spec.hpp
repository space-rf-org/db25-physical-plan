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
    // How this operator can be spread across workers - one of:
    //   full        embarrassingly parallel over its input (a scan, a filter)
    //   partitioned parallel once its input is partitioned by a key (hash join,
    //               hash aggregate): the partitioning is the price of the scaling
    //   ordered     parallel only ACROSS independent ordered groups, serial
    //               within one (a window over partitions, a streaming aggregate
    //               walking group boundaries)
    //   serial      not parallelizable (a limit must count; a fixpoint's
    //               iteration k+1 depends on iteration k)
    //
    // DECLARED, NOT COSTED. Nothing in the cost model reads this - see
    // "parallelism is declared, not costed" below. It exists because the
    // execution simulator needs it and because an operator whose parallel
    // behaviour nobody has stated is an operator nobody has thought about.
    std::string parallelism;
};

// The reference execution-capability profile: which operators the (future)
// engine can execute. Declared in the spec so the planner is functional and
// testable before any engine exists.
struct CapabilityProfile {
    std::string name;
    std::vector<std::string> executes;  // operator names the engine can run
    // The most workers this engine will put on one pipeline. A CAPABILITY, not a
    // hardware fact: the host's core count lives in CalibrationProfile. "This
    // machine has 64 cores" and "this engine uses at most 8 workers" are
    // different statements from different sources, and one number would lose the
    // distinction the two profiles exist to keep. Declared, not costed.
    std::uint32_t max_dop = 1;

    [[nodiscard]] bool can_execute(const std::string& op) const;
};

// PARALLELISM IS DECLARED, NOT COSTED.
//
// The planner records a degree of parallelism and each operator's parallel
// behaviour, and its cost model reads NEITHER. Every plan is costed as though it
// will execute serially. That is a decision, not an oversight, and it was made
// against measurements rather than taste:
//
//   - Where a Sort has to be paid for, the hash-based alternative wins by 12x to
//     14x. For a work-span model to reverse one of those, the loser would have to
//     be more than twelve times as parallel as the winner.
//   - That gap is n*log(n) against n - the complexity class, not a coefficient.
//     It holds from ten rows to a billion, and WIDENS as data grows, so no
//     recalibration of the coefficients can close it.
//   - Window, RecursiveFixpoint, grouping sets and the DML operators have ONE
//     implementation each, so they have no ranking that a span term could invert.
//     UnionAll and HashSetOp are two candidates chosen by applicability, not cost.
//   - Exactly one decision is close enough to be at risk: an aggregate over an
//     already-sorted input, where StreamingAggregate wins by 1.40x. If that ever
//     matters, the fix is a parallelism-aware coefficient on that one operator -
//     not a span term threaded through the whole search.
//
// The decision is cheap to reverse and expensive to pre-empt: adopting a
// work-span cost model later costs exactly what it costs now, while adopting it
// speculatively puts a second cost dimension permanently into the memo and
// branch-and-bound in exchange for nothing measurable.
//
// WHAT WOULD REOPEN IT. The analysis covers the candidates that exist. A
// PARALLEL-AWARE candidate - a partitioned hash join, an intra-node repartition,
// a split-aware scan - is a plan that deliberately does more work to scale
// better, which is precisely the trade a scalar cost cannot express. Adding one
// voids the analysis. tests/test_parallelism.cpp fails when that happens, so the
// question is reopened by the change itself rather than by somebody remembering.

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
