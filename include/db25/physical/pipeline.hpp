#pragma once
// Pipelines: which operators a row flows through without being stopped.
//
// This is the shape an executor needs before it can generate code, and the
// lesson DB25 takes from HyPer: a physical plan is not a tree of operators to be
// interpreted one at a time, it is a set of PIPELINES, each a tight loop over
// rows, separated by the points where rows have to stop.
//
// THE THING THE SPEC CANNOT SAY. physical.spec.sexpr labels each operator
// `pipeline` or `pipeline-breaker`, one label per operator - and a hash join is
// both. It BREAKS its build input, which it materializes into a hash table
// before emitting anything, and it STREAMS its probe input, which flows straight
// through. Whether that is the left or the right input is a decision the search
// makes per plan (see `build_right`), so it is not a property of the operator at
// all. Breaking is a property of an EDGE.
//
// So this file answers the per-edge question, and the spec's per-operator label
// stays what it always was: a coarse summary, useful for reading, insufficient
// for segmenting a plan.
#include "db25/physical/physical_plan.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace db25::physical {

// Why a row stops - or does not - on the edge from an operator to one of its
// inputs. Four answers, not two, because "the pipeline ends here" and "this
// input is buffered" are different facts and a plan that conflated them would
// mislead an executor about where the memory goes.
enum class EdgeKind : std::uint8_t {
    // Rows flow from the input into the operator's output within one loop. The
    // input's pipeline CONTINUES through this operator.
    Streaming,
    // The operator consumes this whole input before it emits anything, holding it
    // in memory: a hash join's build side, a sort, a hash aggregate. This is where
    // an executor allocates, and where a plan's memory footprint is decided.
    Materialized,
    // The operator RE-EXECUTES this input, once per row of another input: a
    // nested loop's right side, a recursive fixpoint's term. Not buffered - that
    // is the point, it is recomputed - but its rows plainly do not flow into one
    // loop with the consumer's.
    Rescanned,
    // The input is its own loop, feeding the same consumer as a sibling: the two
    // sides of a UNION ALL. Nothing is buffered and nothing is recomputed; there
    // are simply two loops. Kept distinct from Materialized so a reader is not
    // told a concatenation allocates.
    Separate,
};

[[nodiscard]] const char* edge_kind_to_string(EdgeKind k) noexcept;

// How `op` consumes input `index`. `build_right` is read only by the hash-family
// operators that have a choosable build side; for everything else it is ignored.
//
// This is the authority. A test asserts it is declared for every input of every
// operator in kAllPhysicalOps, so an operator cannot be added without someone
// deciding where rows stop in it.
[[nodiscard]] EdgeKind edge_kind(PhysicalOp op, std::size_t index, bool build_right) noexcept;

// One pipeline: a maximal run of operators joined by Streaming edges.
//
// `source` is where its rows come from - a scan, or the operator whose
// materialized/rescanned/separate output starts it. `sink` is the operator that
// stops them, or the plan root. `members` lists the operators in the loop, from
// the source upward, which is the order an executor would push a row through.
struct Pipeline {
    const PhysicalNode* source = nullptr;
    const PhysicalNode* sink = nullptr;
    std::vector<const PhysicalNode*> members;
};

// Segment `root` into pipelines, in a legal EXECUTION ORDER: every pipeline comes
// out after the pipelines whose output it consumes, so a hash join's build
// pipeline precedes the probe pipeline that reads its table. That falls out of a
// post-order walk, because a pipeline's dependencies are always rooted deeper in
// the tree than it is - and it makes the numbering a schedule rather than merely
// a stable label a golden can pin.
//
// EVERY node belongs to exactly one pipeline - a test asserts it, because a node
// in none would be an operator no loop ever runs, and a node in two would be one
// an executor generates twice.
[[nodiscard]] std::vector<Pipeline> pipelines(const PhysicalNode& root);

// A readable rendering, one line per pipeline: its index, its source, and the
// operators a row passes through. Deliberately NOT part of the plan s-expr - the
// pipelines are DERIVED from the plan, and putting a derived fact in the golden
// would let the two disagree.
[[nodiscard]] std::string pipelines_to_string(const PhysicalNode& root);

}  // namespace db25::physical
