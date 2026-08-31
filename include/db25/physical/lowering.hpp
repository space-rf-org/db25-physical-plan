#pragma once
// logical -> physical lowering (design D2, the pure-function contract).
//
// Consumes a db25-logical-plan LogicalNode tree and produces a physical plan,
// routing through the Cascades memo: each logical node becomes a memo group with
// a single physical group-expression (Increment 0 is single-candidate), the
// winner is set, and the plan is extracted. The logical -> physical operator
// choice comes from the spec's implementation rules (data, not code); the memo,
// cost model, and property framework it builds on are the earlier units.
//
// The planner is a pure function of its declared inputs. The RuntimeProfile input
// is the dynamic-feedback seam (design D2/D10): its type and the way it enters
// the planner exist now; the producer arrives with the execution engine, so it is
// currently an empty, optional input the lowering accepts and ignores.
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/spec.hpp"

#include "db25/plan/logical_plan.hpp"

#include <string>

namespace db25::physical {

// The dynamic-feedback seam. Observed cardinalities / timings / selectivities from
// a prior or live run enter here; the lowering treats it as an input, never a
// hidden channel. Empty until the execution engine produces one.
struct RuntimeProfile {};

// Inputs to the lowering besides the logical plan. `spec` supplies the
// implementation rules (if null, a built-in single-candidate mapping is used);
// `runtime` is the unpopulated feedback seam.
struct LoweringContext {
    const PhysicalSpec* spec = nullptr;
    const RuntimeProfile* runtime = nullptr;
};

struct LoweringResult {
    bool ok = false;
    std::string error;
    PhysicalNodePtr plan;
    std::size_t memo_groups = 0;  // groups the memo held (one per logical node)
};

// Lower a logical plan to a physical plan. The physical plan borrows expression
// payloads from `root`, which must outlive it.
[[nodiscard]] LoweringResult lower(const plan::LogicalNode& root,
                                   const LoweringContext& ctx = {});

}  // namespace db25::physical
