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
#include "db25/physical/cost.hpp"
#include "db25/physical/join_order.hpp"
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/properties.hpp"
#include "db25/physical/spec.hpp"
#include "db25/physical/distribution_catalog.hpp"
#include "db25/physical/storage_catalog.hpp"

#include "db25/plan/logical_plan.hpp"

#include <cstdint>
#include <string>

namespace db25::physical {

// The dynamic-feedback seam. Observed cardinalities / timings / selectivities from
// a prior or live run enter here; the lowering treats it as an input, never a
// hidden channel. Empty until the execution engine produces one.
struct RuntimeProfile {};

// Inputs to the lowering besides the logical plan. `spec` supplies the
// implementation rules (if null, a built-in single-candidate mapping is used);
// `calibration` and `cardinality` drive the cost-based choice between candidates
// (defaults are used when null); `runtime` is the unpopulated feedback seam.
struct LoweringContext {
    const PhysicalSpec* spec = nullptr;
    const CalibrationProfile* calibration = nullptr;
    const CardinalityModel* cardinality = nullptr;
    // Which storage formats each table is available in. A scan gets one candidate
    // per available format, so the substrate is chosen by cost like any other
    // physical decision. Defaults to row-only when not supplied.
    const StorageCatalog* storage = nullptr;
    // How each table's rows are spread across a cluster. Defaults to
    // single-node, which is DB25 today - and with that default every
    // distribution requirement is already satisfied, so no Exchange is ever
    // inserted and no plan changes. The seam is real and inert until a caller
    // declares a partitioned table.
    const DistributionCatalog* distribution = nullptr;
    // What the QUERY demands of the data it reads. Fresh means it must reflect
    // all committed writes, so a lagging copy is not merely dearer - it is
    // ineligible, and a scan that can only be served from one is discarded.
    // Freshness originates at scans and only ever propagates upward, so forbidding
    // stale scans is exactly equivalent to requiring a fresh result.
    Freshness required_freshness = Freshness::Any;
    // What the CONSUMER demands of the plan's output - an ORDER BY's sort order,
    // say, or a format the caller needs. This is the requirement the top-down
    // search starts from: groups are optimized FOR it rather than merely costed
    // and then patched up with enforcers afterwards. Empty means "no requirement",
    // which is what every caller before property-directed search implied.
    PhysicalProperties required_output;
    // Branch-and-bound pruning. On by default; it is a pure SEARCH optimization
    // and must never change the plan chosen, which is what makes turning it off a
    // usable diagnostic rather than a behaviour switch. A test lowers every case
    // both ways and asserts the rendered plans are identical - pruning that
    // changes a plan is a bug, not a speedup.
    bool prune = true;
    // Structural memo dedup (unit 2.4). Complete, tested and falsifiable - and
    // OFF by default, because it does not currently pay for itself. It collapses
    // repeated logical subtrees, and no query in the corpus repeats one: every
    // join is over two distinct tables. Meanwhile building a structural key per
    // group costs ~3.9us of T6 on the reference query, roughly doubling the
    // stage, to find nothing.
    //
    // Its premise is rule application - join reordering above all - which
    // generates equivalent group-expressions across groups and is what makes
    // re-derivation exponential. Turn this on with that, or for a workload with
    // genuinely repeated subtrees; the mechanism is ready either way.
    bool dedup = false;
    // Associative join reordering (increment 3.8b). A region of INNER / CROSS
    // joins is re-associated by an interval DP, so which tree the query text
    // happened to write is not the tree the planner has to build. On by default;
    // off is a diagnostic, and the pair of runs is what a test compares to show
    // that reordering only ever produced a plan the cost model prefers.
    bool reorder_joins = true;
    // Override the spec's search budget (design D5). Zero means "use the spec's".
    // A test can force the guard on a small query; production reads the spec.
    std::uint32_t max_join_count_override = 0;
    const RuntimeProfile* runtime = nullptr;
};

struct LoweringResult {
    bool ok = false;
    std::string error;
    // Storage the plan BORROWS. Declared before `plan` so it outlives it through
    // destruction as well as through use: reordering re-addresses a conjunct's
    // columns when it moves a join, and the re-addressed copy has to live
    // somewhere the caller owns. Empty for a plan that reordered nothing, which
    // is every plan the earlier increments produced.
    LoweringArena arena;
    PhysicalNodePtr plan;
    std::size_t memo_groups = 0;  // groups the memo held (one per logical node)
    // Physical candidates enumerated across all groups. More than one per group
    // means the plan was CHOSEN by cost rather than merely derived.
    std::size_t candidates_considered = 0;
    // Distinct (group, requirement) pairs the top-down search optimized. Greater
    // than memo_groups means some group really was optimized for more than one
    // requirement - the thing Increment 2 exists to make possible.
    std::size_t optimization_goals = 0;
    // Candidates abandoned before being fully costed because their cost already
    // exceeded the best plan found for the same goal. Evidence that pruning is
    // doing something; zero on a query with nothing to prune.
    std::size_t candidates_pruned = 0;
    // Logical subtrees that turned out to be structurally identical to one
    // already explored, and so shared its memo group instead of building a
    // second. Zero on a query with no repeated subtree - which is most of them.
    std::size_t groups_shared = 0;
    // Joins in the logical plan, and whether that count crossed the spec's search
    // budget. When it did, the plan was chosen by a BOUNDED enumeration rather
    // than the full property-directed search: still costed by the same model, but
    // not guaranteed to be the cheapest plan the full search would have found.
    // Reported rather than silent, because a caller that cares about plan quality
    // needs to know which of the two it got.
    std::size_t join_count = 0;
    bool budget_guard_engaged = false;
    // Join regions the interval DP enumerated - maximal INNER / CROSS subtrees of
    // three or more leaves whose every join has a connecting predicate. ENUMERATED,
    // not reordered: the tree the query wrote is one of the trees the DP considers,
    // so a region can be enumerated and still come out exactly as written, which is
    // the correct outcome when the written order is already the cheapest. Zero on a
    // query with no such region, which is most of them - two tables have only one
    // association, and a region containing a cross product is left alone.
    std::size_t join_regions_enumerated = 0;
};

// Lower a logical plan to a physical plan. The physical plan borrows expression
// payloads from `root`, which must outlive it.
[[nodiscard]] LoweringResult lower(const plan::LogicalNode& root,
                                   const LoweringContext& ctx = {});

}  // namespace db25::physical
