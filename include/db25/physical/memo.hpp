#pragma once
// The Cascades memo - the physical planner's search structure.
//
// A memo stores logically-equivalent subplans as GROUPS, each holding one or
// more GROUP-EXPRESSIONS (a physical operator whose inputs are GROUPS, not
// nodes). Equivalent subplans are therefore stored once and shared across the
// many candidate plans the search enumerates.
//
// Unit 0.2 provides the container and its invariants plus winner extraction. The
// SEARCH itself - exploration, the implementation rules from the IDL, cost-based
// branch-and-bound - arrives in later Increment 0 units; until then a caller
// (or the deterministic lowering of Unit 0.6) populates groups directly and sets
// winners, and extract_winner materializes the chosen physical plan.
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/properties.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace db25::physical {

using GroupId = std::uint32_t;
inline constexpr GroupId kInvalidGroup = std::numeric_limits<GroupId>::max();

// One physical operator whose inputs are GROUPS. Payload mirrors PhysicalNode's
// (borrowed expressions; see physical_plan.hpp). `cost` is filled by costing
// (Unit 0.4); infinite until then.
struct GroupExpr {
    PhysicalOp op;
    std::vector<GroupId> inputs;  // child groups, in operand order

    std::string table_name;                // SeqScan
    StorageFormat scan_format = StorageFormat::Row;  // SeqScan: the format read
    Freshness scan_freshness = Freshness::Fresh;     // SeqScan: does that copy lag?
    const Expr* predicate = nullptr;       // Filter
    std::vector<const Expr*> residual;     // Join: non-key conjuncts to re-check
    std::vector<const Expr*> projections;  // Project
    std::vector<HashKey> hash_keys;        // HashJoin / MergeJoin

    double cost = std::numeric_limits<double>::infinity();
};

// A memo group: a set of equivalent group-expressions, the output schema they
// all produce, and the winner (lowest-cost member) once costing has run.
struct Group {
    GroupId id = kInvalidGroup;
    Schema output;
    // Estimated output rows of this group. Every group-expression in a group is
    // semantically equivalent, so they all produce the same cardinality - it is a
    // property of the GROUP, and costing a parent reads it from its input groups
    // rather than re-deriving it down the tree.
    double rows = 0.0;
    std::vector<GroupExpr> exprs;
    std::optional<std::uint32_t> winner;  // index into `exprs`, set by costing
    // What the WINNING subplan provides. Settled once the group is chosen, so a
    // parent can ask what its inputs already give it (and therefore what it would
    // have to enforce) without walking back down the tree.
    PhysicalProperties provided;

    [[nodiscard]] double best_cost() const noexcept {
        return winner ? exprs[*winner].cost : std::numeric_limits<double>::infinity();
    }
};

// The memo. Owns its groups; group ids are stable indices into that storage.
class Memo {
public:
    // Create an empty group with the given output schema; returns its id.
    GroupId add_group(Schema output);

    // Append a group-expression to a group; returns its index within the group.
    std::uint32_t add_expr(GroupId group, GroupExpr expr);

    // Mark the winning group-expression of a group (an index into its `exprs`).
    void set_winner(GroupId group, std::uint32_t expr_index);

    // Record a group's estimated output cardinality.
    void set_rows(GroupId group, double rows);

    // Record what the group's winning subplan provides.
    void set_provided(GroupId group, PhysicalProperties provided);

    // Choose the lowest-cost group-expression as the winner. Returns false if the
    // group has no expressions. This is the cost-based choice the memo exists to
    // make; with a single candidate it degenerates to that candidate.
    bool select_cheapest(GroupId group);

    [[nodiscard]] const Group& group(GroupId id) const { return groups_[id]; }
    [[nodiscard]] Group& group(GroupId id) { return groups_[id]; }
    [[nodiscard]] std::size_t size() const noexcept { return groups_.size(); }

    void set_root(GroupId id) noexcept { root_ = id; }
    [[nodiscard]] GroupId root() const noexcept { return root_; }

    // Materialize the winning physical subtree rooted at `id` (defaults to the
    // memo root). Where the winning operator REQUIRES a property its input does
    // not provide (a MergeJoin over an unsorted input, say), the matching enforcer
    // is inserted here - and the cost of doing so was already charged to that
    // candidate when it was chosen, so the plan that is built is the plan that was
    // costed. Returns nullptr if `id` is invalid or any reachable group has no
    // winner set.
    [[nodiscard]] PhysicalNodePtr extract_winner(GroupId id = kInvalidGroup) const;

private:
    std::vector<Group> groups_;
    GroupId root_ = kInvalidGroup;
};

}  // namespace db25::physical
