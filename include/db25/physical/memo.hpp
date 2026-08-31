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
    const Expr* predicate = nullptr;       // Filter / HashJoin residual
    std::vector<const Expr*> projections;  // Project
    std::vector<HashKey> hash_keys;        // HashJoin

    double cost = std::numeric_limits<double>::infinity();
};

// A memo group: a set of equivalent group-expressions, the output schema they
// all produce, and the winner (lowest-cost member) once costing has run.
struct Group {
    GroupId id = kInvalidGroup;
    Schema output;
    std::vector<GroupExpr> exprs;
    std::optional<std::uint32_t> winner;  // index into `exprs`, set by costing

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

    [[nodiscard]] const Group& group(GroupId id) const { return groups_[id]; }
    [[nodiscard]] Group& group(GroupId id) { return groups_[id]; }
    [[nodiscard]] std::size_t size() const noexcept { return groups_.size(); }

    void set_root(GroupId id) noexcept { root_ = id; }
    [[nodiscard]] GroupId root() const noexcept { return root_; }

    // Materialize the winning physical subtree rooted at `id` (defaults to the
    // memo root). Returns nullptr if `id` is invalid or any reachable group has
    // no winner set.
    [[nodiscard]] PhysicalNodePtr extract_winner(GroupId id = kInvalidGroup) const;

private:
    std::vector<Group> groups_;
    GroupId root_ = kInvalidGroup;
};

}  // namespace db25::physical
