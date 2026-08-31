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
#include "db25/physical/cost.hpp"
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/properties.hpp"

#include <cstdint>
#include <deque>
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
    // What THIS candidate's output looks like, derived when it was added. Stored
    // per-expression rather than only for the group's winner, because answering
    // "which candidate is cheapest GIVEN a requirement" needs to know what each
    // one already provides and therefore what each would have to enforce.
    PhysicalProperties provided;
};

// A group's answer to one question: "the best plan for this group that satisfies
// `required`". The requirement is part of the key, not an afterthought - that is
// the whole point. `cost` is the cost of that plan INCLUDING whatever enforcement
// the requirement costs, so two entries in the same group are directly comparable
// only against their own key, never against each other.
struct WinnerEntry {
    PhysicalProperties required;          // the key: what was asked of this group
    std::uint32_t expr_index = 0;         // index into Group::exprs
    double cost = std::numeric_limits<double>::infinity();
    PhysicalProperties provided;          // what the chosen candidate provides
    // The requirement each INPUT was optimized under to reach this cost, in
    // operand order. Without it the winner is not reproducible: extraction would
    // have to guess which goal each child was solved for, and a child optimized
    // for "sorted on k" is a different plan from the same child optimized for
    // nothing. Cascades calls these the child goals; they are part of the winner,
    // not a detail of how it was found.
    std::vector<PhysicalProperties> input_required;
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

    // The group's winners, one per DISTINCT REQUIREMENT it has been optimized
    // for. A single optional winner - one best plan per group, settled bottom-up
    // before any consumer stated a requirement - is the wrong unit of
    // memoization for Cascades: it makes it impossible to ASK a group for a
    // property, only to charge it for lacking one. Linear scan is deliberate;
    // the number of distinct requirements asked of one group is tiny.
    //
    // A DEQUE, not a vector, and that is load-bearing: winner_for() hands out
    // pointers into this container, and property-directed search holds one while
    // recursing into child groups - which optimizes them for new requirements and
    // so inserts more winners. A vector would reallocate and dangle the caller's
    // pointer; a deque never invalidates references to existing elements on
    // push_back. (ASan caught exactly this the first time a test held a winner
    // across a second select_cheapest call.) Returning by value instead would
    // also be safe, but copies two property vectors on every lookup, and lookups
    // are on the search's hot path against a microsecond budget.
    std::deque<WinnerEntry> winners;

    // The unconstrained requirement: "give me your cheapest plan, I need nothing
    // in particular". This is what bottom-up lowering asks for, and it remains a
    // perfectly ordinary member of the map rather than a special case.
    [[nodiscard]] static const PhysicalProperties& unconstrained() noexcept {
        static const PhysicalProperties kAny{};
        return kAny;
    }

    [[nodiscard]] const WinnerEntry* winner_for(const PhysicalProperties& required) const noexcept {
        for (const WinnerEntry& w : winners) {
            if (w.required == required) return &w;
        }
        return nullptr;
    }

    // Convenience accessors for the unconstrained winner - what every caller
    // before property-directed search was implicitly asking for.
    [[nodiscard]] const WinnerEntry* winner() const noexcept {
        return winner_for(unconstrained());
    }
    [[nodiscard]] double best_cost() const noexcept {
        const WinnerEntry* w = winner();
        return w ? w->cost : std::numeric_limits<double>::infinity();
    }
    [[nodiscard]] PhysicalProperties provided() const {
        const WinnerEntry* w = winner();
        return w ? w->provided : PhysicalProperties{};
    }
};

// The memo. Owns its groups; group ids are stable indices into that storage.
class Memo {
public:
    // Create an empty group with the given output schema; returns its id.
    GroupId add_group(Schema output);

    // Append a group-expression to a group; returns its index within the group.
    std::uint32_t add_expr(GroupId group, GroupExpr expr);

    // Record the winner for a requirement, replacing any entry under that key.
    void set_winner(GroupId group, const PhysicalProperties& required,
                    std::uint32_t expr_index, double cost, PhysicalProperties provided,
                    std::vector<PhysicalProperties> input_required = {});

    // Record a group's estimated output cardinality.
    void set_rows(GroupId group, double rows);

    // Choose this group's cheapest candidate FOR `required`, charging each one the
    // enforcement its own output would need to meet that requirement, and record
    // it under that key. Returns false if the group has no expressions, or if no
    // candidate can satisfy the requirement at any price (an unmet Fresh
    // requirement, which no enforcer can establish).
    //
    // Asking the same group for two different requirements is the point: it may
    // legitimately answer with two different plans.
    bool select_cheapest(GroupId group, const PhysicalProperties& required,
                         const CalibrationProfile& cal);

    // The unconstrained choice - "your cheapest plan, no requirement" - which is
    // what bottom-up lowering asks for.
    bool select_cheapest(GroupId group, const CalibrationProfile& cal);

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

    // Extract the winner a group holds for a SPECIFIC requirement. Returns
    // nullptr when the group was never optimized for it.
    [[nodiscard]] PhysicalNodePtr extract_winner_for(GroupId id,
                                                     const PhysicalProperties& required) const;

private:
    std::vector<Group> groups_;
    GroupId root_ = kInvalidGroup;
};

}  // namespace db25::physical
