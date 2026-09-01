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
#include "db25/physical/arity_vec.hpp"
#include "db25/physical/cost.hpp"
#include "db25/physical/physical_plan.hpp"
#include "db25/physical/properties.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>
#include <utility>
#include <limits>
#include <optional>
#include <vector>

namespace db25::physical {

using GroupId = std::uint32_t;
inline constexpr GroupId kInvalidGroup = std::numeric_limits<GroupId>::max();

// One physical operator whose inputs are GROUPS. Payload mirrors PhysicalNode's
// (borrowed expressions; see physical_plan.hpp). `cost` is filled by costing
// (Unit 0.4); infinite until then.
// One physical operator ALTERNATIVE within a group. It carries only what
// distinguishes it from its siblings - the algorithm, the substrate it reads, and
// what that costs. Everything the alternatives share (which relation, which
// predicate, which keys, which input groups) belongs to the GROUP, because they
// are alternative algorithms for ONE logical operator, not different operators.
//
// Holding the shared payload per candidate meant copying a string and three
// vectors for every alternative of every group - duplicating, per algorithm
// choice, data that no algorithm choice can change.
struct GroupExpr {
    PhysicalOp op;
    StorageFormat scan_format = StorageFormat::Row;  // SeqScan: the format read
    Freshness scan_freshness = Freshness::Fresh;     // SeqScan: does that copy lag?
    // HashJoin: which input is MATERIALIZED into the hash table; the other streams
    // past it. Building the SMALLER side is what makes a hash join cheap, and
    // which side is smaller is not knowable until cardinalities are - so it is a
    // costed CANDIDATE rather than a rule, settled by the same comparison as every
    // other physical decision. See is_build_side_choosable for where it is sound.
    bool build_right = true;

    double cost = std::numeric_limits<double>::infinity();
    // What THIS candidate's output looks like, derived when it was added. Stored
    // per-expression rather than only for the group's winner, because answering
    // "which candidate is cheapest GIVEN a requirement" needs to know what each
    // one already provides and therefore what each would have to enforce.
    PhysicalProperties provided;
    // What this candidate requires of each input, in operand order. A function of
    // op and hash_keys alone, both fixed once the candidate exists - so it is
    // computed once at exploration rather than rebuilt (with its vector) on every
    // candidate of every goal the search evaluates.
    ArityVec<PhysicalProperties> input_reqs;
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
    ArityVec<PhysicalProperties> input_required;
};

// A memo group: a set of equivalent group-expressions, the output schema they
// all produce, and the winner (lowest-cost member) once costing has run.
struct Group {
    GroupId id = kInvalidGroup;

    // --- the logical operator every candidate in this group implements ---
    ArityVec<GroupId> inputs;              // child groups, in operand order
    std::string table_name;                // Scan
    const Expr* predicate = nullptr;       // Filter
    std::vector<const Expr*> residual;     // Join: non-key conjuncts to re-check
    // Project: one borrowed expression per output column.
    // Aggregate: the GROUP BY keys followed by the aggregate calls, split at
    // `op_split`. One vector for both is this struct's stated
    // union-by-convention on the logical operator - and here it is also load
    // bearing; see the note below.
    std::vector<const Expr*> op_exprs;
    std::vector<HashKey> hash_keys;        // equi-join keys
    // Join: the relational join being implemented. Part of the GROUP payload, not
    // of a candidate: every group-expression in a group implements the SAME
    // logical operator, and an inner join is not equivalent to a left join.
    ast::JoinType join_kind = ast::JoinType::Inner;
    ast::SetOp set_op = ast::SetOp::Union; // HashSetOp / UnionAll: which one
    // Where `op_exprs` divides, per operator: for an Aggregate the number of
    // leading grouping keys (the rest are aggregate calls); for a ValuesScan the
    // number of COLUMNS per row (the list is flattened row-major). Declared HERE,
    // packed against the one-byte enums above, so it lands in padding that already
    // existed rather than adding eight bytes of its own. Moving it elsewhere in
    // this struct costs 8 bytes and, at the current size, that is enough to change
    // the deque packing - see the note below.
    std::uint32_t op_split = 0;
    std::vector<SortKey> sort_keys;        // Sort: the order it establishes
    LimitSpec limits;                      // Limit: LIMIT / OFFSET

    // WHY THE AGGREGATE PAYLOAD SHARES ONE VECTOR. Groups live in a std::deque,
    // and libstdc++ packs 512/sizeof(T) elements into each deque node - one per
    // node once sizeof(T) reaches 512, two while it is 256. Giving the Aggregate
    // its own `group_keys` and `aggregates` vectors pushed sizeof(Group) from 256
    // to 304, which halved the packing and doubled the memo's node allocations FOR
    // EVERY QUERY: +4 on the five-group budget query, and growing with every later
    // operator that wanted a payload field of its own. The allocation budget test
    // caught it, reporting it as a count rather than as a size, which is why this
    // note exists. Keep sizeof(Group) at or below 256.

    // BORROWED, on the same contract as every expression payload in this planner:
    // the logical plan outlives the physical plan. Copying it here meant one full
    // vector-of-ColumnSchema copy per group, and the group's schema is only ever
    // read to fill the extracted node's own copy - so the copy was pure duplication.
    // The referent must outlive the memo.
    const Schema* output = nullptr;
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
    // A vector, and the search addresses entries BY INDEX rather than by pointer.
    //
    // This container was briefly a deque, to keep pointers valid while the search
    // recursed and inserted more winners. That was correct and expensive: an
    // empty libstdc++ deque allocates its map and a node immediately, so every
    // Group cost two heap allocations at construction whether or not it ever held
    // a winner - and profiling put 37% of the optimizer's instructions in
    // malloc/free. Indices are stable under append without costing anything, so
    // they buy the same safety for free. winner_for() still returns a pointer for
    // callers that use it immediately; anything holding a result ACROSS a call
    // that may insert must hold an index (see winner_index_for).
    std::vector<WinnerEntry> winners;

    // The unconstrained requirement: "give me your cheapest plan, I need nothing
    // in particular". This is what bottom-up lowering asks for, and it remains a
    // perfectly ordinary member of the map rather than a special case.
    [[nodiscard]] static const PhysicalProperties& unconstrained() noexcept {
        static const PhysicalProperties kAny{};
        return kAny;
    }

    // Index of the winner for `required`, if this group has been optimized for
    // it. An INDEX, not a pointer: entries are only ever appended or replaced in
    // place, so an index stays valid across the inserts a recursive search makes.
    [[nodiscard]] std::optional<std::uint32_t> winner_index_for(
        const PhysicalProperties& required) const noexcept {
        for (std::uint32_t i = 0; i < winners.size(); ++i) {
            if (winners[i].required == required) return i;
        }
        return std::nullopt;
    }

    // Convenience for callers that consume the result immediately. INVALIDATED by
    // any later insert into this group's winners - hold an index instead if the
    // result must outlive such a call.
    [[nodiscard]] const WinnerEntry* winner_for(const PhysicalProperties& required) const noexcept {
        const auto i = winner_index_for(required);
        return i ? &winners[*i] : nullptr;
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

// The structural identity of a group, as exploration knows it: the logical
// operator, the schema it produces, the groups it consumes, and its own payload.
// Two groups with equal keys are interchangeable, so only one need exist.
// `comparable` is false when any payload expression is one the structural
// comparison does not exhaustively cover - such a group is never shared.
struct GroupKey {
    int logical_op = -1;
    // BORROWED from the logical node, like every expression payload in this
    // planner: the logical plan already has to outlive the physical plan. Copying
    // the schema here instead cost two full vector-of-ColumnSchema copies per
    // group (each column carries two std::strings) - one for the key and one for
    // the index's stored copy - and measured as more than half of unit 2.4's cost.
    const std::string* table_name = nullptr;
    const Schema* output = nullptr;
    ArityVec<GroupId> inputs;
    const Expr* predicate = nullptr;
    std::vector<const Expr*> residual;
    std::vector<const Expr*> op_exprs;
    std::vector<HashKey> hash_keys;
    // Part of the key. Two joins over the same inputs with the same keys and the
    // same predicate are NOT the same group if one is inner and the other left:
    // deduplicating them would let one query's join silently reuse another's
    // winner. `logical_op` cannot cover this - both are LogicalOp::Join.
    ast::JoinType join_kind = ast::JoinType::Inner;
    // Both are part of the key for the same reason the join kind is: two Sorts
    // differing only in their keys, or two Limits differing only in their bounds,
    // are different operators sharing one LogicalOp.
    std::vector<SortKey> sort_keys;
    LimitSpec limits;
    ast::SetOp set_op = ast::SetOp::Union;
    std::uint32_t op_split = 0;
    bool comparable = false;
    std::uint64_t hash = 0;

    [[nodiscard]] bool equals(const GroupKey& o) const noexcept;
    void finish() noexcept;  // compute `hash` and `comparable`
};

// The memo. Owns its groups; group ids are stable indices into that storage.
class Memo {
public:
    // Create an empty group with the given output schema; returns its id. The
    // schema is BORROWED - it must outlive the memo (see Group::output).
    GroupId add_group(const Schema& output);

    // The id of an existing group structurally identical to `key`, if there is
    // one. Buckets by hash and then VERIFIES field by field: a hash collision
    // that merged two different subtrees would plan one query as another.
    [[nodiscard]] std::optional<GroupId> find_group(const GroupKey& key) const;

    // Record `key` as the identity of group `id`, so later structurally identical
    // subtrees find it. A key marked not-comparable is not indexed at all.
    void index_group(const GroupKey& key, GroupId id);

    // Append a group-expression to a group; returns its index within the group.
    std::uint32_t add_expr(GroupId group, GroupExpr expr);

    // Record the winner for a requirement, replacing any entry under that key.
    void set_winner(GroupId group, const PhysicalProperties& required,
                    std::uint32_t expr_index, double cost, PhysicalProperties provided,
                    ArityVec<PhysicalProperties> input_required = {});

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
    // A deque so growing the memo never COPIES a Group - a Group owns its
    // candidate and winner vectors, and vector-of-Group reallocation was copying
    // all of them. It also makes Group references stable, which the search relies
    // on while recursing.
    std::deque<Group> groups_;
    // Structural index: hash bucket -> the groups carrying that hash, with their
    // keys kept for exact verification.
    std::unordered_map<std::uint64_t, std::vector<std::pair<GroupKey, GroupId>>> index_;
    GroupId root_ = kInvalidGroup;
};

}  // namespace db25::physical
