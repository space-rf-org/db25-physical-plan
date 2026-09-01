#pragma once
// Physical properties and the enforcer framework (design D4).
//
// A physical plan node's output has PHYSICAL PROPERTIES - the order its rows are
// in and the storage format they are in. The planner reasons about these
// uniformly: it DERIVES what a subplan provides, and where an operator (or the
// consumer) REQUIRES a property the subplan does not provide, it inserts the
// minimal ENFORCER (a Sort, a FormatConvert) to establish it. Storage format is
// the HTAP substrate (design D4): row vs column becomes exactly the same
// machinery as sort order, not a special case.
#include "db25/physical/physical_plan.hpp"

#include <optional>
#include <span>
#include <vector>

namespace db25::physical {

struct PhysicalProperties {
    std::vector<SortKey> sort;                  // the order the rows are in
    StorageFormat format = StorageFormat::Any;  // the storage format they are in
    // Whether the rows reflect all committed writes. As a REQUIREMENT this is a
    // correctness constraint - no enforcer can establish it, so a subplan that
    // cannot provide it is discarded rather than fixed up.
    Freshness freshness = Freshness::Any;
};

// Value equality. Requirements are the KEY a group's winners are memoized under
// (memo.hpp), so two requirements that ask for the same thing must compare equal
// or the same work would be redone under a second key.
[[nodiscard]] bool operator==(const SortKey& a, const SortKey& b) noexcept;
[[nodiscard]] bool operator==(const PhysicalProperties& a, const PhysicalProperties& b) noexcept;

// Does `provided` satisfy `required`? Freshness: a Fresh requirement admits only
// Fresh data - and unlike the others it can never be enforced into existence.
// Format: `required` is Any, or the two are equal. Sort: `required` is a prefix
// of `provided` (the rows are sorted at least
// as specifically as required - a superset order still satisfies a prefix
// requirement). An empty required sort is always satisfied.
[[nodiscard]] bool satisfies(const PhysicalProperties& provided,
                             const PhysicalProperties& required);

// The physical properties an operator's output has, given what its inputs
// provide. Representation-independent (as the cost formulas are), so the tree
// form below and the memo form - where inputs are GROUPS - derive identically.
[[nodiscard]] PhysicalProperties derive_op(PhysicalOp op, const std::vector<HashKey>& keys,
                                           StorageFormat scan_format,
                                           std::span<const PhysicalProperties> input_props,
                                           Freshness scan_freshness = Freshness::Fresh);

// Is `op` a legal implementation for a join with these equi-keys? A MergeJoin
// merges two sorted streams ON THE JOIN KEYS, so with no keys there is nothing to
// merge on and it is not a valid way to compute the (cross) product - regardless
// of what it would cost. Every other operator is unconditionally applicable.
[[nodiscard]] bool is_applicable(PhysicalOp op, const std::vector<HashKey>& keys,
                                 ast::JoinType join_kind = ast::JoinType::Inner);

// The part of a consumer's requirement an operator can discharge by REQUIRING IT
// OF ITS INPUT instead of having an enforcer placed above it - nullopt when it
// cannot (a hash join destroys order, a scan is a leaf). One half of the
// enforce-vs-push-down choice; the search costs both routes rather than assuming
// pushing down is better, because a Sort below a Filter sorts rows the Filter is
// about to discard.
[[nodiscard]] std::optional<PhysicalProperties> pushdown_requirement(
    PhysicalOp op, const PhysicalProperties& required);

// What each input of an operator must PROVIDE for that operator to be usable.
// A MergeJoin needs both inputs sorted on its join keys - the requirement that
// makes it cheaper than a HashJoin when the order is already there and dearer
// when it must be enforced. Every other operator is indifferent. The returned
// vector is one entry per input, in operand order.
[[nodiscard]] std::vector<PhysicalProperties> required_input_properties(
    PhysicalOp op, const std::vector<HashKey>& keys);

// The physical properties of the output of `node`, derived bottom-up.
[[nodiscard]] PhysicalProperties derive(const PhysicalNode& node);

// Return a plan whose output satisfies `required`, wrapping `input` in the
// minimal enforcers needed (a FormatConvert to the required format, then a Sort
// to the required order). If `input` already satisfies `required`, it is returned
// unchanged - no enforcer inserted.
//
// Returns NULLPTR when the requirement cannot be enforced at all - today that
// means an unmet Fresh requirement, since no operator turns stale rows into fresh
// ones. The postcondition is checked, not assumed: whatever this returns
// non-null satisfies `required`. A caller must treat nullptr as "this subplan
// cannot serve this requirement" and fail, rather than emitting the input
// unchanged - which is what it used to do silently.
[[nodiscard]] PhysicalNodePtr enforce(PhysicalNodePtr input,
                                      const PhysicalProperties& required);

}  // namespace db25::physical
