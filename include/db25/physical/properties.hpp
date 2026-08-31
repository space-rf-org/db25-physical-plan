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

#include <vector>

namespace db25::physical {

struct PhysicalProperties {
    std::vector<SortKey> sort;                  // the order the rows are in
    StorageFormat format = StorageFormat::Any;  // the storage format they are in
};

// Does `provided` satisfy `required`? Format: `required` is Any, or the two are
// equal. Sort: `required` is a prefix of `provided` (the rows are sorted at least
// as specifically as required - a superset order still satisfies a prefix
// requirement). An empty required sort is always satisfied.
[[nodiscard]] bool satisfies(const PhysicalProperties& provided,
                             const PhysicalProperties& required);

// The physical properties an operator's output has, given what its inputs
// provide. Representation-independent (as the cost formulas are), so the tree
// form below and the memo form - where inputs are GROUPS - derive identically.
[[nodiscard]] PhysicalProperties derive_op(PhysicalOp op, const std::vector<HashKey>& keys,
                                           StorageFormat scan_format,
                                           const std::vector<PhysicalProperties>& input_props);

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
[[nodiscard]] PhysicalNodePtr enforce(PhysicalNodePtr input,
                                      const PhysicalProperties& required);

}  // namespace db25::physical
