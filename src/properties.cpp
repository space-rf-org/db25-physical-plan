#include "db25/physical/properties.hpp"

#include <cstddef>
#include <utility>

namespace db25::physical {
namespace {

bool same_key(const SortKey& a, const SortKey& b) {
    return a.column == b.column && a.descending == b.descending;
}

// Is `required` a prefix of `provided`? (provided sorted at least as specifically)
bool sort_prefix(const std::vector<SortKey>& provided, const std::vector<SortKey>& required) {
    if (required.size() > provided.size()) return false;
    for (std::size_t i = 0; i < required.size(); ++i) {
        if (!same_key(provided[i], required[i])) return false;
    }
    return true;
}

const PhysicalNode* child0(const PhysicalNode& n) {
    return n.children.empty() ? nullptr : n.children[0].get();
}

}  // namespace

bool satisfies(const PhysicalProperties& provided, const PhysicalProperties& required) {
    if (required.format != StorageFormat::Any && required.format != provided.format) {
        return false;
    }
    return sort_prefix(provided.sort, required.sort);
}

PhysicalProperties derive(const PhysicalNode& node) {
    switch (node.op) {
        case PhysicalOp::SeqScan:
            // A base scan produces the table's stored format, in no guaranteed order.
            return PhysicalProperties{{}, node.scan_format};
        case PhysicalOp::Filter:
            // A filter drops rows but preserves order and format.
            return child0(node) ? derive(*child0(node)) : PhysicalProperties{};
        case PhysicalOp::Project:
            // Increment 0 conservatively preserves the child's order and format
            // (a later increment tracks whether the projection keeps the sort
            // columns / changes the format).
            return child0(node) ? derive(*child0(node)) : PhysicalProperties{};
        case PhysicalOp::HashJoin:
            // A hash join materializes rows in no guaranteed order, row format.
            return PhysicalProperties{{}, StorageFormat::Row};
        case PhysicalOp::Sort: {
            // Establishes its order; preserves the child's format.
            PhysicalProperties p;
            p.sort = node.sort_keys;
            p.format = child0(node) ? derive(*child0(node)).format : StorageFormat::Any;
            return p;
        }
        case PhysicalOp::FormatConvert: {
            // Changes the format to its target; preserves the child's order.
            PhysicalProperties p;
            p.sort = child0(node) ? derive(*child0(node)).sort : std::vector<SortKey>{};
            p.format = node.target_format;
            return p;
        }
    }
    return PhysicalProperties{};
}

PhysicalNodePtr enforce(PhysicalNodePtr input, const PhysicalProperties& required) {
    if (satisfies(derive(*input), required)) {
        return input;  // already satisfied - no enforcer inserted
    }
    PhysicalNodePtr node = std::move(input);

    // Format first, so the sort (if also needed) runs on the required format.
    if (required.format != StorageFormat::Any && derive(*node).format != required.format) {
        node = make_format_convert(std::move(node), required.format);
    }
    // Then the order.
    if (!sort_prefix(derive(*node).sort, required.sort)) {
        node = make_sort(std::move(node), required.sort);
    }
    return node;
}

}  // namespace db25::physical
