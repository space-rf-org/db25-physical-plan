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

PhysicalProperties derive_op(PhysicalOp op, const std::vector<HashKey>& keys,
                             StorageFormat scan_format,
                             const std::vector<PhysicalProperties>& input_props) {
    const auto in = [&](std::size_t i) {
        return i < input_props.size() ? input_props[i] : PhysicalProperties{};
    };
    switch (op) {
        case PhysicalOp::SeqScan:
            // A base scan produces the table's stored format, in no guaranteed order.
            return PhysicalProperties{{}, scan_format};
        case PhysicalOp::Filter:
        case PhysicalOp::Project:
            // Both drop or reshape columns but preserve order and format. (Project
            // preserves conservatively in Increment 1; a later increment tracks
            // whether the projection actually keeps the sort columns.)
            return in(0);
        case PhysicalOp::HashJoin:
            // Builds a hash table: output order is not guaranteed.
            return PhysicalProperties{{}, StorageFormat::Row};
        case PhysicalOp::MergeJoin: {
            // Consumes both inputs in key order and emits in that same order, so
            // the join keys' order survives - the property that can save a later
            // Sort and is exactly why it may win over a HashJoin.
            PhysicalProperties p;
            p.sort.reserve(keys.size());
            for (const HashKey& k : keys) p.sort.push_back(SortKey{k.left_index, false});
            p.format = StorageFormat::Row;
            return p;
        }
        case PhysicalOp::Sort:
            // Establishes its order (filled in by the caller from sort_keys);
            // preserves the child's format.
            return PhysicalProperties{{}, in(0).format};
        case PhysicalOp::FormatConvert:
            // Changes the format to its target (filled in by the caller); preserves order.
            return PhysicalProperties{in(0).sort, StorageFormat::Any};
    }
    return PhysicalProperties{};
}

std::vector<PhysicalProperties> required_input_properties(PhysicalOp op,
                                                          const std::vector<HashKey>& keys) {
    std::vector<PhysicalProperties> reqs(expected_arity(op));

    // The reference engine's join operators consume ROW-format input (they
    // materialize rows into a hash table or merge row streams), so a columnar
    // subplan feeding a join must be converted - and that conversion is priced,
    // which is what makes the substrate choice a real trade rather than "columnar
    // is always cheaper". A vectorized columnar join is a later operator; when it
    // arrives it simply declares a different requirement here.
    if (op == PhysicalOp::HashJoin || op == PhysicalOp::MergeJoin) {
        for (PhysicalProperties& r : reqs) r.format = StorageFormat::Row;
    }

    if (op == PhysicalOp::MergeJoin) {
        PhysicalProperties left, right;
        left.sort.reserve(keys.size());
        right.sort.reserve(keys.size());
        for (const HashKey& k : keys) {
            left.sort.push_back(SortKey{k.left_index, false});
            right.sort.push_back(SortKey{k.right_index, false});
        }
        left.format = StorageFormat::Row;
        right.format = StorageFormat::Row;
        if (reqs.size() == 2) { reqs[0] = std::move(left); reqs[1] = std::move(right); }
    }
    return reqs;  // every other operator: no requirement on its inputs
}

PhysicalProperties derive(const PhysicalNode& node) {
    std::vector<PhysicalProperties> input_props;
    input_props.reserve(node.children.size());
    for (const auto& c : node.children) input_props.push_back(derive(*c));

    PhysicalProperties p = derive_op(node.op, node.hash_keys, node.scan_format, input_props);
    // The two enforcers carry their established property in their own payload.
    if (node.op == PhysicalOp::Sort) p.sort = node.sort_keys;
    if (node.op == PhysicalOp::FormatConvert) p.format = node.target_format;
    return p;
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
