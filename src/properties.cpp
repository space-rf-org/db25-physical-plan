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

bool operator==(const SortKey& a, const SortKey& b) noexcept {
    return same_key(a, b);
}

bool operator==(const PhysicalProperties& a, const PhysicalProperties& b) noexcept {
    if (a.format != b.format || a.freshness != b.freshness) return false;
    if (a.sort.size() != b.sort.size()) return false;
    for (std::size_t i = 0; i < a.sort.size(); ++i) {
        if (!same_key(a.sort[i], b.sort[i])) return false;
    }
    return true;
}

bool satisfies(const PhysicalProperties& provided, const PhysicalProperties& required) {
    // Correctness first: stale data cannot answer a query that must see the
    // latest writes, and no enforcer exists that would change that.
    if (required.freshness == Freshness::Fresh && provided.freshness != Freshness::Fresh) {
        return false;
    }
    if (required.format != StorageFormat::Any && required.format != provided.format) {
        return false;
    }
    return sort_prefix(provided.sort, required.sort);
}

PhysicalProperties derive_op(PhysicalOp op, const std::vector<HashKey>& keys,
                             StorageFormat scan_format,
                             const std::vector<PhysicalProperties>& input_props,
                             Freshness scan_freshness) {
    const auto in = [&](std::size_t i) {
        return i < input_props.size() ? input_props[i] : PhysicalProperties{};
    };
    // Staleness propagates upward and never washes out: anything derived from a
    // lagging copy is itself lagging.
    const auto combined_freshness = [&]() {
        Freshness f = Freshness::Fresh;
        for (const PhysicalProperties& p : input_props) {
            if (p.freshness == Freshness::Stale) f = Freshness::Stale;
        }
        return f;
    };
    switch (op) {
        case PhysicalOp::SeqScan:
            // A base scan produces the table's stored format, in no guaranteed
            // order, as fresh as the copy it reads.
            return PhysicalProperties{{}, scan_format, scan_freshness};
        case PhysicalOp::Filter:
        case PhysicalOp::Project:
            // Both drop or reshape columns but preserve order and format. (Project
            // preserves conservatively in Increment 1; a later increment tracks
            // whether the projection actually keeps the sort columns.)
            return in(0);
        case PhysicalOp::HashJoin:
            // Builds a hash table: output order is not guaranteed.
            return PhysicalProperties{{}, StorageFormat::Row, combined_freshness()};
        case PhysicalOp::NestedLoopJoin:
            // Emits pairs in left-outer-loop order. That IS the left input's
            // order, but claiming it would be claiming a property of a specific
            // execution strategy; until the engine guarantees it, derive nothing.
            return PhysicalProperties{{}, StorageFormat::Row, combined_freshness()};
        case PhysicalOp::MergeJoin: {
            // Consumes both inputs in key order and emits in that same order, so
            // the join keys' order survives - the property that can save a later
            // Sort and is exactly why it may win over a HashJoin.
            PhysicalProperties p;
            p.sort.reserve(keys.size());
            for (const HashKey& k : keys) p.sort.push_back(SortKey{k.left_index, false});
            p.format = StorageFormat::Row;
            p.freshness = combined_freshness();
            return p;
        }
        case PhysicalOp::Sort:
            // Establishes its order (filled in by the caller from sort_keys);
            // preserves the child's format.
            return PhysicalProperties{{}, in(0).format, in(0).freshness};
        case PhysicalOp::FormatConvert:
            // Changes the format to its target (filled in by the caller); preserves order.
            return PhysicalProperties{in(0).sort, StorageFormat::Any, in(0).freshness};
    }
    return PhysicalProperties{};
}

bool is_applicable(PhysicalOp op, const std::vector<HashKey>& keys) {
    // Both keyed joins need at least one equi-key: a merge join has nothing to
    // merge on without one, and a hash join has nothing to hash on. Note this
    // cannot be left to costing - with no keys a merge join also requires no
    // sort, so it would incur no enforcement and its cheaper per-row cost would
    // WIN, producing a merge join over a cross product. Applicability has to be
    // decided before cost.
    //
    // A keyless join is not unplannable: it is a NestedLoopJoin, the one join
    // that needs no key. Guarding only MergeJoin (as this did originally) did not
    // close the class - it moved the same unexecutable plan onto HashJoin.
    if (op == PhysicalOp::MergeJoin || op == PhysicalOp::HashJoin) return !keys.empty();
    return true;
}

std::optional<PhysicalProperties> pushdown_requirement(PhysicalOp op,
                                                      const PhysicalProperties& required) {
    // Which operators can satisfy a consumer's requirement by REQUIRING IT OF
    // THEIR INPUT rather than having an enforcer stacked on top of them. Only the
    // order- and format-preserving single-input operators can: derive_op says a
    // Filter and a Project pass their input's properties through unchanged, so an
    // input that arrives sorted leaves sorted.
    //
    // This is one half of the enforce-vs-push-down choice; it is not automatically
    // the better half. Pushing a Sort BELOW a Filter sorts rows the Filter is
    // about to discard, so the search costs both routes and takes the cheaper -
    // it does not assume that earlier is better.
    switch (op) {
        case PhysicalOp::Filter:
        case PhysicalOp::Project:
            return required;
        default:
            // A hash join and a nested loop destroy order; a merge join produces
            // its own; a scan is a leaf. None can pass a requirement downward.
            return std::nullopt;
    }
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
    if (op == PhysicalOp::HashJoin || op == PhysicalOp::MergeJoin ||
        op == PhysicalOp::NestedLoopJoin) {
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

    PhysicalProperties p =
        derive_op(node.op, node.hash_keys, node.scan_format, input_props, node.scan_freshness);
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
    // Verify rather than assume. The enforcers establish order and format; NO
    // enforcer establishes freshness, so a Fresh requirement over stale input
    // reaches here unsatisfied. Returning the input anyway - as this used to -
    // hands back a plan that reads lagging data for a query that must not, which
    // is precisely the correctness hole the Freshness property exists to close.
    // Checking the postcondition rather than special-casing freshness also means
    // any future unenforceable property is caught the day it is added.
    if (!satisfies(derive(*node), required)) {
        return nullptr;
    }
    return node;
}

}  // namespace db25::physical
