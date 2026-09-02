#include "db25/physical/join_order.hpp"

#include <cstddef>
#include <cstdint>

namespace db25::physical {
namespace {

// Sentinel for "no leaf owns this column", which can only happen if a conjunct
// references outside the region - a reason to leave the region alone, not to fail.
constexpr std::uint32_t kNoLeaf = 0xFFFFFFFFu;

// Only INNER and CROSS joins belong to a region. An outer join is not associative
// - moving it changes which rows are null-extended - and a LATERAL join's right
// input reads its left, so it cannot be moved at all. Either ENDS the region and
// becomes one of its leaves, which is what keeps reordering sound around them
// rather than forbidding queries that contain them.
bool reorderable(const plan::LogicalNode& n) {
    return n.op == plan::LogicalOp::Join && (n.join_type == ast::JoinType::Inner ||
                                             n.join_type == ast::JoinType::Cross);
}

// An expression the planner is willing to re-address. Deliberately a SHORT list.
// A Subquery owns a whole inner plan whose indices live in their own space, and an
// OuterRef points outside this region entirely - translating either by adding an
// offset would silently point somewhere else. A conjunct containing one is not a
// reason to refuse the query; it is a reason not to reorder around it.
bool translatable(const plan::Expr* e) {
    if (e == nullptr) return false;
    switch (e->kind) {
        case plan::ExprKind::ColumnRef:
        case plan::ExprKind::Literal:
        case plan::ExprKind::Parameter:
            return true;
        case plan::ExprKind::BinaryOp:
        case plan::ExprKind::UnaryOp:
        case plan::ExprKind::Cast:
        case plan::ExprKind::IsNull:
        case plan::ExprKind::BooleanTest:
        case plan::ExprKind::Between:
        case plan::ExprKind::Like:
        case plan::ExprKind::InList:
        case plan::ExprKind::Case:
        case plan::ExprKind::Row:
            break;
        default:
            return false;  // Subquery, OuterRef, aggregates, window functions
    }
    for (const auto& c : e->children) {
        if (!translatable(c.get())) return false;
    }
    return true;
}

// Which leaf owns a region column. Linear over leaves, which is at most 64 and in
// practice a handful.
std::uint32_t leaf_of(const JoinRegion& r, std::uint32_t region_column) {
    for (std::size_t i = 0; i < r.leaves.size(); ++i) {
        if (region_column >= r.leaf_offset[i] &&
            region_column < r.leaf_offset[i] + r.leaf_width[i]) {
            return static_cast<std::uint32_t>(i);
        }
    }
    return kNoLeaf;
}

bool collect_conjuncts(const plan::Expr* e, std::uint32_t base, JoinRegion& r) {
    if (e == nullptr) return true;  // a CROSS join has no predicate
    if (e->kind == plan::ExprKind::BinaryOp && e->bin_op == ast::BinaryOp::And) {
        // Split on AND for the same reason the join-key extractor does: a
        // conjunct that mentions two leaves can be placed on the join that first
        // brings them together, and one that mentions all of them cannot. Keeping
        // the AND whole would force every conjunct to the top.
        if (e->children.size() != 2) return false;
        return collect_conjuncts(e->children[0].get(), base, r) &&
               collect_conjuncts(e->children[1].get(), base, r);
    }
    if (!translatable(e)) return false;
    RegionConjunct c;
    c.expr = e;
    c.base = base;
    // The leaves it touches, so the search can ask which joins may carry it.
    struct Walk {
        static bool go(const plan::Expr* x, std::uint32_t b, const JoinRegion& reg,
                       std::uint64_t& mask) {
            if (x == nullptr) return true;
            if (x->kind == plan::ExprKind::ColumnRef) {
                const std::uint32_t leaf = leaf_of(reg, b + x->input_index);
                if (leaf == kNoLeaf) return false;
                mask |= (std::uint64_t{1} << leaf);
            }
            for (const auto& ch : x->children) {
                if (!go(ch.get(), b, reg, mask)) return false;
            }
            return true;
        }
    };
    if (!Walk::go(e, base, r, c.leaf_mask)) return false;
    r.conjuncts.push_back(c);
    return true;
}

// The region's leaf count, without building anything. A two-table join - the
// common case, and the one with nothing to re-associate - reaches
// collect_join_region on every query, and must not pay for four vectors' worth of
// allocation just to be told there is nothing to do. Returns false past the
// bitmask's 64 leaves.
bool count_leaves(const plan::LogicalNode& n, std::size_t& count) {
    if (!reorderable(n)) {
        ++count;
        return count <= 64;
    }
    if (n.child_count() != 2) return false;
    return count_leaves(*n.child(0), count) && count_leaves(*n.child(1), count);
}

bool walk_region(const plan::LogicalNode& n, JoinRegion& r, std::uint32_t& column) {
    if (!reorderable(n)) {
        r.leaves.push_back(&n);
        r.leaf_offset.push_back(column);
        r.leaf_width.push_back(static_cast<std::uint32_t>(n.output.size()));
        column += static_cast<std::uint32_t>(n.output.size());
        return true;
    }
    if (n.child_count() != 2) return false;
    // The join's own input concatenation starts where its FIRST leaf does, and an
    // in-order traversal gives every subtree a contiguous range - which is the
    // whole reason a single offset suffices instead of a rewrite.
    const std::uint32_t base = column;
    if (!walk_region(*n.child(0), r, column)) return false;
    if (!walk_region(*n.child(1), r, column)) return false;
    // Collected AFTER the children, so every leaf the predicate names already has
    // an offset to resolve against.
    return collect_conjuncts(n.predicate.get(), base, r);
}

// Clone `e`, adding `delta` to every ColumnRef index. `delta` is signed because a
// conjunct can move DOWN as well as up: the top join of ((A B) C) numbers from
// A's first column, and the same conjunct on A (B C)'s lower join numbers from
// B's, which is a larger offset and so a negative shift.
plan::ExprPtr clone_shifted(const plan::Expr& e, std::int64_t delta) {
    auto out = std::make_unique<plan::Expr>(e.kind);
    out->type = e.type;
    out->nullability = e.nullability;
    out->source = e.source;
    out->ref_table_id = e.ref_table_id;
    out->ref_column_id = e.ref_column_id;
    out->value = e.value;
    out->bin_op = e.bin_op;
    out->un_op = e.un_op;
    out->func_name = e.func_name;
    out->distinct = e.distinct;
    out->target_type = e.target_type;
    out->type_precision = e.type_precision;
    out->type_scale = e.type_scale;
    out->type_length = e.type_length;
    out->expr_flags = e.expr_flags;
    out->bool_test = e.bool_test;
    out->param_index = e.param_index;
    // The one field that CHANGES, and the only reason this function exists.
    out->input_index =
        e.kind == plan::ExprKind::ColumnRef
            ? static_cast<std::uint32_t>(static_cast<std::int64_t>(e.input_index) + delta)
            : e.input_index;
    out->children.reserve(e.children.size());
    for (const auto& c : e.children) {
        out->children.push_back(clone_shifted(*c, delta));
    }
    return out;
}

}  // namespace

bool collect_join_region(const plan::LogicalNode& n, JoinRegion& out) {
    if (!reorderable(n)) return false;
    // Fewer than three leaves has nothing to re-associate - the two-way case is
    // what build-side selection already covers. More than 64 exceeds the leaf-set
    // bitmask, and is far past any search budget worth running. Asked FIRST,
    // because answering it is free and building the region is not.
    std::size_t leaves = 0;
    if (!count_leaves(n, leaves) || leaves < 3) return false;
    out = JoinRegion{};
    out.leaves.reserve(leaves);
    out.leaf_offset.reserve(leaves);
    out.leaf_width.reserve(leaves);
    std::uint32_t column = 0;
    if (!walk_region(n, out, column)) return false;
    // The region's columns must BE its leaves' columns concatenated, in leaf
    // order. That is the property every index in this file rests on, and an inner
    // or cross join's output is exactly that concatenation - so this holds. It is
    // CHECKED rather than assumed because everything downstream is index
    // arithmetic that would silently re-address columns if it ever stopped
    // holding, and a wrong column is a wrong answer, not a slow plan.
    if (column != n.output.size()) return false;
    out.output = &n.output;
    return true;
}

std::uint64_t range_mask(std::size_t first, std::size_t last) {
    if (last <= first) return 0;
    // Built by subtraction rather than a shift-by-64, which is undefined.
    const std::uint64_t below_last =
        last >= 64 ? ~std::uint64_t{0} : (std::uint64_t{1} << last) - 1;
    const std::uint64_t below_first = (std::uint64_t{1} << first) - 1;
    return below_last & ~below_first;
}

bool placed_here(const RegionConjunct& c, std::size_t first, std::size_t split,
                 std::size_t last) {
    const std::uint64_t here = range_mask(first, last);
    if ((c.leaf_mask & ~here) != 0) return false;  // reaches outside this subtree
    // A child that is itself a join can carry the conjunct lower, and the lowest
    // join that can is where it goes - that is what makes the placement unique, so
    // a conjunct is applied exactly once however the tree is associated. A child
    // that is a single leaf is not a join and has nowhere to put a predicate, so a
    // conjunct confined to one leaf stops at the join directly above it.
    if (split - first >= 2 && (c.leaf_mask & ~range_mask(first, split)) == 0) return false;
    if (last - split >= 2 && (c.leaf_mask & ~range_mask(split, last)) == 0) return false;
    return true;
}

const plan::Expr* translate_conjunct(const RegionConjunct& c, const JoinRegion& region,
                                     std::size_t first, LoweringArena& arena) {
    // A join over [first, last) concatenates its inputs' columns, and because both
    // inputs are contiguous leaf ranges meeting at the split, that concatenation IS
    // the region's columns from `leaf_offset[first]` onward, in the region's order.
    // So the whole translation is one subtraction.
    const std::int64_t delta = static_cast<std::int64_t>(c.base) -
                               static_cast<std::int64_t>(region.leaf_offset[first]);
    if (delta == 0) return c.expr;  // already numbered the way this join sees it
    arena.exprs.push_back(clone_shifted(*c.expr, delta));
    return arena.exprs.back().get();
}

}  // namespace db25::physical
