#include "db25/physical/structural_key.hpp"

#include <cstddef>
#include <variant>

namespace db25::physical {
namespace {

std::uint64_t mix(std::uint64_t h, std::uint64_t v) noexcept {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}
std::uint64_t hash_str(const std::string& s) noexcept {
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a
    for (const char c : s) { h ^= static_cast<unsigned char>(c); h *= 1099511628211ULL; }
    return h;
}

// The kinds whose payload is compared exhaustively below. Anything absent is
// never shared - see the header for why that is the safe default rather than a
// gap to be closed opportunistically.
bool kind_is_comparable(plan::ExprKind k) noexcept {
    switch (k) {
        case plan::ExprKind::ColumnRef:
        case plan::ExprKind::OuterRef:
        case plan::ExprKind::Literal:
        case plan::ExprKind::BinaryOp:
        case plan::ExprKind::UnaryOp:
            return true;
        default:
            // ScalarFunction / Aggregate / WindowFunction / Cast / Subquery /
            // Parameter / BooleanTest and the rest all carry payload this file
            // does not compare (func_name is compared nowhere, window specs,
            // aggregate FILTER and ORDER BY, cast modifiers, an OWNED sub_plan).
            return false;
    }
}

std::uint64_t hash_literal(const plan::LiteralValue& v) noexcept {
    if (const auto* i = std::get_if<std::int64_t>(&v.value)) {
        return mix(1, static_cast<std::uint64_t>(*i));
    }
    if (const auto* d = std::get_if<double>(&v.value)) {
        std::uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(*d));
        __builtin_memcpy(&bits, d, sizeof(bits));
        return mix(2, bits);
    }
    if (const auto* s = std::get_if<std::string>(&v.value)) return mix(3, hash_str(*s));
    if (const auto* b = std::get_if<bool>(&v.value)) return mix(4, *b ? 1u : 0u);
    return 5;  // monostate / NULL
}

bool literal_equal(const plan::LiteralValue& a, const plan::LiteralValue& b) noexcept {
    return a.value == b.value;  // variant equality: same arm AND same value
}

}  // namespace

bool expr_is_comparable(const plan::Expr* e) noexcept {
    if (e == nullptr) return true;
    if (!kind_is_comparable(e->kind)) return false;
    // A comparable kind can still carry payload this file does not compare.
    if (e->filter != nullptr || !e->agg_order_by.empty() || e->sub_plan != nullptr) return false;
    for (const auto& c : e->children) {
        if (!expr_is_comparable(c.get())) return false;
    }
    return true;
}

bool expr_structurally_equal(const plan::Expr* a, const plan::Expr* b) noexcept {
    if (a == b) return a == nullptr || expr_is_comparable(a);
    if (a == nullptr || b == nullptr) return false;
    if (!kind_is_comparable(a->kind) || !kind_is_comparable(b->kind)) return false;
    if (a->kind != b->kind) return false;
    // Type and nullability are part of identity: the same slot read as two
    // different types is not the same expression to a cost model or an executor.
    if (a->type != b->type || a->nullability != b->nullability) return false;
    if (a->filter != nullptr || b->filter != nullptr) return false;
    if (!a->agg_order_by.empty() || !b->agg_order_by.empty()) return false;
    if (a->sub_plan != nullptr || b->sub_plan != nullptr) return false;

    switch (a->kind) {
        case plan::ExprKind::ColumnRef:
            if (a->input_index != b->input_index) return false;
            break;
        case plan::ExprKind::OuterRef:
            if (a->input_index != b->input_index || a->outer_depth != b->outer_depth) return false;
            break;
        case plan::ExprKind::Literal:
            if (!literal_equal(a->value, b->value)) return false;
            break;
        case plan::ExprKind::BinaryOp:
            if (a->bin_op != b->bin_op || a->expr_flags != b->expr_flags) return false;
            break;
        case plan::ExprKind::UnaryOp:
            if (a->un_op != b->un_op || a->expr_flags != b->expr_flags) return false;
            break;
        default:
            return false;
    }
    if (a->children.size() != b->children.size()) return false;
    for (std::size_t i = 0; i < a->children.size(); ++i) {
        if (!expr_structurally_equal(a->children[i].get(), b->children[i].get())) return false;
    }
    return true;
}

std::uint64_t expr_structural_hash(const plan::Expr* e) noexcept {
    if (e == nullptr) return 0x9e3779b9ULL;
    std::uint64_t h = mix(7, static_cast<std::uint64_t>(e->kind));
    h = mix(h, static_cast<std::uint64_t>(e->type));
    h = mix(h, e->nullability);
    switch (e->kind) {
        case plan::ExprKind::ColumnRef:  h = mix(h, e->input_index); break;
        case plan::ExprKind::OuterRef:   h = mix(h, e->input_index);
                                         h = mix(h, e->outer_depth); break;
        case plan::ExprKind::Literal:    h = mix(h, hash_literal(e->value)); break;
        case plan::ExprKind::BinaryOp:   h = mix(h, static_cast<std::uint64_t>(e->bin_op));
                                         h = mix(h, e->expr_flags); break;
        case plan::ExprKind::UnaryOp:    h = mix(h, static_cast<std::uint64_t>(e->un_op));
                                         h = mix(h, e->expr_flags); break;
        default: break;
    }
    for (const auto& c : e->children) h = mix(h, expr_structural_hash(c.get()));
    return h;
}

bool schema_equal(const plan::Schema& a, const plan::Schema& b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        // alias is part of identity: two aliases of the same base table share
        // (table_id, column_id), so without it a self-join's two sides would
        // compare equal and be merged into one group.
        if (a[i].name != b[i].name || a[i].type != b[i].type ||
            a[i].nullable != b[i].nullable || a[i].table_id != b[i].table_id ||
            a[i].column_id != b[i].column_id || a[i].alias != b[i].alias ||
            a[i].hidden != b[i].hidden) {
            return false;
        }
    }
    return true;
}

std::uint64_t schema_hash(const plan::Schema& s) noexcept {
    std::uint64_t h = mix(11, s.size());
    for (const plan::ColumnSchema& c : s) {
        h = mix(h, hash_str(c.name));
        h = mix(h, static_cast<std::uint64_t>(c.type));
        h = mix(h, c.nullable ? 1u : 0u);
        h = mix(h, c.table_id);
        h = mix(h, c.column_id);
        h = mix(h, hash_str(c.alias));
        h = mix(h, c.hidden ? 1u : 0u);
    }
    return h;
}

}  // namespace db25::physical
