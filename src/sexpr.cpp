#include "db25/physical/sexpr.hpp"

#include "db25/ast/node_types.hpp"  // binary_op_to_string, unary_op_to_string, data_type_to_string

#include <string>
#include <variant>

namespace db25::physical {
namespace {

// ---- expression rendering ------------------------------------------------
// A compact, deterministic renderer for the expression kinds Increment 0 lowers
// (column refs, literals, comparisons). Anything else falls through to a generic
// `(<kind> <child>...)` form so the render is always total and structure-
// sensitive; richer per-kind rendering is added as later increments lower more.

std::string render_literal(const plan::LiteralValue& v) {
    if (const auto* i = std::get_if<std::int64_t>(&v.value)) return std::to_string(*i);
    if (const auto* d = std::get_if<double>(&v.value)) return std::to_string(*d);
    if (const auto* s = std::get_if<std::string>(&v.value)) return "'" + *s + "'";
    if (const auto* b = std::get_if<bool>(&v.value)) return *b ? "true" : "false";
    return "NULL";  // std::monostate
}

std::string render_expr(const Expr* e) {
    if (e == nullptr) return "_";
    switch (e->kind) {
        case plan::ExprKind::ColumnRef:
            return "(col #" + std::to_string(e->input_index) + ")";
        case plan::ExprKind::OuterRef:
            return "(outer d" + std::to_string(e->outer_depth) + " #" +
                   std::to_string(e->input_index) + ")";
        case plan::ExprKind::Literal:
            return "(lit " + render_literal(e->value) + ")";
        case plan::ExprKind::BinaryOp: {
            const Expr* l = e->children.size() > 0 ? e->children[0].get() : nullptr;
            const Expr* r = e->children.size() > 1 ? e->children[1].get() : nullptr;
            return std::string("(") + ast::binary_op_to_string(e->bin_op) + " " +
                   render_expr(l) + " " + render_expr(r) + ")";
        }
        case plan::ExprKind::UnaryOp: {
            const Expr* o = e->children.empty() ? nullptr : e->children[0].get();
            return std::string("(") + ast::unary_op_to_string(e->un_op) + " " +
                   render_expr(o) + ")";
        }
        case plan::ExprKind::ScalarFunction:
        case plan::ExprKind::Aggregate:
        case plan::ExprKind::WindowFunction: {
            std::string s = "(" + e->func_name;
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
        default: {
            std::string s = std::string("(") + plan::expr_kind_to_string(e->kind);
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
    }
}

// ---- schema rendering ----------------------------------------------------
std::string render_schema(const Schema& schema) {
    std::string s = "[";
    for (std::size_t i = 0; i < schema.size(); ++i) {
        const ColumnSchema& c = schema[i];
        if (i != 0) s += " ";
        s += c.name;
        s += ":";
        s += ast::data_type_to_string(c.type);
        if (c.nullable) s += "?";
    }
    return s + "]";
}

// ---- node rendering ------------------------------------------------------
void render_node(const PhysicalNode& n, const std::string& indent, std::string& out) {
    out += indent;
    out += "(";
    out += physical_op_to_string(n.op);

    switch (n.op) {
        case PhysicalOp::SeqScan:
            out += " table=" + n.table_name;
            out += std::string(" fmt=") + storage_format_to_string(n.scan_format);
            // Only the interesting case is rendered: a fresh scan is the norm.
            if (n.scan_freshness == Freshness::Stale) out += " stale";
            break;
        case PhysicalOp::Filter:
            out += " pred=" + render_expr(n.predicate);
            break;
        case PhysicalOp::Project: {
            out += " exprs=[";
            for (std::size_t i = 0; i < n.projections.size(); ++i) {
                if (i != 0) out += " ";
                out += render_expr(n.projections[i]);
            }
            out += "]";
            break;
        }
        case PhysicalOp::HashJoin:
        case PhysicalOp::MergeJoin:
        case PhysicalOp::NestedLoopJoin: {
            // The join kind comes FIRST, and is always printed - including for an
            // inner join. A physical join that does not say which relational join
            // it implements is ambiguous to any reader, and a field that is only
            // printed when it differs from a default trains readers to assume the
            // default is what silence means. Silence is what let INNER and LEFT
            // render byte-identical goldens.
            out += " kind=";
            out += join_kind_to_string(n.join_kind);
            out += " keys=[";
            for (std::size_t i = 0; i < n.hash_keys.size(); ++i) {
                if (i != 0) out += " ";
                out += "(L#" + std::to_string(n.hash_keys[i].left_index) + " R#" +
                       std::to_string(n.hash_keys[i].right_index) + ")";
            }
            out += "]";
            if (!n.residual.empty()) {
                out += " residual=[";
                for (std::size_t i = 0; i < n.residual.size(); ++i) {
                    if (i != 0) out += " ";
                    out += render_expr(n.residual[i]);
                }
                out += "]";
            }
            break;
        }
        case PhysicalOp::Sort: {
            out += " by=[";
            for (std::size_t i = 0; i < n.sort_keys.size(); ++i) {
                if (i != 0) out += " ";
                const SortKey& k = n.sort_keys[i];
                out += "#" + std::to_string(k.column) + (k.descending ? " desc" : " asc");
                // Printed only when the query asked, so an enforcer's key (which
                // has no opinion) is visibly different from an explicit request.
                if (k.nulls_specified) out += k.nulls_first ? " nulls-first" : " nulls-last";
            }
            out += "]";
            break;
        }
        case PhysicalOp::HashAggregate:
        case PhysicalOp::StreamingAggregate: {
            out += " keys=[";
            for (std::size_t i = 0; i < n.group_keys.size(); ++i) {
                if (i != 0) out += " ";
                out += render_expr(n.group_keys[i]);
            }
            out += "] aggs=[";
            for (std::size_t i = 0; i < n.aggregates.size(); ++i) {
                if (i != 0) out += " ";
                out += render_expr(n.aggregates[i]);
            }
            out += "]";
            break;
        }
        case PhysicalOp::Limit: {
            if (n.limits.has_limit) out += " limit=" + std::to_string(n.limits.limit);
            if (n.limits.has_offset) out += " offset=" + std::to_string(n.limits.offset);
            break;
        }
        case PhysicalOp::FormatConvert:
            out += std::string(" to=") + storage_format_to_string(n.target_format);
            break;
    }

    out += " out=" + render_schema(n.output);

    for (const auto& child : n.children) {
        out += "\n";
        render_node(*child, indent + "  ", out);
    }
    out += ")";
}

}  // namespace

std::string physical_to_sexpr(const PhysicalNode& node) {
    std::string out;
    render_node(node, "", out);
    return out;
}

}  // namespace db25::physical
