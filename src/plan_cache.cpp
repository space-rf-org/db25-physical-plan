#include "db25/physical/plan_cache.hpp"

#include <cstring>
#include <variant>

namespace db25::physical {
namespace {

std::uint64_t mix(std::uint64_t h, std::uint64_t v) noexcept {
    h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    return h;
}
std::uint64_t mix_str(std::uint64_t h, const std::string& s) noexcept {
    h = mix(h, s.size());
    for (const char c : s) h = mix(h, static_cast<unsigned char>(c));
    return h;
}
// Doubles by their BITS, not by value. Two coefficients differing in the last
// place can produce different plans, and rounding them into agreement would be a
// way of returning the wrong one.
std::uint64_t mix_double(std::uint64_t h, double d) noexcept {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(d));
    std::memcpy(&bits, &d, sizeof(bits));
    return mix(h, bits);
}

std::uint64_t hash_literal(std::uint64_t h, const plan::LiteralValue& v) {
    // The ARM as well as the value: the integer 1 and the boolean true are
    // different literals, and a plan renders them differently.
    h = mix(h, v.value.index());
    if (const auto* i = std::get_if<std::int64_t>(&v.value)) return mix(h, static_cast<std::uint64_t>(*i));
    if (const auto* d = std::get_if<double>(&v.value)) return mix_double(h, *d);
    if (const auto* s = std::get_if<std::string>(&v.value)) return mix_str(h, *s);
    if (const auto* b = std::get_if<bool>(&v.value)) return mix(h, *b ? 1u : 0u);
    return h;  // monostate: NULL
}

std::uint64_t hash_schema(std::uint64_t h, const plan::Schema& s) {
    h = mix(h, s.size());
    for (const plan::ColumnSchema& c : s) {
        h = mix_str(h, c.name);
        h = mix(h, static_cast<std::uint64_t>(c.type));
        h = mix(h, c.nullable ? 1u : 0u);
        h = mix(h, c.table_id);
        h = mix(h, c.column_id);
        h = mix_str(h, c.alias);
        // `hidden` looks like a bind-time detail (it only governs `SELECT *`
        // expansion) and would be easy to leave out. It cannot be: the schema
        // is carried into the physical plan, and a hit hands back the schema
        // the CACHE owns - so a difference the key ignored is a difference the
        // caller would then read off the wrong plan.
        h = mix(h, c.hidden ? 1u : 0u);
    }
    return h;
}

std::uint64_t hash_node(std::uint64_t h, const plan::LogicalNode* n);
std::uint64_t hash_expr(std::uint64_t h, const plan::Expr* e);

std::uint64_t hash_sort_key(std::uint64_t h, const plan::SortKeyIR& k) {
    h = hash_expr(h, k.expr.get());
    h = mix(h, k.descending ? 1u : 0u);
    h = mix(h, k.nulls_order_explicit ? 1u : 0u);
    h = mix(h, k.nulls_first ? 1u : 0u);
    return h;
}

// EVERY field, positionally. Unlike the memo's structural key - which may declare
// an expression uncomparable and simply decline to share it - this digest has no
// escape hatch: a query it refused to key would be a query that is never cached.
// So the answer to "is this payload compared?" has to be yes for all of it, and
// the field list here is checked against plan::Expr rather than sampled from it.
std::uint64_t hash_expr(std::uint64_t h, const plan::Expr* e) {
    if (e == nullptr) return mix(h, 0xFFFFULL);
    h = mix(h, static_cast<std::uint64_t>(e->kind));
    h = mix(h, static_cast<std::uint64_t>(e->type));
    h = mix(h, e->nullability);
    h = mix(h, e->input_index);
    h = mix(h, e->ref_table_id);
    h = mix(h, e->ref_column_id);
    h = mix(h, e->outer_depth);
    h = hash_literal(h, e->value);
    h = mix(h, static_cast<std::uint64_t>(e->bin_op));
    h = mix(h, static_cast<std::uint64_t>(e->un_op));
    h = mix_str(h, e->func_name);
    h = mix(h, e->distinct ? 1u : 0u);
    h = mix(h, static_cast<std::uint64_t>(e->target_type));
    h = mix(h, e->type_precision);
    h = mix(h, e->type_scale);
    h = mix(h, e->type_length);
    h = mix(h, e->expr_flags);
    h = mix(h, static_cast<std::uint64_t>(e->bool_test));
    h = mix(h, static_cast<std::uint64_t>(e->subquery_kind));
    h = mix(h, e->correlated ? 1u : 0u);
    h = mix(h, e->param_index);
    // The OVER clause, the FILTER predicate, the ordered-aggregate keys: all
    // owned payload the planner reads, all of it in the key.
    h = mix(h, e->window.partition_by.size());
    for (const auto& p : e->window.partition_by) h = hash_expr(h, p.get());
    h = mix(h, e->window.order_by.size());
    for (const auto& k : e->window.order_by) h = hash_sort_key(h, k);
    h = mix(h, e->window.frame.present ? 1u : 0u);
    h = mix_str(h, e->window.frame.spec);
    h = hash_expr(h, e->filter.get());
    h = mix(h, e->agg_order_by.size());
    for (const auto& k : e->agg_order_by) h = hash_sort_key(h, k);
    // A subquery's own plan, digested inline. The memo's key declines to compare
    // one and simply never shares such a group; declining here would mean never
    // caching a query that contains a subquery, which is most interesting ones.
    h = hash_node(h, e->sub_plan.get());
    h = mix(h, e->children.size());
    for (const auto& c : e->children) h = hash_expr(h, c.get());
    return h;
}

std::uint64_t hash_node(std::uint64_t h, const plan::LogicalNode* n) {
    if (n == nullptr) return mix(h, 0xEEEEULL);
    h = mix(h, static_cast<std::uint64_t>(n->op));
    h = hash_schema(h, n->output);
    h = mix_str(h, n->table_name);
    h = mix_str(h, n->alias);
    h = hash_expr(h, n->predicate.get());
    h = mix(h, static_cast<std::uint64_t>(n->join_type));
    h = mix(h, n->exprs.size());
    for (const auto& e : n->exprs) h = hash_expr(h, e.get());
    h = mix(h, n->group_keys.size());
    for (const auto& e : n->group_keys) h = hash_expr(h, e.get());
    h = mix(h, n->aggregates.size());
    for (const auto& e : n->aggregates) h = hash_expr(h, e.get());
    h = mix(h, n->grouping_sets.size());
    for (const auto& set : n->grouping_sets) {
        h = mix(h, set.size());
        for (const std::uint32_t i : set) h = mix(h, i);
    }
    h = mix(h, n->window_functions.size());
    for (const auto& e : n->window_functions) h = hash_expr(h, e.get());
    h = mix(h, n->sort_keys.size());
    for (const auto& k : n->sort_keys) h = hash_sort_key(h, k);
    h = mix(h, n->value_rows.size());
    for (const auto& row : n->value_rows) {
        h = mix(h, row.size());
        for (const auto& e : row) h = hash_expr(h, e.get());
    }
    // LIMIT and OFFSET are part of the SHAPE, not incidental constants: the
    // cardinality model reads them (LimitSpec::rows_out), so two queries
    // differing only in `LIMIT 10` versus `LIMIT 1000000` can get different
    // plans. Keying them out would return one query's plan for the other's.
    h = mix(h, n->has_limit ? 1u : 0u);
    h = mix(h, static_cast<std::uint64_t>(n->limit));
    h = mix(h, n->has_offset ? 1u : 0u);
    h = mix(h, static_cast<std::uint64_t>(n->offset));
    h = mix(h, static_cast<std::uint64_t>(n->set_op));
    h = mix(h, n->target_columns.size());
    for (const std::string& c : n->target_columns) h = mix_str(h, c);
    h = mix(h, n->assignments.size());
    for (const plan::Assignment& a : n->assignments) {
        h = mix(h, a.target_column_id);
        h = hash_expr(h, a.value.get());
    }
    h = mix(h, static_cast<std::uint64_t>(n->conflict_action));
    h = mix(h, n->conflict_columns.size());
    for (const std::string& c : n->conflict_columns) h = mix_str(h, c);
    h = mix(h, n->children.size());
    for (const auto& c : n->children) h = hash_node(h, c.get());
    return h;
}

std::uint64_t hash_properties(std::uint64_t h, const PhysicalProperties& p) {
    h = mix(h, p.sort.size());
    for (const SortKey& k : p.sort) {
        h = mix(h, k.column);
        h = mix(h, k.descending ? 1u : 0u);
        h = mix(h, k.nulls_specified ? 1u : 0u);
        h = mix(h, k.nulls_first ? 1u : 0u);
    }
    h = mix(h, static_cast<std::uint64_t>(p.format));
    h = mix(h, static_cast<std::uint64_t>(p.freshness));
    h = mix(h, static_cast<std::uint64_t>(p.distribution.kind));
    for (const std::uint32_t k : p.distribution.keys) h = mix(h, k);
    return h;
}

std::uint64_t hash_calibration(std::uint64_t h, const CalibrationProfile& c) {
    h = mix_str(h, c.name);
    for (const double d : {c.scan_row, c.column_scan_row, c.filter_row, c.project_row,
                           c.hash_build_row, c.hash_probe_row, c.merge_join_row,
                           c.nested_loop_pair, c.sort_row, c.convert_row, c.limit_row,
                           c.hash_aggregate_row, c.grouping_set_row,
                           c.streaming_aggregate_row, c.window_row, c.hash_distinct_row,
                           c.streaming_distinct_row, c.union_all_row, c.set_op_row,
                           c.values_row, c.recursive_row, c.working_table_row,
                           c.table_write_row, c.exchange_row}) {
        h = mix_double(h, d);
    }
    h = mix(h, c.simd_width);
    h = mix(h, c.cache_line);
    h = mix(h, c.cluster_nodes);
    return h;
}

std::uint64_t hash_cardinality(std::uint64_t h, const CardinalityModel& c) {
    // The base-row map is an unordered_map, so its iteration order is not stable
    // across runs or across two maps with the same contents. Combine each entry
    // with an ORDER-INDEPENDENT operation - xor of per-entry digests - so the key
    // depends on the contents and not on the bucket layout. Summing digests would
    // do as well; what must not happen is folding them in sequence.
    std::uint64_t rows = 0;
    for (const auto& [name, n] : c.base_rows) {
        rows ^= mix_double(mix_str(0x9E37ULL, name), n);
    }
    h = mix(h, rows);
    h = mix(h, c.base_rows.size());
    for (const double d : {c.default_base, c.filter_selectivity, c.join_selectivity,
                           c.group_selectivity, c.distinct_selectivity,
                           c.recursive_iterations, c.working_table_rows}) {
        h = mix_double(h, d);
    }
    return h;
}

}  // namespace

PlanCacheKey plan_cache_key(const plan::LogicalNode& root, const LoweringContext& ctx) {
    PlanCacheKey key;
    key.structure = hash_node(0xC0FFEEULL, &root);

    std::uint64_t h = 0xD15EA5EULL;
    // Every declared input that could change what the planner CHOOSES. A key that
    // omitted one of these would return a plan optimized under a different cost
    // model, a different catalog or a different requirement, and call it a hit.
    h = hash_calibration(h, ctx.calibration != nullptr ? *ctx.calibration
                                                       : default_calibration());
    h = hash_cardinality(h, ctx.cardinality != nullptr ? *ctx.cardinality : CardinalityModel{});
    if (ctx.storage != nullptr) {
        std::uint64_t f = 0;
        for (const auto& [name, avail] : ctx.storage->formats) {
            std::uint64_t e = mix_str(0x517ULL, name);
            for (const FormatAvailability& a : avail) {
                e = mix(e, static_cast<std::uint64_t>(a.format));
                e = mix(e, static_cast<std::uint64_t>(a.freshness));
            }
            f ^= e;  // order-independent, as above
        }
        h = mix(h, f);
        h = mix(h, ctx.storage->formats.size());
    }
    if (ctx.distribution != nullptr) {
        std::uint64_t d = 0;
        for (const auto& [name, dist] : ctx.distribution->tables) {
            std::uint64_t e = mix_str(0xD157ULL, name);
            e = mix(e, static_cast<std::uint64_t>(dist.kind));
            for (const std::uint32_t k : dist.keys) e = mix(e, k);
            d ^= e;
        }
        h = mix(h, d);
        h = mix(h, ctx.distribution->tables.size());
    }
    h = mix(h, static_cast<std::uint64_t>(ctx.required_freshness));
    h = hash_properties(h, ctx.required_output);
    h = mix(h, ctx.prune ? 1u : 0u);
    h = mix(h, ctx.dedup ? 1u : 0u);
    h = mix(h, ctx.reorder_joins ? 1u : 0u);
    h = mix(h, ctx.max_join_count_override);
    // The SPEC is the implementation rules: a different rule set is a different
    // planner. Its version identifies it; a null spec means the built-in mapping,
    // which is a different rule set again.
    h = mix(h, ctx.spec != nullptr ? static_cast<std::uint64_t>(ctx.spec->version) + 1 : 0);
    key.inputs = h;
    return key;
}

const CachedPlan* PlanCache::get(const PlanCacheKey& key) const {
    const auto it = entries_.find(key);
    if (it == entries_.end()) {
        ++misses_;
        return nullptr;
    }
    ++hits_;
    return it->second.get();
}

void PlanCache::put(PlanCacheKey key, plan::LogicalNodePtr logical, LoweringResult result) {
    auto entry = std::make_unique<CachedPlan>();
    entry->logical = std::move(logical);
    entry->result = std::move(result);
    entries_[key] = std::move(entry);
}

void PlanCache::clear() {
    entries_.clear();
    hits_ = 0;
    misses_ = 0;
}

}  // namespace db25::physical
