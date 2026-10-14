#pragma once
// Per-column statistics, and the catalog the cardinality model consults for them.
//
// Until now every selectivity in the model was a FLAT CONSTANT: a filter kept
// 10% of its rows whether it was `WHERE id = 7` or `WHERE salary > 0`, and a
// GROUP BY produced one group per ten input rows whether it grouped by a primary
// key or by a boolean. Those constants are honest - they are named, they live in
// one place, and their comments say they are guesses - but they are guesses, and
// on the benchmark corpus they are the entire remaining estimate error. A filter
// is a property of the DATA, and no rule estimates it; only a statistic does.
//
// WHAT THIS IS NOT. It is not a histogram. `ndv` and a numeric range answer the
// predicates that dominate real queries (equality on a key, a range on an
// ordered column, a null test) and answer them exactly where the data is
// uniform. Where it is skewed they are wrong in a bounded, explainable way, and
// a most-common-value list is the next thing to add - deliberately not this
// increment, because a histogram that nobody has measured against is a second
// set of invented numbers wearing a better hat.
//
// PROVENANCE. These numbers come from the caller, who has the catalog. They are
// on exactly the same footing as `CardinalityModel::base_rows`, which has always
// been caller-seeded: the planner does not measure, it is told. When an engine
// exists, ANALYZE replaces the telling.
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>

namespace db25::physical {

// What is known about one column. Every field is OPTIONAL in the sense that
// zero/false means "not known", and an estimator that finds a field unknown must
// fall back to the model's default rather than treat zero as a measurement. A
// column with `ndv = 0` is not a column with no distinct values.
struct ColumnStats {
    // Distinct non-null values. 0 means UNKNOWN, which is why this is a double
    // rather than a count: the fallback path must be able to tell "I was not
    // told" from "I was told one".
    double ndv = 0.0;
    // Fraction of rows where the column is NULL, in [0, 1]. Answers IS NULL
    // exactly, and corrects every other predicate, since NULL satisfies none of
    // them.
    double null_fraction = 0.0;
    // The numeric domain, for range predicates. `has_range` is what separates
    // "the column runs 0 to 0" from "nobody said". Only meaningful for ordered
    // types; a VARCHAR column leaves it false and its ranges fall back.
    bool has_range = false;
    double min_value = 0.0;
    double max_value = 0.0;
    // The column is unique within its table (a primary key, or a UNIQUE
    // constraint). Carried separately from `ndv == rows` because it is a
    // GUARANTEE rather than a measurement: a join onto a unique key cannot
    // multiply, and that is knowable without trusting a sampled count.
    bool unique = false;
};

// What is known about one table.
struct TableStats {
    double rows = 0.0;  // 0 = unknown; the model falls back to `base_rows`
    std::unordered_map<std::uint32_t, ColumnStats> columns;  // by column_id
};

// Statistics for the tables a query touches, keyed by the catalog's stable
// identifiers. Keyed by ID and not by NAME deliberately: two aliases of one
// table in a self-join share (table_id, column_id) and must share statistics,
// while two tables with the same name in different schemas must not. The
// logical plan already carries both ids on every ColumnRef, so an estimator
// needs no schema walk to ask this question.
class StatisticsCatalog {
public:
    void add_table(std::uint32_t table_id, TableStats stats) {
        tables_[table_id] = std::move(stats);
    }
    void add_column(std::uint32_t table_id, std::uint32_t column_id, ColumnStats stats) {
        tables_[table_id].columns[column_id] = stats;
    }
    [[nodiscard]] const TableStats* table(std::uint32_t table_id) const {
        const auto it = tables_.find(table_id);
        return it == tables_.end() ? nullptr : &it->second;
    }
    [[nodiscard]] const ColumnStats* column(std::uint32_t table_id,
                                            std::uint32_t column_id) const {
        const TableStats* t = table(table_id);
        if (t == nullptr) return nullptr;
        const auto it = t->columns.find(column_id);
        return it == t->columns.end() ? nullptr : &it->second;
    }
    [[nodiscard]] bool empty() const noexcept { return tables_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return tables_.size(); }

private:
    std::unordered_map<std::uint32_t, TableStats> tables_;
};

// Load a statistics catalog from an s-expr file, for the reason the lab
// calibration profile is loadable: a checked-in file makes a test deterministic
// and makes the numbers reviewable. Returns nullopt and sets `error` on a
// malformed file; an unknown key is an ERROR, not a warning, on the same rule
// the machine spec follows.
[[nodiscard]] std::optional<StatisticsCatalog> load_statistics(const std::string& path,
                                                               std::string& error);

}  // namespace db25::physical
