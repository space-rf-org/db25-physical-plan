#include "db25/physical/statistics.hpp"

#include "db25/physical/sexpr_read.hpp"

#include <charconv>
#include <fstream>
#include <sstream>

namespace db25::physical {
namespace {

bool atom_double(const SNode& root, const char* key, double& out, bool& present) {
    const SNode* kv = root.child(key);
    present = kv != nullptr;
    if (!present) return true;
    const std::string s = value_of(*kv);
    double v = 0.0;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    if (res.ec != std::errc{}) return false;
    out = v;
    return true;
}

bool atom_uint(const SNode& root, const char* key, std::uint32_t& out) {
    const SNode* kv = root.child(key);
    if (kv == nullptr) return false;
    const std::string s = value_of(*kv);
    unsigned long v = 0;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    if (res.ec != std::errc{}) return false;
    out = static_cast<std::uint32_t>(v);
    return true;
}

// Every key a form admits, so an unknown one can be REJECTED. A statistics file
// with a typo'd key would otherwise load as silence - the estimator would fall
// back, the plan would be the un-statted one, and nothing would say why.
bool known_keys_only(const SNode& form, const std::vector<std::string>& allowed,
                     std::string& error) {
    for (const SNode& c : form.list) {
        if (!c.is_list || c.list.empty()) continue;
        const std::string& head = c.list.front().atom;
        bool ok = false;
        for (const std::string& a : allowed) {
            if (a == head) { ok = true; break; }
        }
        if (!ok) {
            error = "unknown key '" + head + "' in (" + form.head() + " ...)";
            return false;
        }
    }
    return true;
}

}  // namespace

std::optional<StatisticsCatalog> load_statistics(const std::string& path, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open statistics file: " + path; return std::nullopt; }
    std::ostringstream ss;
    ss << in.rdbuf();

    SNode root;
    if (!read_sexpr(ss.str(), root, error)) return std::nullopt;
    if (!root.is_list || root.head() != "statistics") {
        error = "top-level form must be (statistics ...)";
        return std::nullopt;
    }

    StatisticsCatalog cat;
    for (const SNode& t : root.list) {
        if (!t.is_list || t.list.empty() || t.head() != "table") continue;
        if (!known_keys_only(t, {"id", "rows", "column"}, error)) return std::nullopt;
        std::uint32_t table_id = 0;
        if (!atom_uint(t, "id", table_id) || table_id == 0) {
            error = "(table ...) needs a non-zero (id N)";
            return std::nullopt;
        }
        TableStats ts;
        bool present = false;
        if (!atom_double(t, "rows", ts.rows, present)) {
            error = "(table id " + std::to_string(table_id) + ") has a malformed (rows ...)";
            return std::nullopt;
        }
        for (const SNode& c : t.list) {
            if (!c.is_list || c.list.empty() || c.head() != "column") continue;
            if (!known_keys_only(c, {"id", "ndv", "nulls", "min", "max", "unique"}, error)) {
                return std::nullopt;
            }
            std::uint32_t column_id = 0;
            if (!atom_uint(c, "id", column_id) || column_id == 0) {
                error = "(column ...) needs a non-zero (id N)";
                return std::nullopt;
            }
            ColumnStats cs;
            bool has_min = false;
            bool has_max = false;
            if (!atom_double(c, "ndv", cs.ndv, present) ||
                !atom_double(c, "nulls", cs.null_fraction, present) ||
                !atom_double(c, "min", cs.min_value, has_min) ||
                !atom_double(c, "max", cs.max_value, has_max)) {
                error = "malformed statistic on (column id " + std::to_string(column_id) + ")";
                return std::nullopt;
            }
            // A range needs BOTH ends. One alone describes a half-open domain the
            // interpolation cannot use, and silently treating the missing end as
            // zero would invent a bound.
            if (has_min != has_max) {
                error = "(column id " + std::to_string(column_id) +
                        ") has only one of (min ...) / (max ...); a range needs both";
                return std::nullopt;
            }
            cs.has_range = has_min && has_max;
            if (cs.has_range && cs.max_value < cs.min_value) {
                error = "(column id " + std::to_string(column_id) + ") has max below min";
                return std::nullopt;
            }
            if (const SNode* u = c.child("unique")) {
                const std::string v = value_of(*u);
                cs.unique = (v == "true" || v == "t" || v == "1");
            }
            ts.columns[column_id] = cs;
        }
        cat.add_table(table_id, std::move(ts));
    }
    return cat;
}

}  // namespace db25::physical
