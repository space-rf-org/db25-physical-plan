#include "db25/physical/cost.hpp"

#include "db25/physical/sexpr_read.hpp"

#include <charconv>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>

namespace db25::physical {
namespace {

double atom_double(const SNode& root, const char* key, double dflt) {
    const SNode* kv = root.child(key);
    if (kv == nullptr) return dflt;
    const std::string s = value_of(*kv);
    double v = dflt;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    return res.ec == std::errc{} ? v : dflt;
}

std::uint32_t atom_uint(const SNode& root, const char* key, std::uint32_t dflt) {
    const SNode* kv = root.child(key);
    if (kv == nullptr) return dflt;
    const std::string s = value_of(*kv);
    unsigned long v = dflt;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), v);
    return res.ec == std::errc{} ? static_cast<std::uint32_t>(v) : dflt;
}

const PhysicalNode* child_at(const PhysicalNode& n, std::size_t i) {
    return i < n.children.size() ? n.children[i].get() : nullptr;
}

}  // namespace

CalibrationProfile default_calibration() {
    return CalibrationProfile{};  // the in-struct defaults ARE the built-in profile
}

std::optional<CalibrationProfile> load_lab_calibration(const std::string& path,
                                                       std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open calibration file: " + path; return std::nullopt; }
    std::ostringstream ss;
    ss << in.rdbuf();

    SNode root;
    if (!read_sexpr(ss.str(), root, error)) return std::nullopt;
    if (!root.is_list || root.head() != "calibration") {
        error = "top-level form must be (calibration ...)";
        return std::nullopt;
    }

    CalibrationProfile cal;
    if (const SNode* n = root.child("name")) cal.name = value_of(*n);
    cal.scan_row = atom_double(root, "scan-row", cal.scan_row);
    cal.filter_row = atom_double(root, "filter-row", cal.filter_row);
    cal.project_row = atom_double(root, "project-row", cal.project_row);
    cal.hash_build_row = atom_double(root, "hash-build-row", cal.hash_build_row);
    cal.hash_probe_row = atom_double(root, "hash-probe-row", cal.hash_probe_row);
    cal.simd_width = atom_uint(root, "simd-width", cal.simd_width);
    cal.cache_line = atom_uint(root, "cache-line", cal.cache_line);
    return cal;
}

CalibrationProfile calibration_from(CalibrationSource source, const std::string& lab_path,
                                    std::string& error) {
    switch (source) {
        case CalibrationSource::PinnedLab: {
            auto cal = load_lab_calibration(lab_path, error);
            return cal ? *cal : default_calibration();  // error already set on failure
        }
        case CalibrationSource::Live:
        case CalibrationSource::Cached:
            // Documented stub seam: the measuring / caching producers arrive with
            // the execution engine. Until then both resolve to the default so the
            // seam is exercised end-to-end.
            return default_calibration();
    }
    return default_calibration();
}

double CardinalityModel::rows(const PhysicalNode& node) const {
    switch (node.op) {
        case PhysicalOp::SeqScan: {
            const auto it = base_rows.find(node.table_name);
            return it != base_rows.end() ? it->second : default_base;
        }
        case PhysicalOp::Filter: {
            const PhysicalNode* c = child_at(node, 0);
            return c ? rows(*c) * filter_selectivity : 0.0;
        }
        case PhysicalOp::Project: {
            const PhysicalNode* c = child_at(node, 0);
            return c ? rows(*c) : 0.0;
        }
        case PhysicalOp::HashJoin: {
            const PhysicalNode* l = child_at(node, 0);
            const PhysicalNode* r = child_at(node, 1);
            if (!l || !r) return 0.0;
            return rows(*l) * rows(*r) * join_selectivity;
        }
    }
    return 0.0;
}

double cost_of(const PhysicalNode& node, const CalibrationProfile& cal,
               const CardinalityModel& card) {
    double child_cost = 0.0;
    for (const auto& c : node.children) {
        child_cost += cost_of(*c, cal, card);
    }

    double own = 0.0;
    switch (node.op) {
        case PhysicalOp::SeqScan:
            own = card.rows(node) * cal.scan_row;
            break;
        case PhysicalOp::Filter: {
            const PhysicalNode* c = child_at(node, 0);
            own = (c ? card.rows(*c) : 0.0) * cal.filter_row;  // predicate per input row
            break;
        }
        case PhysicalOp::Project: {
            const PhysicalNode* c = child_at(node, 0);
            own = (c ? card.rows(*c) : 0.0) * cal.project_row;
            break;
        }
        case PhysicalOp::HashJoin: {
            const PhysicalNode* l = child_at(node, 0);  // probe side
            const PhysicalNode* r = child_at(node, 1);  // build side
            own = (r ? card.rows(*r) : 0.0) * cal.hash_build_row +
                  (l ? card.rows(*l) : 0.0) * cal.hash_probe_row;
            break;
        }
    }
    return child_cost + own;
}

}  // namespace db25::physical
