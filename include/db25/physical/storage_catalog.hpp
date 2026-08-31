#pragma once
// Which storage formats each table is actually available in - the HTAP fact the
// substrate decision turns on (design D4).
//
// A declared INPUT to the planner, like the calibration and cardinality models,
// so the pure-function contract holds: the planner does not reach into a live
// catalog, it is told what exists. A table absent from the map is assumed
// row-only, the conservative default.
#include "db25/physical/physical_plan.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace db25::physical {

// One way a table can be read: a storage format, and whether that copy reflects
// all committed writes. A columnar replica typically lags; the primary row store
// typically does not. Implicitly constructible from a bare StorageFormat, which
// then means "fresh".
struct FormatAvailability {
    StorageFormat format = StorageFormat::Row;
    Freshness freshness = Freshness::Fresh;

    FormatAvailability(StorageFormat f, Freshness fr = Freshness::Fresh)
        : format(f), freshness(fr) {}
};

struct StorageCatalog {
    std::unordered_map<std::string, std::vector<FormatAvailability>> formats;

    // The ways `table` can be read. Never empty: an unknown table falls back to a
    // fresh row-format copy.
    [[nodiscard]] std::vector<FormatAvailability> formats_for(const std::string& table) const {
        const auto it = formats.find(table);
        if (it == formats.end() || it->second.empty()) {
            return {FormatAvailability{StorageFormat::Row}};
        }
        return it->second;
    }
};

}  // namespace db25::physical
