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

struct StorageCatalog {
    std::unordered_map<std::string, std::vector<StorageFormat>> formats;

    // The formats `table` can be read in. Never empty: an unknown table falls
    // back to row-only.
    [[nodiscard]] std::vector<StorageFormat> formats_for(const std::string& table) const {
        const auto it = formats.find(table);
        if (it == formats.end() || it->second.empty()) return {StorageFormat::Row};
        return it->second;
    }
};

}  // namespace db25::physical
