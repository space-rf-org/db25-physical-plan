#pragma once
// How each table's rows are spread across the nodes of a cluster.
//
// A declared INPUT to the planner, exactly like the StorageCatalog it sits
// beside: the planner is TOLD what exists rather than reaching into a live
// catalog, which is what keeps lowering a pure function of its inputs.
//
// A table absent from the map is `Single` - every row in one place - which is
// what DB25 is today. That default is why adding distribution moves no plan and
// no golden: with nothing declared, every requirement is already satisfied and
// no Exchange is ever inserted. The machinery is real and inert, which is the
// only honest way to add a property before there is a cluster to exercise it.
#include "db25/physical/physical_plan.hpp"

#include <string>
#include <unordered_map>

namespace db25::physical {

struct DistributionCatalog {
    std::unordered_map<std::string, Distribution> tables;

    [[nodiscard]] Distribution for_table(const std::string& table) const {
        const Distribution* d = find(table);
        return d != nullptr ? *d : Distribution{DistributionKind::Single, {}};
    }

    // The entry itself, or null for a table nobody declared. A POINTER because
    // the memo borrows it: a Distribution owns its key list, and owning one per
    // memo group pushed sizeof(Group) past the deque packing boundary - which the
    // static_assert in memo.hpp caught on the line that did it. The catalog is a
    // declared input that outlives the memo, exactly like the logical plan whose
    // expressions the memo already borrows.
    [[nodiscard]] const Distribution* find(const std::string& table) const {
        const auto it = tables.find(table);
        return it == tables.end() ? nullptr : &it->second;
    }
};

}  // namespace db25::physical
