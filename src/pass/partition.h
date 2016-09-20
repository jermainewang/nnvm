/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief The k-cuts partition algorithm.
 */
#ifndef NNVM_PASS_PARTITION_H_
#define NNVM_PASS_PARTITION_H_

#include <nnvm/base.h>
#include <nnvm/graph.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <algorithm>
#include <queue>
#include <sstream>

namespace nnvm {
namespace pass {

class CutAlgorithm;

typedef int64_t cost_t;

class NodeEntryGroups {
  // Some NodeEntrys should have same partition schemes. They should be put in one group.
 public:
  // "equals" is a map from NodeEntryId to NodeEntryId indicating the two nodes should be grouped
  // together. NodeEntry without any groups could be missing in the map, and they will be put in
  // a group that has only one node entry.
  NodeEntryGroups(uint32_t num_node_entries,
                  const std::unordered_map<uint32_t, uint32_t>& equals);

  const std::unordered_set<uint32_t>& operator[](uint32_t group_id) const {
    return groups_[group_id];
  }
  uint32_t group_id(uint32_t entry_id) const {
    return entry2group_.at(entry_id);
  }

 private:
  // Each group is a set of NodeEntryId.
  std::vector<std::unordered_set<uint32_t>> groups_;
  // Map from NodeEntryId to NodeEntryGroupId.
  std::unordered_map<uint32_t, uint32_t> entry2group_;
};

class BFS {
  // The stored nodes and entries are represented by ids in IndexedGraph.
  // Note: This BFS does not consider control dependencies between nodes.
  friend class CutAlgorithm;
 public:
  // Pair: (levelid, index_within_level).
  typedef std::pair<uint32_t, uint32_t> Index;

  // Constructor.
  BFS(Graph* src, const NodeEntryGroups* groups);

  // Run BFS from the given start node. Treat graph as undirected one.
  void Run(uint32_t start_node_id);
 
  inline Index GetNodeBFSIndex(uint32_t nodeid) const {
    return node2index_.at(nodeid);
  }

  inline Index GetNodeEntryBFSIndex(uint32_t entry_id) const {
    return entry2index_.at(entry_id);
  }

  // Print graph in a readable way.
  void Print() const;
  
 private:
  void AddNode(uint32_t levelid, uint32_t nodeid);

  void AddNodeEntry(uint32_t levelid, uint32_t entry_id);

  // Pointer to the source graph (no ownership).
  Graph* src_graph_;
  const NodeEntryGroups* entry_groups_;

  // Entry to all its producer/consumer nodes.
  std::vector<std::unordered_set<uint32_t>> entry_to_nodes_;
  // Node to all its input/output nodes.
  std::vector<std::unordered_set<uint32_t>> node_to_entries_;

  // BFS levels.
  // All NodeEntries (both inputs/outputs) of Node in level i should be found
  // in entry level (i - 1) and (i).
  std::vector<std::vector<uint32_t>> node_levels_;
  std::vector<std::vector<uint32_t>> entry_group_levels_;

  std::unordered_map<uint32_t, Index> node2index_;
  std::unordered_map<uint32_t, Index> entry2index_;
};

class Region {
 public:
  // Constructors.
  Region() {}
  Region(const TShape& shp):
    entry_shape_(shp), region_offset_(shp.ndim()), region_shape_(shp) {
    for (size_t i = 0; i < shp.ndim(); ++i) {
      region_offset_[i] = 0;
    }
  }
  Region(const TShape& ent_shp, const TShape& reg_off,
         const TShape& reg_shp):
    entry_shape_(ent_shp), region_offset_(reg_off),
    region_shape_(reg_shp) {}

  inline const TShape& shape() const { return region_shape_; }

  inline const TShape& offset() const { return region_offset_; }

  inline const TShape& entry_shape() const { return entry_shape_; }

  // Partition this region into two sub-regions.
  std::pair<Region, Region> Split2(const Scheme& sch) const;

  // Area of the region.
  inline cost_t Area() const { return region_shape_.Size(); }

  // Compute the intersection area.
  static cost_t IntersectArea(const Region& r1, const Region& r2);

  // Compute the conversion cost from r1 to r2. The scheme only
  // partitions regions into two parts.
  static cost_t ConvertCost2(const Region& r1, const Scheme& sch1,
                             const Region& r2, const Scheme& sch2);

 private:
  // Shape of the entry this region belongs to.
  TShape entry_shape_;
  // Region offset, and shape.
  TShape region_offset_, region_shape_;
};

struct DPEntry {
  uint32_t entry_group_id;
  Region region;
  // Recorded optimal schemes for each one-cut algorithm.
  std::vector<Scheme> chosen_schemes;
};

struct DPOp {
  uint32_t node_id;
  std::vector<BFS::Index> input_entry_index;
  std::vector<BFS::Index> output_entry_index;
  std::vector<SchemeRequest> aligned_requests;
  std::vector<Region> input_ghost_regions;
  std::vector<Region> output_ghost_regions;
  // Recorded optimal request for each one-cut algorithm.
  std::vector<size_t> chosen_aligned_requests;
};

struct DPState {
  // Entry schemes represented by this state.
  std::vector<Scheme> schemes;
  // Minimal cost to reach this state.
  cost_t cost = 0;
  // The state index in the previous BFS level that is used to reach the optimal
  // cost of this state.
  int prev_state_index = -1;
  // Aligned request chosen for each operator in this state to get the minimal cost.
  std::vector<size_t> chosen_aligned_requests;

  explicit DPState(const std::vector<Scheme>& schemes): schemes(schemes) {}
};

class CutAlgorithm {
 public:
  // Constructor.
  CutAlgorithm(Graph* src, const BFS& bfs);

  // One cut algorithm. Return the minimal cost.
  cost_t OneCut();

  // K-cut algorithm. Return the minimal cost.
  cost_t KCuts(uint32_t K);

  // Get schemes of a node entry.
  const std::vector<Scheme>& GetEntryScheme(uint32_t entry_id) const;

  // Print debug information.
  void Print() const;

 private:
  // Init all DP states. Create auxiliary structures for the main algorithm.
  void Init();
  
  // Clear all states computed by DP, but leave those auxiliary structures.
  void Reset();

  inline bool IsVariable(const DPOp& op) const {
    return src_graph_->indexed_graph()[op.node_id].source->is_variable();
  }

  std::pair<cost_t, size_t> ConversionCost(
      const DPOp& op,
      const DPState* prev_state,
      const DPState* next_state,
      size_t lvl) const;

  // Extract optimal plan from the states computed by one-cut algorithm. The plan includes
  // schemes of each node entry and which aligned request is used for each node.
  // Return the minimal cost.
  cost_t ExtractOptimalPlan();

  Graph* src_graph_;
  const BFS& bfs_;

  std::vector<std::vector<DPOp>> dp_operators_;
  std::vector<std::vector<DPEntry>> dp_entries_;

  std::vector<std::vector<DPState>> dp_states_;
};

}  // namespace pass
}  // namespace nnvm

#endif
