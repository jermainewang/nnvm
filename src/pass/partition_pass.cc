/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition_pass.cc
 * \brief Auto-partition dataflow graph
 */
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <algorithm>
#include <queue>
#include <sstream>

using namespace std;

namespace nnvm {
namespace pass {
namespace {

// Visitor function for PrintPass.
void PrintPassVisitor(const NodePtr& n) {
  if (n->op()) {
    LOG(INFO) << "PrintPass: Op \"" << n->op()->name << "\"";
  }
}

class NodeEntryGroups {
  // Some NodeEntrys should have same partition schemes. They should be put in one group.
 public:
  // "equals" is a map from NodeEntryId to NodeEntryId indicating the two nodes should be grouped
  // together. NodeEntry without any groups could be missing in the map, and they will be put in
  // a group that has only one node entry.
  NodeEntryGroups(uint32_t num_node_entries, const unordered_map<uint32_t, uint32_t>& equals) {
    // TODO(minjie): Currently only support disjoint equal pairs.
    for (const auto& eq : equals) {
      const uint32_t ent1 = eq.first;
      const uint32_t ent2 = eq.second;
      CHECK(entry2group_.find(ent1) == entry2group_.end());
      CHECK(entry2group_.find(ent2) == entry2group_.end());
      const uint32_t groupid = groups_.size();
      entry2group_[ent1] = groupid;
      entry2group_[ent2] = groupid;
      groups_.push_back({ent1, ent2});
    }
    // Add remaining entries.
    for (uint32_t entry_id = 0; entry_id < num_node_entries; ++entry_id) {
      if (entry2group_.find(entry_id) == entry2group_.end()) {
        if (equals.find(entry_id) == equals.end()) {
          entry2group_[entry_id] = groups_.size();
          groups_.push_back({entry_id});
        }
      }
    }
  }
  const unordered_set<uint32_t>& operator[](uint32_t group_id) const {
    return groups_[group_id];
  }
  uint32_t group_id(uint32_t entry_id) const {
    return entry2group_.at(entry_id);
  }

 private:
  // Each group is a set of NodeEntryId.
  vector<unordered_set<uint32_t>> groups_;
  // Map from NodeEntryId to NodeEntryGroupId.
  unordered_map<uint32_t, uint32_t> entry2group_;
};

class BFS {
  // The stored nodes and entries are represented by ids in IndexedGraph.
  // Note: This BFS does not consider control dependencies between nodes.
 public:
  // Pair: (levelid, index_within_level).
  typedef pair<uint32_t, uint32_t> Index;

  // Constructor.
  BFS(Graph* src, const NodeEntryGroups* groups): src_graph_(src), entry_groups_(groups) {
    const IndexedGraph& graph = src_graph_->indexed_graph();
    entry_to_nodes_.resize(graph.num_node_entries());
    node_to_entries_.resize(graph.num_nodes());
    for (uint32_t node_id = 0; node_id < graph.num_nodes(); ++node_id) {
      const IndexedGraph::Node& node = graph[node_id];
      // For all input entries, put the node in the adj list.
      for (const IndexedGraph::NodeEntry& in_ent : node.inputs) {
        const uint32_t in_ent_id = graph.entry_id(in_ent);
        entry_to_nodes_[in_ent_id].insert(node_id);
        node_to_entries_[node_id].insert(in_ent_id);
        //LOG(INFO) << "NodeEntry #" << in_ent_id << " -> Node #" << node_id;
      }
      // For all output entries, put the node in the adj list.
      for (uint32_t outidx = 0; outidx < node.source->num_outputs(); ++outidx) {
        const uint32_t out_ent_id = graph.entry_id(node_id, outidx);
        entry_to_nodes_[out_ent_id].insert(node_id);
        node_to_entries_[node_id].insert(out_ent_id);
        //LOG(INFO) << "Node #" << node_id << " -> NodeEntry #" << out_ent_id;
      }
    }
  }

  // Run BFS from the given start node. Treat graph as undirected one.
  void Run(uint32_t start_node_id) {
    queue<pair<uint32_t, uint32_t>> queue;  // (level, id)
    queue.push(make_pair(0, start_node_id));
    unordered_set<uint32_t> visited_nodes, visited_entries;
    while (!queue.empty()) {
      uint32_t level = 0, id = 0;
      tie(level, id) = queue.front();
      queue.pop();

      if (level % 2 == 0 && visited_nodes.count(id) == 0) {
        // This is a Node.
        visited_nodes.insert(id);
        AddNode(level / 2, id);
        // Put all its input/output entries into queue.
        for (const uint32_t ent_id : node_to_entries_[id]) {
          if (visited_entries.count(ent_id) == 0) {
            queue.push(make_pair(level + 1, ent_id));
          }
        }
      } else if (visited_entries.count(id) == 0) {
        // This is a NodeEntry.
        visited_entries.insert(id);
        AddNodeEntry(level / 2, id);
        // Put all its producers/consumers into queue.
        for (const uint32_t node_id : entry_to_nodes_[id]) {
          if (visited_nodes.count(node_id) == 0) {
            queue.push(make_pair(level + 1, node_id));
          }
        }
      }
    }
  }
 
  inline Index GetNodeBFSIndex(uint32_t nodeid) const {
    return node2index_.at(nodeid);
  }

  inline Index GetNodeEntryBFSIndex(uint32_t entry_id) const {
    return entry2index_.at(entry_id);
  }

  void Print() const {
    const IndexedGraph& graph = src_graph_->indexed_graph();
    const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
    for (size_t i = 0; i < node_levels_.size(); ++i) {
      LOG(INFO) << "Level Node: [";
      for (uint32_t nodeid : node_levels_[i]) {
        const Op* op = graph[nodeid].source->op();
        if (op) {
          LOG(INFO) << "\t#" << nodeid << ": " << op->name << ",";
        } else {
          LOG(INFO) << "\t#" << nodeid << ": null,";
        }
      }
      LOG(INFO) << "]";
      if (i < entry_group_levels_.size()) {
        LOG(INFO) << "Level NodeEntry: [";
        for (const uint32_t groupid : entry_group_levels_[i]) {
          ostringstream oss;
          oss << "\t{";
          for (const uint32_t entid : (*entry_groups_)[groupid]) {
            oss << "#" << entid << ": " << shapes[entid] << ", ";
          }
          LOG(INFO) << oss.str() << "},";
        }
        LOG(INFO) << "]";
      }
    }
  }
  
 private:
  void AddNode(uint32_t levelid, uint32_t nodeid) {
    if (levelid >= node_levels_.size()) {
      // New level.
      node_levels_.push_back(vector<uint32_t>());
    }
    CHECK_LT(levelid, node_levels_.size());
    const uint32_t level_index = node_levels_[levelid].size();
    node_levels_[levelid].push_back(nodeid);
    node2index_[nodeid] = make_pair(levelid, level_index);
  }

  void AddNodeEntry(uint32_t levelid, uint32_t entry_id) {
    if (entry2index_.find(entry_id) != entry2index_.end()) {
      // Already been added (from another node in the group).
      return;
    }
    if (levelid >= entry_group_levels_.size()) {
      // New level.
      entry_group_levels_.push_back(vector<uint32_t>());
    }
    CHECK_LT(levelid, entry_group_levels_.size());
    const uint32_t level_index = entry_group_levels_[levelid].size();
    const uint32_t entry_group_id = entry_groups_->group_id(entry_id);
    entry_group_levels_[levelid].push_back(entry_group_id);
    // For all entry in the group, make its index.
    for (const uint32_t ent : (*entry_groups_)[entry_group_id]) {
      CHECK(entry2index_.find(ent) == entry2index_.end()) << "Entry should not be added twice";
      entry2index_[ent] = make_pair(levelid, level_index);
    }
  }

  // Pointer to the source graph (no ownership).
  Graph* src_graph_;
  const NodeEntryGroups* entry_groups_;

  // Entry to all its producer/consumer nodes.
  vector<unordered_set<uint32_t>> entry_to_nodes_;
  // Node to all its input/output nodes.
  vector<unordered_set<uint32_t>> node_to_entries_;

  // BFS levels.
  // All NodeEntries (both inputs/outputs) of Node in level i should be found
  // in entry level (i - 1) and (i).
  vector<vector<uint32_t>> node_levels_;
  vector<vector<uint32_t>> entry_group_levels_;

  unordered_map<uint32_t, Index> node2index_;
  unordered_map<uint32_t, Index> entry2index_;
};

}  // namespace

// Pass function that simply print the names of all operators in the graph.
Graph PrintPass(Graph src) {
  DFSVisit(src.outputs, PrintPassVisitor);
  return src;
}

NNVM_REGISTER_PASS(PrintPass)
.describe("Print names of all operators in the graph.")
.set_body(PrintPass)
.set_change_graph(false);


Graph PartitionPass(Graph src) {
  // TODO
  CHECK_NE(src.attrs.count("forward2backward"), 0) 
    << "Gradient entry mapping information is required.";
  CHECK_NE(src.attrs.count("backward2forward"), 0) 
    << "Gradient entry mapping information is required.";
  const unordered_map<uint32_t, uint32_t>& backward2forward =
    src.GetAttr<unordered_map<uint32_t, uint32_t>>("backward2forward");

  const IndexedGraph& graph = src.indexed_graph();
  const uint32_t start_node_id = graph.node_id(src.outputs[0].node.get());
  // Construct equal set from gradient information. All output gradient entry should have the
  // same partition scheme with its corresponding input entry.
  unordered_map<uint32_t, uint32_t> equal;
  for (const NodeEntry& out_ent : src.outputs) {
    const uint32_t out_ent_id = graph.entry_id(out_ent);
    if (backward2forward.find(out_ent_id) != backward2forward.end()) {
      // This is a gradient output entry. Add equilibrium of it and its forward entry.
      const uint32_t fwd_ent_id = backward2forward.at(out_ent_id);
      equal[out_ent_id] = fwd_ent_id;
    }
  }
  NodeEntryGroups groups(graph.num_node_entries(), equal);

  // Call BFS.
  BFS bfs(&src, &groups);
  bfs.Run(start_node_id);
  bfs.Print();
  return src;
}

NNVM_REGISTER_PASS(PartitionPass)
.describe("Partition tensors in graph and place them to multiple devices.")
.set_body(PartitionPass)
.depend_graph_attr("shape")  // Shape information from InferShapePass.
.depend_graph_attr("forward2backward")  // Gradient information from GradientPass.
.depend_graph_attr("backward2forward")  // Gradient information from GradientPass.
.set_change_graph(true);


}  // namespace pass
}  // namespace nnvm
