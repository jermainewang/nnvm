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

class BFS {
  // The stored nodes and entries are represented by ids in IndexedGraph.
  // Note: This BFS does not consider control dependencies between nodes.
 public:
  // Pair: (levelid, index_within_level).
  typedef pair<uint32_t, uint32_t> Index;

  // Constructor.
  BFS(Graph* src): src_graph_(src) {
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
      if (i < entry_levels_.size()) {
        LOG(INFO) << "Level NodeEntry: [";
        for (uint32_t entid : entry_levels_[i]) {
          LOG(INFO) << "\t#" << entid << ": " << shapes[entid] << ",";
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
    const uint32_t level_index = node_levels_[levelid].size();
    node_levels_[levelid].push_back(nodeid);
    node2index_[nodeid] = make_pair(levelid, level_index);
  }

  void AddNodeEntry(uint32_t levelid, uint32_t entry_id) {
    if (levelid >= entry_levels_.size()) {
      // New level.
      entry_levels_.push_back(vector<uint32_t>());
    }
    const uint32_t level_index = entry_levels_[levelid].size();
    entry_levels_[levelid].push_back(entry_id);
    entry2index_[entry_id] = make_pair(levelid, level_index);
  }

  // Pointer to the source graph (no ownership).
  Graph* src_graph_;

  // Entry to all its producer/consumer nodes.
  vector<unordered_set<uint32_t>> entry_to_nodes_;
  // Node to all its input/output nodes.
  vector<unordered_set<uint32_t>> node_to_entries_;

  // BFS levels.
  // All NodeEntries (both inputs/outputs) of Node in level i should be found
  // in entry level (i - 1) and (i).
  vector<vector<uint32_t>> node_levels_;
  vector<vector<uint32_t>> entry_levels_;

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
  const IndexedGraph& graph = src.indexed_graph();
  const uint32_t start_node_id = graph.node_id(src.outputs[0].node.get());
  BFS bfs(&src);
  bfs.Run(start_node_id);
  bfs.Print();
  return src;
}

NNVM_REGISTER_PASS(PartitionPass)
.describe("Partition tensors in graph and place them to multiple devices.")
.set_body(PartitionPass)
.depend_graph_attr("shape")
.set_change_graph(true);


}  // namespace pass
}  // namespace nnvm
