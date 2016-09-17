/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief The k-cuts partition algorithm.
 */

#include "./partition.h"

using namespace std;

namespace nnvm {
namespace pass {
namespace {
// Return the number of possible partition schemes given the shape.
inline size_t NumPossibleSchemes(const TShape& shape) {
  return shape.ndim() + 1;
}
// Change the given scheme to a new one. Return false if all the possible schemes have
// been iterated.
bool NextScheme(const TShape& shape, Scheme* scheme) {
  CHECK_NE(scheme->type, Scheme::kRed);
  if (scheme->type == Scheme::kRep) {
    return false;
  }
  CHECK_EQ(scheme->type, Scheme::kCut);
  CHECK_GE(scheme->dim, 0);
  if (scheme->dim + 1 < shape.ndim()) {
    ++scheme->dim;
  } else {
    scheme->dim = -1;
    scheme->type = Scheme::kRep;
  }
  return true;
}
bool NextSchemeVec(const vector<DPEntry>& entries, vector<Scheme>* schvec) {
  for (size_t i = 0; i < entries.size(); ++i) {
    if (NextScheme(entries[i].region.shape(), &(*schvec)[i])) {
      return true;
    } else {
      (*schvec)[i] = Scheme::Cut(0);
    }
  }
  return false;
}
}  // namespace

NodeEntryGroups::NodeEntryGroups(
    uint32_t num_node_entries, const std::unordered_map<uint32_t, uint32_t>& equals) {
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

BFS::BFS(Graph* src, const NodeEntryGroups* groups): src_graph_(src), entry_groups_(groups) {
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

void BFS::Run(uint32_t start_node_id) {
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

void BFS::Print() const {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  for (size_t i = 0; i < node_levels_.size(); ++i) {
    LOG(INFO) << "Level Node: [";
    for (uint32_t nodeid : node_levels_[i]) {
      const Node* node = graph[nodeid].source;
      LOG(INFO) << "\t#" << nodeid << ": \"" << node->attrs.name << "\""
                << (node->is_variable()? "(variable)" : "");
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

void BFS::AddNode(uint32_t levelid, uint32_t nodeid) {
  if (levelid >= node_levels_.size()) {
    // New level.
    node_levels_.push_back(vector<uint32_t>());
  }
  CHECK_LT(levelid, node_levels_.size());
  const uint32_t level_index = node_levels_[levelid].size();
  node_levels_[levelid].push_back(nodeid);
  node2index_[nodeid] = make_pair(levelid, level_index);
}

void BFS::AddNodeEntry(uint32_t levelid, uint32_t entry_id) {
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

CutAlgorithm::CutAlgorithm(Graph* src, const BFS& bfs): src_graph_(src), bfs_(bfs) {
  const IndexedGraph& idxgraph = src->indexed_graph();
  const OpMap<FAlignedSchemes>& align_map =
    Op::GetAttr<FAlignedSchemes>("FAlignedSchemes");
  const ShapeVector& shapes =
    src_graph_->GetAttr<ShapeVector>("shape");
  // Init DP structures.
  dp_operators_.resize(bfs.node_levels_.size());
  for (size_t i = 0; i < bfs.node_levels_.size(); ++i) {
    for (size_t j = 0; j < bfs.node_levels_[i].size(); ++j) {
      DPOp dpop;
      // Node id.
      const uint32_t node_id = bfs.node_levels_[i][j];
      dpop.node_id = node_id;
      // Input/Output entries.
      const Node* node = idxgraph[dpop.node_id].source;
      //LOG(INFO) << "!!" << node->attrs.name;
      vector<TShape> input_shapes, output_shapes;
      for (const NodeEntry& in_ent : node->inputs) {
        const uint32_t in_ent_id = idxgraph.entry_id(in_ent);
        dpop.input_entry_index.push_back(bfs.GetNodeEntryBFSIndex(in_ent_id));
        input_shapes.push_back(shapes[in_ent_id]);
        //LOG(INFO) << "!!in shape #" << in_ent_id << " " << shapes[in_ent_id];
      }
      for (size_t k = 0; k < node->num_outputs(); ++k) {
        const uint32_t out_ent_id = idxgraph.entry_id(node_id, k);
        dpop.output_entry_index.push_back(bfs.GetNodeEntryBFSIndex(out_ent_id));
        output_shapes.push_back(shapes[out_ent_id]);
        //LOG(INFO) << "!!out shape #" << out_ent_id << " " << shapes[out_ent_id];
      }
      // Aligned requests.
      if (node->is_variable()) {
        // Variable node. Any scheme should be aligned.
      } else {
        CHECK_NOTNULL(node->op());
        FAlignedSchemes align_func = align_map[node->op()];
        dpop.aligned_requests = align_func(node->attrs, input_shapes, output_shapes);
      }
      dp_operators_[i].push_back(dpop);
    }
  }
  dp_entries_.resize(bfs.entry_group_levels_.size());
  for (size_t i = 0; i < bfs.entry_group_levels_.size(); ++i) {
    for (size_t j = 0; j < bfs.entry_group_levels_[i].size(); ++j) {
      DPEntry dpent;
      const uint32_t ent_group_id = bfs.entry_group_levels_[i][j];
      dpent.entry_group_id = ent_group_id;
      const uint32_t ent_id = *((*bfs.entry_groups_)[ent_group_id].begin());
      dpent.region = Region(shapes[ent_id]);
      dp_entries_[i].push_back(dpent);
    }
  }
  // Init DP states.
  Init();
}

const std::vector<Scheme>& CutAlgorithm::GetEntryScheme(uint32_t entry_id) const {
  const BFS::Index& index = bfs_.GetNodeEntryBFSIndex(entry_id);
  return dp_entries_[index.first][index.second].chosen_schemes;
}

void CutAlgorithm::Init() {
  dp_states_.resize(dp_entries_.size());
  for (size_t i = 0; i < dp_states_.size(); ++i) {
    vector<Scheme> schemes(dp_entries_[i].size(), Scheme::Cut(0));
    dp_states_[i].emplace_back(schemes);
    while (NextSchemeVec(dp_entries_[i], &schemes)) {
      // Create new state for each scheme combinations of the entries in this level.
      dp_states_[i].emplace_back(schemes);
    }
    LOG(INFO) << "DP Level #" << i << " size=" << dp_states_[i].size();
  }
}

void CutAlgorithm::Reset() {
  for (auto& lvl : dp_states_) {
    for (auto& state: lvl) {
      state.cost = 0;
      state.chosen_aligned_requests.clear();
    }
  }
}

void CutAlgorithm::OneCut() {
}

}  // namespace pass
}  // namespace nnvm
