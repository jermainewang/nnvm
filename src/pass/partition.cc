/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief The k-cuts partition algorithm.
 */

#include "./partition.h"

using namespace std;

namespace {
ostream& operator << (ostream& os, const nnvm::Scheme& sch) {
  using nnvm::Scheme;
  switch (sch.type) {
  case Scheme::kCut: return os << "C" << sch.dim;
  case Scheme::kRep: return os << "Rp";
  case Scheme::kRed: return os << "Rd";
  default:
    LOG(FATAL) << "Unknown scheme type: " << sch.type;
  }
  return os;
}
ostream& operator << (ostream& os, const nnvm::pass::Region& region) {
  return os << "[" << region.offset()
            << " + " << region.shape()
            << " in: " << region.entry_shape() << "]";
}
nnvm::TShape operator + (const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] += shp2[i];
  }
  return ret;
}
nnvm::TShape operator - (const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] -= shp2[i];
  }
  return ret;
}
nnvm::TShape max(const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] = std::max(ret[i], shp2[i]);
  }
  return ret;
}
nnvm::TShape min(const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    ret[i] = std::min(ret[i], shp2[i]);
  }
  return ret;
}
}  // namespace

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
// Return the pointer list to the contents given the index.
// prev_level and next_level could be nullptr when the function is called for the operators
// in the first and last BFS levels.
template<typename T>
vector<const T*> ExtractFromIndex(
    const vector<BFS::Index>& index,
    const vector<T>* prev_level,
    const vector<T>* next_level,
    size_t current_level) {
  vector<const T*> ret;
  for (const BFS::Index& idx : index) {
    if (idx.first == current_level) {
      // Content is in next level.
      CHECK_NOTNULL(next_level);
      ret.push_back(&(next_level->at(idx.second)));
    } else if (idx.first == current_level - 1) {
      // Content is in prev level.
      CHECK_NOTNULL(prev_level);
      ret.push_back(&(prev_level->at(idx.second)));
    } else {
      LOG(FATAL) << "Invalid entry index (" << idx.first << ", " << idx.second
                 << ") for operator in level #" << current_level;
    }
  }
  return ret;
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

pair<Region, Region> Region::Split2(const Scheme& sch) const {
  switch (sch.type) {
  case Scheme::kCut:
    {
    TShape shp = region_shape_;
    CHECK(shp[sch.dim] % 2 == 0) << "Dimension " << sch.dim << " of size "
      << shp[sch.dim] << " cannot be splitted into two.";
    shp[sch.dim] /= 2;
    TShape offset = region_offset_;
    offset[sch.dim] += shp[sch.dim];
    Region r1(entry_shape_, region_offset_, shp);
    Region r2(entry_shape_, offset, shp);
    return make_pair(r1, r2);
    }
  case Scheme::kRep:
    {
    return make_pair(*this, *this);
    }
  default:
    LOG(FATAL) << "Scheme: " << sch << " is not supported for split.";
  }
  return pair<Region, Region>();
}
  
cost_t Region::IntersectArea(const Region& r1, const Region& r2) {
  const TShape& r1_end = r1.offset() + r1.shape();
  const TShape& r2_end = r2.offset() + r2.shape();
  const TShape& st = max(r1.offset(), r2.offset());
  const TShape& ed = min(r1_end, r2_end);
  cost_t cost = 1;
  for (size_t i = 0; i < st.ndim(); ++i) {
    if (ed[i] <= st[i]) {
      // No intersection.
      return 0;
    } else {
      cost *= ed[i] - st[i];
    }
  }
  return cost;
}

// Note that it is possible that r1 and r2 have different areas. Consider following
// matmult example:
//  - First cut: C x R = red -> R
//  - Second cut: R x r = R
cost_t Region::ConvertCost2(const Region& r1, const Scheme& sch1,
                            const Region& r2, const Scheme& sch2) {
  CHECK_NE(sch2.type, Scheme::kRed)
    << "Reduction scheme is intermediate and could not be used as conversion target";
  cost_t cost = 0;
  if (sch1.type == Scheme::kRed) {
    // Reduction scheme requires special calculation.
    // Note that if source scheme is reduction, the area of source region and target
    // region may be different.
    if (sch2.type == Scheme::kCut) {
      cost = r1.Area();
    } else if (sch2.type == Scheme::kRep) {
      cost = 2 * r1.Area();
    } else {
      LOG(FATAL) << "Invalid target scheme: " << sch2;
    }
  } else {
    if (sch1.type == Scheme::kRep) {
      // If the source scheme is replication, then all data could be fetched locally.
    } else {
      const pair<Region, Region>& r1split = r1.Split2(sch1);
      const pair<Region, Region>& r2split = r2.Split2(sch2);
      cost += Region::IntersectArea(r1split.first, r2split.second);
      cost += Region::IntersectArea(r1split.second, r2split.first);
    }
    if (sch2.type == Scheme::kRep) {
      // If target scheme is replication, extra cost is required to replicate the area
      // that does not overlap with the source one (i.e, r2 - r1).
      cost += r2.Area() - Region::IntersectArea(r1, r2);
    }
  }
  CHECK_GE(cost, 0);
  return cost;
}

CutAlgorithm::CutAlgorithm(Graph* src, const BFS& bfs): src_graph_(src), bfs_(bfs) {
  const IndexedGraph& idxgraph = src->indexed_graph();
  const OpMap<FAlignedSchemes>& align_map =
    Op::GetAttr<FAlignedSchemes>("FAlignedSchemes");
  const ShapeVector& shapes =
    src_graph_->GetAttr<ShapeVector>("shape");
  // Init DP structures.
  dp_entries_.resize(bfs.entry_group_levels_.size());
  for (size_t i = 0; i < bfs.entry_group_levels_.size(); ++i) {
    for (size_t j = 0; j < bfs.entry_group_levels_[i].size(); ++j) {
      DPEntry dpent;
      const uint32_t ent_group_id = bfs.entry_group_levels_[i][j];
      dpent.entry_group_id = ent_group_id;
      const uint32_t ent_id = *((*bfs.entry_groups_)[ent_group_id].begin());
      // The initial ghost region is the same as region.
      dpent.region = Region(shapes[ent_id]);
      dp_entries_[i].push_back(dpent);
    }
  }
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
        const TShape& in_shape = shapes[in_ent_id];
        dpop.input_entry_index.push_back(bfs.GetNodeEntryBFSIndex(in_ent_id));
        // Initial ghost area shape is the same as node entry shape.
        dpop.input_ghost_regions.emplace_back(in_shape);
        input_shapes.push_back(in_shape);
        //LOG(INFO) << "!!in shape #" << in_ent_id << " " << in_shape;
      }
      for (size_t k = 0; k < node->num_outputs(); ++k) {
        const uint32_t out_ent_id = idxgraph.entry_id(node_id, k);
        const TShape& out_shape = shapes[out_ent_id];
        dpop.output_entry_index.push_back(bfs.GetNodeEntryBFSIndex(out_ent_id));
        // Initial ghost area shape is the same as node entry shape.
        dpop.output_ghost_regions.emplace_back(out_shape);
        output_shapes.push_back(out_shape);
        //LOG(INFO) << "!!out shape #" << out_ent_id << " " << out_shape;
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
  
cost_t CutAlgorithm::OneCut() {
  CHECK_GT(dp_states_.size(), 0);
  // Reset DP states.
  Reset();
  // Init state for BFS level 0.
  for (DPState& state: dp_states_[0]) {
    state.cost = 0;
    for (const DPOp& op : dp_operators_[0]) {
      if (IsVariable(op)) {
        // Variable operator. Any scheme should be fine, so no conversion cost.
        // Just put index 0 as the chosen aligned request.
        state.chosen_aligned_requests.push_back(0);
        continue;
      }
      cost_t op_cost = 0;
      size_t chosen_request = 0;
      tie(op_cost, chosen_request) = ConversionCost(op, nullptr, &state, 0);
      state.cost += op_cost;
      state.chosen_aligned_requests.push_back(chosen_request);
    }
  }
  // Do DP.
  for (size_t i = 1; i < dp_states_.size(); ++i) {
    for (size_t j = 0; j < dp_states_[i].size(); ++j) {
      DPState& next_state = dp_states_[i][j];
      // Compute minimal cost to reach this state by looping all possible previous states.
      next_state.cost = std::numeric_limits<cost_t>::max();
      for (size_t k = 0; k < dp_states_[i-1].size(); ++k) {
        DPState& prev_state = dp_states_[i-1][k];
        cost_t state_cost = prev_state.cost;
        vector<size_t> op_requests;
        for (const DPOp& op : dp_operators_[i]) {
          if (IsVariable(op)) {
            // Variable operator. Any scheme should be fine, so conversion cost is zero.
            // Just put index 0 as the chosen aligned request.
            op_requests.push_back(0);
            continue;
          }
          cost_t op_cost = 0;
          size_t chosen_request = 0;
          tie(op_cost, chosen_request) = ConversionCost(op, &prev_state, &next_state, i);
          state_cost += op_cost;
          op_requests.push_back(chosen_request);
        }
        if (state_cost < next_state.cost) {
          // Record this.
          next_state.cost = state_cost;
          next_state.prev_state_index = k;
          next_state.chosen_aligned_requests = std::move(op_requests);
        }
      }
      //LOG(INFO) << "DP cost: level #" << i << " state #" << j << ": " << next_state.cost;
    }
  }
  // If the last level is node level, the total cost should also includes that.
  // TODO

  // Extract the optimal plan.
  return ExtractOptimalPlan();
}

pair<cost_t, size_t> CutAlgorithm::ConversionCost(
    const DPOp& op,
    const DPState* prev_state,
    const DPState* next_state,
    size_t lvl) const {
  //LOG(INFO) << src_graph_->indexed_graph()[op.node_id].source->attrs.name;
  const vector<Scheme>* prev_schemes = (prev_state)? &prev_state->schemes : nullptr;
  const vector<Scheme>* next_schemes = (next_state)? &next_state->schemes : nullptr;
  const vector<DPEntry>* prev_entries = (lvl > 0)? &dp_entries_[lvl-1] : nullptr;
  const vector<DPEntry>* next_entries = (lvl < dp_entries_.size())? &dp_entries_[lvl] : nullptr;
  // Extract schemes for inputs and outputs of the op.
  const vector<const Scheme*>& input_schemes =
    ExtractFromIndex<Scheme>(op.input_entry_index, prev_schemes, next_schemes, lvl);
  const vector<const Scheme*>& output_schemes =
    ExtractFromIndex<Scheme>(op.output_entry_index, prev_schemes, next_schemes, lvl);
  // Extract entries for inputs and outputs of the op.
  const vector<const DPEntry*>& input_entries =
    ExtractFromIndex<DPEntry>(op.input_entry_index, prev_entries, next_entries, lvl);
  const vector<const DPEntry*>& output_entries =
    ExtractFromIndex<DPEntry>(op.output_entry_index, prev_entries, next_entries, lvl);
  const vector<SchemeRequest>& aligned_requests = op.aligned_requests;
  CHECK_EQ(input_schemes.size(), input_entries.size());
  CHECK_EQ(input_schemes.size(), op.input_ghost_regions.size());
  CHECK_EQ(output_schemes.size(), output_entries.size());
  CHECK_EQ(output_schemes.size(), op.output_ghost_regions.size());
  CHECK_GT(aligned_requests.size(), 0);
  cost_t cost = std::numeric_limits<cost_t>::max();
  size_t req_idx = 0;
  for (size_t i = 0; i < aligned_requests.size(); ++i) {
    const SchemeRequest& req = aligned_requests[i];
    CHECK_EQ(input_schemes.size(), req.input_schemes.size());
    CHECK_EQ(output_schemes.size(), req.output_schemes.size());
    cost_t req_cost = 0;
    // Input conversion cost.
    for (size_t j = 0; j < input_schemes.size(); ++j) {
      req_cost += Region::ConvertCost2(input_entries[j]->region,
                                       *input_schemes[j],
                                       op.input_ghost_regions[j],
                                       req.input_schemes[j]);
    }
    // Output conversion cost.
    for (size_t j = 0; j < output_schemes.size(); ++j) {
      req_cost += Region::ConvertCost2(op.output_ghost_regions[j],
                                       req.output_schemes[j],
                                       output_entries[j]->region,
                                       *output_schemes[j]);
    }
    // Save the minimal cost.
    if (req_cost < cost) {
      cost = req_cost;
      req_idx = i;
    }
  }
  return make_pair(cost, req_idx);
}

cost_t CutAlgorithm::ExtractOptimalPlan() {
  size_t num_levels = dp_states_.size();
  cost_t min_cost = std::numeric_limits<cost_t>::max();
  DPState* min_state = nullptr;
  for (DPState& state : dp_states_[num_levels-1]) {
    if (state.cost < min_cost) {
      min_cost = state.cost;
      min_state = &state;
    }
  }
  LOG(INFO) << "Min cost: " << min_cost;
  for (int i = dp_states_.size() - 1; i >= 0; --i) {
    CHECK_EQ(dp_entries_[i].size(), min_state->schemes.size());
    CHECK_EQ(dp_operators_[i].size(), min_state->chosen_aligned_requests.size());
    // Record best scheme for each entry.
    for (size_t j = 0; j < dp_entries_[i].size(); ++j) {
      dp_entries_[i][j].chosen_schemes.push_back(
          min_state->schemes[j]);
    }
    // Record best aligned request for each operator. Variable operator will be ignored.
    for (size_t j = 0; j < dp_operators_[i].size(); ++j) {
      if (!IsVariable(dp_operators_[i][j])) {
        dp_operators_[i][j].chosen_aligned_requests.push_back(
            min_state->chosen_aligned_requests[j]);
      }
    }
    if (i > 0) {
      min_state = &dp_states_[i-1][min_state->prev_state_index];
    }
  }
  // TODO handle situation where the last BFS level is an operator level.
  return min_cost;
}

void CutAlgorithm::Print() const {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  for (size_t i = 0; i < dp_operators_.size(); ++i) {
    LOG(INFO) << "Level Node: [";
    for (const auto& dp_op : dp_operators_[i]) {
      const uint32_t nodeid = dp_op.node_id;
      const Node* node = graph[nodeid].source;
      LOG(INFO) << "\t#" << nodeid << ": \"" << node->attrs.name << "\""
                << (node->is_variable()? "(variable)" : "");
    }
    LOG(INFO) << "]";
    if (i < dp_entries_.size()) {
      LOG(INFO) << "Level NodeEntry: [";
      for (const auto& dp_ent : dp_entries_[i]) {
        const uint32_t groupid = dp_ent.entry_group_id;
        ostringstream oss;
        oss << "\t{";
        for (const uint32_t entid : (*bfs_.entry_groups_)[groupid]) {
          oss << "#" << entid << " ";
        }
        oss << "}, " << dp_ent.region << "[";
        for (const Scheme& sch : dp_ent.chosen_schemes) {
          oss << sch << " ";
        }
        oss << "]";
        LOG(INFO) << oss.str();
      }
      LOG(INFO) << "]";
    }
  }
}

// K-cut algorithm.
cost_t CutAlgorithm::KCuts(uint32_t K) {
  if (K == 0) {
    return 0;
  }
  // Compute one-cut.
  cost_t cut_cost = OneCut();
  // Prune entry regions.
  for (auto& ent_lvl : dp_entries_) {
    for (DPEntry& ent : ent_lvl) {
      const Scheme& cur_sch = ent.chosen_schemes[ent.chosen_schemes.size() - 1];
      ent.region = ent.region.Split2(cur_sch).first;
    }
  }
  // Compute ghost regions.
  for (auto& op_lvl : dp_operators_) {
    for (DPOp& op : op_lvl) {
      if (IsVariable(op)) {
        // For variable operator, all schemes are aligned. In fact, the ghost area is not
        // being considered when computing the conversion cost (the cost is always zero).
        // Therefore, no need to compute ghost regions for this.
        continue;
      }
      size_t cur_req = op.chosen_aligned_requests[op.chosen_aligned_requests.size() - 1];
      const SchemeRequest& req = op.aligned_requests[cur_req];
      CHECK_LT(cur_req, op.aligned_requests.size());
      // Inputs.
      for (size_t i = 0; i < op.input_ghost_regions.size(); ++i) {
        op.input_ghost_regions[i] = op.input_ghost_regions[i].Split2(
            req.input_schemes[i]).first;
      }
      // Outputs.
      for (size_t i = 0; i < op.output_ghost_regions.size(); ++i) {
        const Scheme& out_sch = req.output_schemes[i];
        if (out_sch.type != Scheme::kRed) {
          // The ghost area of the Reduction scheme is unchanged. Otherwise, split
          // the ghost area to form the new one.
          op.output_ghost_regions[i] =
            op.output_ghost_regions[i].Split2(out_sch).first;
        }
      }
    }
  }
  //LOG(INFO) << "!!!!!!!!!!!!!!!!!!!!!!!After prune";
  //Print();
  // Compute (k-1)-cut.
  return cut_cost + 2 * KCuts(K - 1);
}

}  // namespace pass
}  // namespace nnvm
