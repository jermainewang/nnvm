/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition.h
 * \brief The k-cuts partition algorithm.
 */

#include "./partition.h"

using namespace std;

namespace {
ostream& operator << (ostream& os, const nnvm::pass::Scheme& sch) {
  using nnvm::pass::Scheme;
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

nnvm::TShape operator / (const nnvm::TShape& shp1, const nnvm::TShape& shp2) {
  using nnvm::TShape;
  CHECK_EQ(shp1.ndim(), shp2.ndim());
  TShape ret = shp1;
  for (size_t i = 0; i < shp1.ndim(); ++i) {
    CHECK(shp2[i] != 0 && ret[i] % shp2[i] == 0);
    ret[i] /= shp2[i];
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
string GetPrefix(const string& str) {
  // TODO(minjie): Very ugly way of getting forward op name.
  string ret;
  size_t pos = str.find_last_of('_');
  if (pos == 0 || pos == string::npos) {
    return str;
  } else {
    return str.substr(0, pos);
  }
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
// If `ignore_jump` is set to true, entries that are not in the prev or next levels will be
// ignored. Rather than fail with error, it will put a nullptr as a placeholder.
template<typename T>
vector<const T*> ExtractFromIndex(
    const vector<Levels::Index>& index,
    const vector<T>* prev_level,
    const vector<T>* next_level,
    size_t current_level,
    bool ignore_jump) {
  vector<const T*> ret;
  for (const Levels::Index& idx : index) {
    if (idx.first == current_level) {
      // Content is in next level.
      CHECK_NOTNULL(next_level);
      ret.push_back(&(next_level->at(idx.second)));
    } else if (idx.first == current_level - 1) {
      // Content is in prev level.
      CHECK_NOTNULL(prev_level);
      ret.push_back(&(prev_level->at(idx.second)));
    } else if (ignore_jump) {
      ret.push_back(nullptr);
    } else {
      LOG(FATAL) << "Invalid entry index (" << idx.first << ", " << idx.second
                 << ") for operator in level #" << current_level;
    }
  }
  return ret;
}

class GridIndexMap {
 public:
  GridIndexMap(const Grid& grid) {
    gridindex2block_.resize(grid.TotalNumBlocks(), 0);
    for (size_t i = 0; i < grid.TotalNumBlocks(); ++i) {
      const Block& blk = grid.BlockAt(i);
      const size_t hash = BlockIndexHash(grid, blk.index, blk.replication_id);
      CHECK(hash < gridindex2block_.size());
      gridindex2block_[hash] = hash;
    }
  }

  // Hash value of the given grid index. If the index is a valid one, the value is
  // guaranteed to be within range [0, total_num_blocks).
  static size_t BlockIndexHash(const Grid& grid, const TShape& index, uint32_t rep_id) {
    CHECK_EQ(index.ndim(), grid.shape().ndim());
    CHECK_LT(rep_id, grid.num_replicates());
    size_t hash = 0, mult = 1;
    for (int i = index.ndim() - 1; i >= 0; --i) {
      CHECK_LT(index[i], grid.num_blocks()[i]);
      hash += index[i] * mult;
      mult *= grid.num_blocks()[i];
    }
    hash += rep_id * mult;
    return hash;
  }

  inline const Block& GetBlock(const Grid& grid, const TShape& index, uint32_t rep_id) const {
    size_t hash = gridindex2block_.at(BlockIndexHash(grid, index, rep_id));
    return grid.BlockAt(hash);
  }

  inline Block& GetBlock(Grid& grid, const TShape& index, uint32_t rep_id) {
    size_t hash = gridindex2block_.at(BlockIndexHash(grid, index, rep_id));
    return grid.BlockAt(hash);
  }

 private:
  // A map from grid index to the block index in the vector representation.
  // Since the grid index could be mapped to a continuous range from [0, total_num_blocks),
  // a vector could be used here instead of a map.
  std::vector<size_t> gridindex2block_;
};

class IndexIter {
 public:
  IndexIter(const TShape& limit): limit_(limit), index_(limit.ndim(), 0) { }

  bool Next() {
    for (int i = index_.ndim() - 1; i >= 0; --i) {
      index_[i] += 1;
      if (index_[i] == limit_[i]) {
        index_[i] = 0;
      } else {
        return true;
      }
    }
    return false;
  }

  const TShape& Get() const { return index_; }

 private:
  // (No ownership).
  const TShape& limit_;
  TShape index_;
};

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

void Levels::AddNode(uint32_t levelid, uint32_t nodeid) {
  CHECK(node2index_.find(nodeid) == node2index_.end())
    << "Node #" << nodeid << " has already been added to level #" << levelid;
  if (levelid >= node_levels_.size()) {
    // New level.
    node_levels_.push_back(vector<uint32_t>());
  }
  CHECK_LT(levelid, node_levels_.size());
  const uint32_t level_index = node_levels_[levelid].size();
  node_levels_[levelid].push_back(nodeid);
  node2index_[nodeid] = make_pair(levelid, level_index);
}

void Levels::AddNodeEntry(uint32_t levelid, uint32_t entry_id) {
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

BFS::BFS(Graph* src, const NodeEntryGroups* groups): Levels(groups), src_graph_(src) {
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
    }
    // For all output entries, put the node in the adj list.
    for (uint32_t outidx = 0; outidx < node.source->num_outputs(); ++outidx) {
      const uint32_t out_ent_id = graph.entry_id(node_id, outidx);
      entry_to_nodes_[out_ent_id].insert(node_id);
      node_to_entries_[node_id].insert(out_ent_id);
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

    if (level % 2 == 0) {
      if (visited_nodes.count(id) > 0) {
        continue;
      }
      // This is a Node.
      visited_nodes.insert(id);
      AddNode(level / 2, id);
      // Put all its input/output entries into queue.
      for (const uint32_t ent_id : node_to_entries_[id]) {
        if (visited_entries.count(ent_id) == 0) {
          queue.push(make_pair(level + 1, ent_id));
        }
      }
    } else {
      if (visited_entries.count(id) > 0) {
        continue;
      }
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
  LOG(INFO) << "NodeEntry To Node";
  for (uint32_t entid = 0; entid < entry_to_nodes_.size(); ++entid) {
    ostringstream oss;
    oss << "Entry#" << entid << ": ";
    for (uint32_t nodeid : entry_to_nodes_[entid]) {
      oss << nodeid << " ";
    }
    LOG(INFO) << oss.str();
  }
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
  LOG(INFO) << "#Levels: " << node_levels_.size();
}

NeuralLevels::NeuralLevels(Graph* src, const NodeEntryGroups* groups):
  Levels(groups), src_graph_(src) {
  // Create node groups.
  const IndexedGraph& graph = src_graph_->indexed_graph();
  nodeid2group_.resize(graph.num_nodes(), -1); // -1 means the node does not belong to any group.
  unordered_map<string, size_t> prefix2group;
  for (uint32_t nodeid = 0; nodeid < graph.num_nodes(); ++nodeid) {
    const string& name = graph[nodeid].source->attrs.name;
    CHECK(name == "" || name[0] == '_' || name.find_first_of('_') == name.find_last_of('_'))
      << "Unsupported operator name: \"" << name << "\"";
    if (name == "" || name == "sum_grad") {
      // TODO(minjie): This is an ugly way to ignore non-symbolic operators.
      // These nodes will be put in a group that contains only themselves.
      nodeid2group_[nodeid] = node_groups_.size();
      node_groups_.push_back({nodeid});
      continue;
    }
    const string& prefix = GetPrefix(name);
    if (prefix2group.find(prefix) == prefix2group.end()) {
      // New node group.
      //LOG(INFO) << "Group " << prefix;
      prefix2group[prefix] = node_groups_.size();
      node_groups_.push_back(vector<uint32_t>());
    }
    size_t groupid = prefix2group[prefix];
    nodeid2group_[nodeid] = groupid;
    node_groups_[groupid].push_back(nodeid);
  }
  for (size_t i = 0; i < node_groups_.size(); ++i) {
    LOG(INFO) << "Group #" << i << ": {";
    for (uint32_t nodeid : node_groups_[i]) {
      LOG(INFO) << "\t#" << nodeid << ": " << graph[nodeid].source->attrs.name << ",";
    }
    LOG(INFO) << "}";
  }
  // Following is the same as in BFS. Create undirected graph from original graph.
  entry_to_nodes_.resize(graph.num_node_entries());
  node_to_entries_.resize(graph.num_nodes());
  for (uint32_t node_id = 0; node_id < graph.num_nodes(); ++node_id) {
    const IndexedGraph::Node& node = graph[node_id];
    // For all input entries, put the node in the adj list.
    for (const IndexedGraph::NodeEntry& in_ent : node.inputs) {
      const uint32_t in_ent_id = graph.entry_id(in_ent);
      entry_to_nodes_[in_ent_id].insert(node_id);
      node_to_entries_[node_id].insert(in_ent_id);
    }
    // For all output entries, put the node in the adj list.
    for (uint32_t outidx = 0; outidx < node.source->num_outputs(); ++outidx) {
      const uint32_t out_ent_id = graph.entry_id(node_id, outidx);
      entry_to_nodes_[out_ent_id].insert(node_id);
      node_to_entries_[node_id].insert(out_ent_id);
    }
  }
}

void NeuralLevels::Run() {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  // Organize nodes in topological order.
  vector<uint32_t> topo_order;
  DFSVisit(src_graph_->outputs, [&](const NodePtr& node) {
      //LOG(INFO) << "Node #" << graph.node_id(node.get())
                //<< ": " << node->attrs.name;
      topo_order.push_back(graph.node_id(node.get()));
    });
  // Add node group in topo order.
  int curlevel = -1;
  vector<size_t> levelid(graph.num_nodes(), 0);
  vector<int> group2level(node_groups_.size(), -1);
  for (size_t i = 0; i < topo_order.size(); ++i) {
    const uint32_t nodeid = topo_order[i];
    const size_t groupid = nodeid2group_[nodeid];
    if (group2level[groupid] < 0) {
      if (i > 0 && levelid[i - 1] < curlevel) {
        // XXX(minjie): Special treatment for operators not appear in forward pass.
        // This is not a generic rule!
        group2level[groupid] = levelid[i - 1];
      } else {
        ++curlevel;
        group2level[groupid] = curlevel;
      }
    }
    levelid[nodeid] = group2level[groupid];
  }
  /*for (uint32_t nodeid = 0; nodeid < graph.num_nodes(); ++nodeid) {
      LOG(INFO) << "Node #" << nodeid
                << ": " << graph[nodeid].source->attrs.name
                << " " << levelid[nodeid];
  }*/
  // Make levels.
  for (uint32_t nodeid = 0; nodeid < graph.num_nodes(); ++nodeid) {
    AddNode(levelid[nodeid], nodeid);
  }
  for (uint32_t entid = 0; entid < graph.num_node_entries(); ++entid) {
    // Always add node entry to the smallest levels of its connected nodes.
    size_t entlvl = std::numeric_limits<size_t>::max();
    for (uint32_t nodeid : entry_to_nodes_[entid]) {
      entlvl = std::min(entlvl, levelid[nodeid]);
    }
    AddNodeEntry(entlvl, entid);
  }
}

void NeuralLevels::Print() const {
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  /*LOG(INFO) << "NodeEntry To Node";
  for (uint32_t entid = 0; entid < entry_to_nodes_.size(); ++entid) {
    ostringstream oss;
    oss << "Entry#" << entid << ": ";
    for (uint32_t nodeid : entry_to_nodes_[entid]) {
      oss << nodeid << " ";
    }
    LOG(INFO) << oss.str();
  }*/
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
  LOG(INFO) << "#Levels: " << node_levels_.size();
}


bool Region::CanSplit2(const Scheme& sch) const {
  switch (sch.type) {
  case Scheme::kCut:
    return region_shape_[sch.dim] % 2 == 0;
  case Scheme::kRep:
    return true;
  case Scheme::kRed:
    return false;
  default:
    LOG(FATAL) << "Scheme: " << sch << " is not supported for split.";
  }
  return false;
}

pair<Region, Region> Region::Split2(const Scheme& sch) const {
  switch (sch.type) {
  case Scheme::kCut:
    {
    TShape shp = region_shape_;
    CHECK_LT(sch.dim, region_shape_.ndim());
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
    } else if (!r1.CanSplit2(sch1) || !r2.CanSplit2(sch2)) {
      // Cannot split given the scheme. Return a very large cost that is guaranteed to
      // be worse.
      cost = 100 * (r1.Area() + r2.Area());
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

CutAlgorithm::CutAlgorithm(Graph* src, const Levels& levels, const NodeEntryGroups& groups):
  src_graph_(src), levels_(levels), entry_groups_(groups) {
  const IndexedGraph& idxgraph = src->indexed_graph();
  const OpMap<FAlignedSchemes>& align_map =
    Op::GetAttr<FAlignedSchemes>("FAlignedSchemes");
  const ShapeVector& shapes =
    src_graph_->GetAttr<ShapeVector>("shape");
  // Init DP structures.
  dp_entries_.resize(levels.NumEntryGroupLevels());
  for (size_t i = 0; i < levels.NumEntryGroupLevels(); ++i) {
    const auto& eglevel = levels.GetEntryGroupLevel(i);
    for (size_t j = 0; j < eglevel.size(); ++j) {
      DPEntry dpent;
      const uint32_t ent_group_id = eglevel[j];
      dpent.entry_group_id = ent_group_id;
      const uint32_t ent_id = *(entry_groups_[ent_group_id].begin());
      // The initial ghost region is the same as region.
      dpent.region = Region(shapes[ent_id]);
      dp_entries_[i].push_back(dpent);
    }
  }
  dp_operators_.resize(levels.NumNodeLevels());
  for (size_t i = 0; i < levels.NumNodeLevels(); ++i) {
    const auto& nodelevel = levels.GetNodeLevel(i);
    for (size_t j = 0; j < nodelevel.size(); ++j) {
      DPOp dpop;
      // Node id.
      const uint32_t node_id = nodelevel[j];
      dpop.node_id = node_id;
      // Input/Output entries.
      const Node* node = idxgraph[dpop.node_id].source;
      //LOG(INFO) << "!!" << node->attrs.name;
      vector<TShape> input_shapes, output_shapes;
      for (const NodeEntry& in_ent : node->inputs) {
        const uint32_t in_ent_id = idxgraph.entry_id(in_ent);
        const TShape& in_shape = shapes[in_ent_id];
        dpop.input_entry_index.push_back(levels.GetNodeEntryIndex(in_ent_id));
        // Initial ghost area shape is the same as node entry shape.
        dpop.input_ghost_regions.emplace_back(in_shape);
        input_shapes.push_back(in_shape);
        //LOG(INFO) << "!!in shape #" << in_ent_id << " " << in_shape;
      }
      for (size_t k = 0; k < node->num_outputs(); ++k) {
        const uint32_t out_ent_id = idxgraph.entry_id(node_id, k);
        const TShape& out_shape = shapes[out_ent_id];
        dpop.output_entry_index.push_back(levels.GetNodeEntryIndex(out_ent_id));
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

const std::vector<Scheme>& CutAlgorithm::GetEntrySchemes(uint32_t entry_id) const {
  const Levels::Index& index = levels_.GetNodeEntryIndex(entry_id);
  return dp_entries_[index.first][index.second].chosen_schemes;
}

const std::vector<SchemeRequest>& CutAlgorithm::GetSchemeRequests(
    uint32_t node_id) const {
  const Levels::Index& index = levels_.GetNodeIndex(node_id);
  return dp_operators_[index.first][index.second].aligned_requests;
}

// Get scheme requests chosen for the given node.
const std::vector<size_t>& CutAlgorithm::GetChosenSchemeRequests(
    uint32_t node_id) const {
  const Levels::Index& index = levels_.GetNodeIndex(node_id);
  return dp_operators_[index.first][index.second].chosen_aligned_requests;
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
    LOG(INFO) << "Running DP #" << i;
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
          //LOG(INFO) << src_graph_->indexed_graph()[op.node_id].source->attrs.name;
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
  const bool ignore_jump = levels_.AllowCrossLevelEdges();
  // Extract schemes for inputs and outputs of the op.
  const vector<const Scheme*>& input_schemes =
    ExtractFromIndex<Scheme>(op.input_entry_index, prev_schemes, next_schemes, lvl, ignore_jump);
  const vector<const Scheme*>& output_schemes =
    ExtractFromIndex<Scheme>(op.output_entry_index, prev_schemes, next_schemes, lvl, ignore_jump);
  // Extract entries for inputs and outputs of the op.
  const vector<const DPEntry*>& input_entries =
    ExtractFromIndex<DPEntry>(op.input_entry_index, prev_entries, next_entries, lvl, ignore_jump);
  const vector<const DPEntry*>& output_entries =
    ExtractFromIndex<DPEntry>(op.output_entry_index, prev_entries, next_entries, lvl, ignore_jump);
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
      if (input_schemes[j] == nullptr) {
        // Cannot get the scheme from either prev or next state. This may
        // because of the shortcut (jump edge) in the graph. In this case, simply
        // ignore the cost.
        CHECK(ignore_jump);
        continue;
      }
      req_cost += Region::ConvertCost2(input_entries[j]->region,
                                       *input_schemes[j],
                                       op.input_ghost_regions[j],
                                       req.input_schemes[j]);
    }
    // Output conversion cost.
    for (size_t j = 0; j < output_schemes.size(); ++j) {
      if (output_schemes[j] == nullptr) {
        // Cannot get the scheme from either prev or next state. This may
        // because of the shortcut (jump edge) in the graph. In this case, simply
        // ignore the cost.
        CHECK(ignore_jump);
        continue;
      }
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
        for (const uint32_t entid : entry_groups_[groupid]) {
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
  // Compute (k-1)-cut.
  return cut_cost + 2 * KCuts(K - 1);
}

Grid::Grid(const TShape& shape, const vector<Scheme>& schemes):
  shape_(shape), block_shape_(shape), num_blocks_(shape.ndim()) {
  // Initialize the grid with one block.
  Block blk;
  blk.index = TShape(shape.ndim(), 0);
  blocks_.push_back(std::move(blk));
  for (const Scheme& sch : schemes) {
    if (sch.type == Scheme::kRed) {
      replicate_is_reduction_ = true;
      break;
    }
  }
  // Partition the block using the given schemes.
  // Loop the scheme list in reverse order since this is easier (and more efficient)
  // for grid construction.
  for (int i = schemes.size() - 1; i >= 0; --i) {
    const Scheme& sch = schemes[i];
    PushScheme(sch, 2);
  }
  CHECK_EQ(num_blocks_.Size() * num_replicates_, blocks_.size());
  // Change the device group id for the blocks.
  for (size_t i = 0; i < blocks_.size(); ++i) {
    blocks_[i].device_group_id = i;
  }
}

void Grid::PushScheme(const Scheme& sch, size_t num_splits, Grid::SplitFn splitfn) {
  if (num_splits <= 1) {
    return;
  }
  schemes_.push_back(make_pair(sch, num_splits));
  // Double the block list in the grid.
  const size_t old_num_blks = blocks_.size();
  blocks_.resize(num_splits * old_num_blks);
  std::copy(blocks_.begin(), blocks_.begin() + old_num_blks,
            blocks_.begin() + old_num_blks);
  switch (sch.type) {
  case Scheme::kCut:
    {
      // Change block index.
      const int cut_dim = sch.dim;
      for (size_t i = old_num_blks; i < blocks_.size(); ++i) {
        blocks_[i].index[cut_dim] += num_blocks_[cut_dim];
      }
      num_blocks_[cut_dim] *= num_splits;
      block_shape_[cut_dim] /= num_splits;
      break;
    }
  case Scheme::kRep:
    {
      CHECK(!replicate_is_reduction_);
      // Change replication index.
      for (size_t i = old_num_blks; i < blocks_.size(); ++i) {
        blocks_[i].replication_id += num_replicates_;
      }
      num_replicates_ *= num_splits;
      break;
    }
  case Scheme::kRed:
    {
      CHECK(replicate_is_reduction_);
      // Change replication index.
      for (size_t i = old_num_blks; i < blocks_.size(); ++i) {
        blocks_[i].replication_id += num_replicates_;
      }
      num_replicates_ *= num_splits;
      break;
    }
  default:
    LOG(FATAL) << "Invalid scheme type: " << sch.type;
  }
  if (splitfn) {
    // Call splitfn.
    vector<Block*> to(num_splits);
    for (size_t i = 0; i < old_num_blks; ++i) {
      for (size_t j = 0; j < num_splits; ++j) {
        to[j] = &blocks_[i + j * old_num_blks];
      }
      splitfn(blocks_[i], to);
    }
  }
}

pair<Scheme, size_t> Grid::PopScheme(Grid::ConcatFn concatfn) {
  const pair<Scheme, size_t> last = schemes_[schemes_.size() - 1];
  const Scheme& sch = last.first;
  const size_t num_splits = last.second;
  schemes_.pop_back();
  const size_t new_num_blks = blocks_.size() / num_splits;
  vector<Block> new_blocks(new_num_blks);
  for (size_t i = 0; i < new_num_blks; ++i) {
    new_blocks[i].index = blocks_[i].index;
    new_blocks[i].replication_id = blocks_[i].replication_id;
    new_blocks[i].device_group_id = blocks_[i].device_group_id;
  }
  switch (sch.type) {
  case Scheme::kCut:
    {
      num_blocks_[sch.dim] /= num_splits;
      block_shape_[sch.dim] *= num_splits;
      break;
    }
  case Scheme::kRep:
    {
      CHECK(!replicate_is_reduction_);
      num_replicates_ /= num_splits;
      break;
    }
  case Scheme::kRed:
    {
      CHECK(replicate_is_reduction_);
      num_replicates_ /= num_splits;
      break;
    }
  default:
    LOG(FATAL) << "Invalid scheme type: " << sch.type;
  }
  if (concatfn) {
    // Call concatfn.
    vector<const Block*> from(num_splits);
    for (size_t i = 0; i < new_num_blks; ++i) {
      for (size_t j = 0; j < num_splits; ++j) {
        from[j] = &blocks_[i + j * new_num_blks];
      }
      concatfn(from, &new_blocks[i]);
    }
  }
  blocks_.swap(new_blocks);
  return last;
}

vector<NodeEntry> GraphPartitioner::SplitEntry(
    const NodeEntry& from, const std::string& name,
    size_t num_args, size_t dim) {
  CHECK_GT(num_args, 0);
  if (num_args == 1) {
    // Split operation is not needed.
    return {from};
  }
  // TODO: context map
  // Split op name.
  ostringstream oss;
  oss << "__" << name << "_split_" << dim;
  // Split op.
  const Op* split_op = Op::Get("_backward_Concat");
  NodePtr node = Node::Create();
  node->inputs.push_back(from);
  node->attrs.op = split_op;
  node->attrs.name = oss.str();
  node->attrs.dict["num_args"] = std::to_string(num_args);
  node->attrs.dict["dim"] = std::to_string(dim);
  node->attrs.op->attr_parser(&(node->attrs));
  vector<NodeEntry> ret;
  for (uint32_t i = 0; i < num_args; ++i) {
    ret.push_back(NodeEntry{node, i, 0});
  }
  return ret;
}

NodeEntry GraphPartitioner::ConcatEntry(
    const std::vector<NodeEntry>& from, const std::string& name, size_t dim) {
  CHECK(!from.empty());
  if (from.size() == 1) {
    // Concat operation is not needed.
    return from[0];
  }
  const Op* concat_op = Op::Get("Concat");
  // TODO: context map
  // Concat op.
  NodePtr node = Node::Create();
  node->inputs = from;
  node->attrs.op = concat_op;
  node->attrs.dict["num_args"] = std::to_string(from.size());
  node->attrs.dict["dim"] = std::to_string(dim);
  node->attrs.op->attr_parser(&(node->attrs));
  NodeEntry to{node, 0, 0};
  // Concat op name.
  ostringstream oss;
  oss << "__" << name << "_concat_" << dim;
  node->attrs.name = oss.str();
  return to;
}

void GraphPartitioner::AllReduceBlocks(
    const vector<const Block*>& inputs, const vector<Block*>& outputs) {
  CHECK_GT(inputs.size(), 1);
  // TODO: context map
  const Op* sum_op = Op::Get("ElementWiseSum");
  // Split for balanced allreduce.
  vector<vector<NodeEntry>> splitted(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    splitted[i] = SplitEntry(inputs[i]->entry, "red", outputs.size(), 0);
  }
  // Allreduce.
  vector<NodeEntry> reduced(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    NodePtr sum_node = Node::Create();
    sum_node->attrs.op = sum_op;
    sum_node->attrs.name = "__allreduce";
    sum_node->attrs.dict["num_args"] = std::to_string(inputs.size());
    sum_node->attrs.op->attr_parser(&(sum_node->attrs));
    for (size_t j = 0; j < inputs.size(); ++j) {
      sum_node->inputs.push_back(splitted[j][i]);
    }
    reduced[i] = NodeEntry{sum_node, 0, 0};
  }
  // Concat.
  for (size_t i = 0; i < outputs.size(); ++i) {
    outputs[i]->entry = ConcatEntry(reduced, "red", 0);
  }
}

void GraphPartitioner::AllShuffleBlocks(
    const vector<const Block*>& inputs, const vector<Block*>& outputs) {
  CHECK(!inputs.empty() && !outputs.empty());
  size_t fetch_idx = 0;
  for (Block* outblk : outputs) {
    bool find = false;
    // Prioritize local fetch.
    for (const Block* inblk : inputs) {
      if (inblk->device_group_id == outblk->device_group_id) {
        outblk->entry = inblk->entry;
        find = true;
        break;
      }
    }
    if (!find) {
      outblk->entry = inputs[fetch_idx]->entry;
      fetch_idx = (fetch_idx + 1) % inputs.size();
    }
  }
}

void GraphPartitioner::AllReduce(const Grid& input, Grid* output) {
  CHECK_GT(input.TotalNumBlocks(), 0);
  CHECK_EQ(input.num_blocks(), output->num_blocks());
  GridIndexMap ingrid_idx(input), outgrid_idx(*output);
  IndexIter iter(input.num_blocks());
  vector<const Block*> input_blocks(input.num_replicates());
  vector<Block*> output_blocks(output->num_replicates());
  // Do allreduce/shuffle for blocks of the same grid index.
  do {
    const TShape& curidx = iter.Get();
    for (size_t repid = 0; repid < input.num_replicates(); ++repid) {
      input_blocks[repid] = &(ingrid_idx.GetBlock(input, curidx, repid));
    }
    for (size_t repid = 0; repid < output->num_replicates(); ++repid) {
      output_blocks[repid] = &(outgrid_idx.GetBlock(*output, curidx, repid));
    }
    if (input.replicate_is_reduction()) {
      AllReduceBlocks(input_blocks, output_blocks);
    } else {
      AllShuffleBlocks(input_blocks, output_blocks);
    }
  } while(iter.Next());
}

void GraphPartitioner::ConvertGrid(const Grid& from, Grid* to) {
  CHECK_EQ(from.shape(), to->shape());
  CHECK(!to->replicate_is_reduction());
  if (from.num_blocks() == to->num_blocks() &&
      from.num_replicates() == to->num_replicates() &&
      !from.replicate_is_reduction()) {
    // No need for conversion.
    *to = from;
    return;
  }

  // Three phase conversion: split + allreduce/shuffle + concat
  // Note that the split is implemented by _backward_Concat.
  const TShape& max_num_blocks = max(from.num_blocks(), to->num_blocks());

  // Phase: Split
  const TShape& extra_from_cuts = max_num_blocks / from.num_blocks();
  Grid from_split = from;
  for (size_t i = 0; i < extra_from_cuts.ndim(); ++i) {
    if (extra_from_cuts[i] > 1) {
      from_split.PushScheme(
          Scheme::Cut(i), extra_from_cuts[i],
          [&] (const Block& from, const vector<Block*>& to) {
            // Split function here.
            const vector<NodeEntry>& splitted =
              SplitEntry(from.entry, "p1",
                         to.size(), i);
            for (uint32_t idx = 0; idx < to.size(); ++idx) {
              to[idx]->entry = splitted[idx];
            }
          });
    }
  }
  const TShape& extra_to_cuts = max_num_blocks / to->num_blocks();
  for (size_t i = 0; i < extra_to_cuts.ndim(); ++i) {
    if (extra_to_cuts[i] > 1) {
      to->PushScheme(Scheme::Cut(i), extra_to_cuts[i]);
    }
  }
  CHECK_EQ(from_split.num_blocks(), to->num_blocks());

  // Phase: Allreduce/shuffle
  AllReduce(from_split, to);
  
  // Phase: Concat
  for (int i = extra_to_cuts.ndim() - 1; i >= 0; --i) {
    if (extra_to_cuts[i] > 1) {
      to->PopScheme(
          [&] (const vector<const Block*>& from, Block* to) {
            // Concat function.
            vector<NodeEntry> ent;
            for (auto blk : from) {
              ent.push_back(blk->entry);
            }
            to->entry = ConcatEntry(ent, "p3", i);
          });
    }
  }
}

void GraphPartitioner::PerformOp(const vector<const Grid*>& inputs,
                                 const vector<Grid*>& outputs,
                                 const vector<NodePtr>& nodes) {
  // Split operators.
  const uint32_t num_devices = nodes.size();
  // TODO(minjie): NodeEntry version?
  if (inputs.empty()) {
    // Variable nodes.
    CHECK_EQ(outputs.size(), 1);
    for (uint32_t dev = 0; dev < num_devices; ++dev) {
      CHECK(nodes[dev]->attrs.op == nullptr);
      outputs[0]->BlockAt(dev).entry = NodeEntry{nodes[dev], 0, 0};
    }
  } else {
    for (uint32_t dev = 0; dev < num_devices; ++dev) {
      nodes[dev]->inputs.resize(inputs.size());
      for (uint32_t i = 0; i < inputs.size(); ++i) {
        CHECK_EQ(inputs[i]->TotalNumBlocks(), num_devices);
        nodes[dev]->inputs[i] = inputs[i]->BlockAt(dev).entry;
      }
      for (uint32_t i = 0; i < outputs.size(); ++i) {
        CHECK_EQ(outputs[i]->TotalNumBlocks(), num_devices);
        outputs[i]->BlockAt(dev).entry = NodeEntry{nodes[dev], i, 0};
      }
    }
  }
}

Graph GraphPartitioner::Run(uint32_t K) {
  // TODO:
  // - control dependencies
  // - NodeEntry versions
  // - Shape vector
  
  const IndexedGraph& graph = src_graph_->indexed_graph();
  const ShapeVector& shapes = src_graph_->GetAttr<ShapeVector>("shape");
  // Partitioned grid of each entry in the original graph.
  vector<Grid> entry_grids;
  entry_grids.reserve(graph.num_node_entries());
  // Input/Output grids of each operator.
  vector<vector<Grid>> op_input_grids, op_output_grids;
  op_input_grids.resize(graph.num_nodes());
  op_output_grids.resize(graph.num_nodes());
  // Partitioned operators.
  vector<vector<NodePtr>> splitted_nodes;
  splitted_nodes.resize(graph.num_nodes());

  for (uint32_t entid = 0; entid < graph.num_node_entries(); ++entid) {
    LOG(INFO) << "Split entry#" << entid;
    const TShape& shape = shapes[entid];
    const vector<Scheme>& schemes = algo_.GetEntrySchemes(entid);
    entry_grids.emplace_back(shape, schemes);
  }
  for (uint32_t nodeid = 0; nodeid < graph.num_nodes(); ++nodeid) {
    const Node* node = graph[nodeid].source;
    LOG(INFO) << "Split node#" << nodeid << ": " << node->attrs.name;
    if (node->is_variable()) {
      // Variable node does not have input/output grid because it is always
      // aligned. A series of special split operations will be inserted after
      // each variable node to dispatch blocks to different devices.
      const uint32_t out_ent_id = graph.entry_id(nodeid, 0);
      // Create splitted nodes.
      // TODO: version ?
      // TODO: context map
      for (size_t i = 0; i < entry_grids[out_ent_id].TotalNumBlocks(); ++i) {
        NodePtr newnode = Node::Create();
        newnode->attrs = node->attrs;
        newnode->attrs.name = node->attrs.name + "_" + std::to_string(i);
        entry_grids[out_ent_id].BlockAt(i).entry = {newnode, 0, 0};
      }
      continue;
    }
    const vector<SchemeRequest>& allreqs = algo_.GetSchemeRequests(nodeid);
    const vector<size_t>& chosen = algo_.GetChosenSchemeRequests(nodeid);
    const size_t num_inputs = allreqs[0].input_schemes.size();
    const size_t num_outputs = allreqs[0].output_schemes.size();
    vector<vector<Scheme>> input_schemes(num_inputs), output_schemes(num_outputs);
    for (size_t choseid : chosen) {
      for (size_t i = 0; i < num_inputs; ++i) {
        input_schemes[i].push_back(allreqs[choseid].input_schemes[i]);
      }
      for (size_t i = 0; i < num_outputs; ++i) {
        output_schemes[i].push_back(allreqs[choseid].output_schemes[i]);
      }
    }
    vector<Grid> input_grids, output_grids;
    CHECK_EQ(node->inputs.size(), num_inputs);
    CHECK_EQ(node->num_outputs(), num_outputs);
    for (size_t i = 0; i < num_inputs; ++i) {
      const uint32_t in_ent_id = graph.entry_id(node->inputs[i]);
      const TShape& shape = shapes[in_ent_id];
      input_grids.emplace_back(shape, input_schemes[i]);
    }
    for (size_t i = 0; i < num_outputs; ++i) {
      const uint32_t out_ent_id = graph.entry_id(nodeid, i);
      const TShape& shape = shapes[out_ent_id];
      output_grids.emplace_back(shape, output_schemes[i]);
    }
    op_input_grids[nodeid].swap(input_grids);
    op_output_grids[nodeid].swap(output_grids);

    // Split attributes.
    vector<NodeAttrs> attrs = {node->attrs};
    for (size_t choseid : chosen) {
      const SchemeRequest& req = allreqs[choseid];
      CHECK(req.partitioner);
      vector<NodeAttrs> tmp;
      for (const NodeAttrs& parent : attrs) {
        const vector<NodeAttrs>& child = req.partitioner(parent, 2);
        for (const auto& attr : child) {
          tmp.push_back(attr);
        }
      }
      attrs.swap(tmp);
    }
    // Create splitted nodes.
    for (size_t i = 0; i < attrs.size(); ++i) {
      NodePtr n = Node::Create();
      n->attrs = attrs[i];
      n->attrs.name = node->attrs.name + "_" + std::to_string(i);
      splitted_nodes[nodeid].push_back(n);
    }
  }
    
  DFSVisit(src_graph_->outputs, [&](const NodePtr& node) {
    LOG(INFO) << "Processing Node: " << node->attrs.name;
    const uint32_t nodeid = graph.node_id(node.get());
    if (node->is_variable()) {
      // For variable node. Nothing should be done.
      return;
    }
    // Convert input grids.
    vector<const Grid*> aligned_ingrid(node->inputs.size());
    for (size_t i = 0; i < node->inputs.size(); ++i) {
      const uint32_t in_ent_id = graph.entry_id(node->inputs[i]);
      const Grid& ingrid = entry_grids[in_ent_id];
      Grid& aligned = op_input_grids[nodeid][i];
      LOG(INFO) << "\tConvert input #" << i;
      ConvertGrid(ingrid, &aligned);
      aligned_ingrid[i] = &aligned;
    }
    vector<Grid*> outgrid(node->num_outputs());
    vector<Grid*> aligned_outgrid(node->num_outputs());
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      const uint32_t out_ent_id = graph.entry_id(nodeid, i);
      outgrid[i] = &entry_grids[out_ent_id];
      aligned_outgrid[i] = &op_output_grids[nodeid][i];
    }

    LOG(INFO) << "\tPerform op";
    PerformOp(aligned_ingrid, aligned_outgrid, splitted_nodes[nodeid]);

    // Convert output grids.
    for (size_t i = 0; i < node->num_outputs(); ++i) {
      LOG(INFO) << "\tConvert output #" << i;
      ConvertGrid(*aligned_outgrid[i], outgrid[i]);
    }
  });

  // Final graph.
  Graph ret;
  for (const NodeEntry& out_ent : src_graph_->outputs) {
    const uint32_t entid = graph.entry_id(out_ent);
    for (size_t i = 0; i < entry_grids[entid].TotalNumBlocks(); ++i) {
      ret.outputs.push_back(std::move(entry_grids[entid].BlockAt(i).entry));
    }
  }
  const IndexedGraph& retgraph = ret.indexed_graph();
  // TODO: new shape vector.
  ShapeVector new_shapes(retgraph.num_node_entries());
  
  DFSVisit(ret.outputs, [&](const NodePtr& node) {
    LOG(INFO) << node->attrs.name;
  });

  //return ret;
  return *src_graph_;
}

}  // namespace pass
}  // namespace nnvm
