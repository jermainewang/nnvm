/*!
 *  Copyright (c) 2016 by Contributors
 * \file graph.h
 * \brief Configuation of nnvm as well as basic data structure.
 */
#ifndef NNVM_GRAPH_H_
#define NNVM_GRAPH_H_

#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include "./base.h"
#include "./node.h"
#include "./symbolic.h"

namespace nnvm {

class IndexedGraph;
static const uint32_t kInvalidId = 0xffffffff;

namespace attr {
  static const int kGraph = 1;
  static const int kNode = 2;
  static const int kNodeEntry = 4;
};  // namespace attr

/*!
 * \brief Symbolic computation graph.
 *  This is the intermediate representation for optimization pass.
 */
class Graph {
 public:
  static const std::string kNodeAttrPrefix;
  static const std::string kNodeEntryAttrPrefix;
  /*! \brief outputs of the computation graph. */
  std::vector<NodeEntry> outputs;
  /*!
   * \brief Attributes of a graph
   *  [DEPRECATED!] Please use the set/get interface for different attribute types.
   *
   *  Note that attribute is shared pointer and can be shared across graphs.
   *
   *  It is highly recommended to keep each attribute immutable.
   *  It is also safe to implement an copy-on-write semnatics.
   *
   *  Copy when shared_ptr.unique is not true, while reuse original space
   *  when shared_ptr.unique is true.
   */
  std::unordered_map<std::string, std::shared_ptr<any> > attrs;
  /*!
   * \brief [DEPRECATED!] Get the immutable attribute from attrs.
   * \param attr_name the name of the attribute
   * \return the reference to corresponding attribute
   * \tparam T the type of the attribute.
   */
  template<typename T>
  inline const T& GetAttr(const std::string& attr_name) const;

  template<typename T>
  inline T& GetAttr(const std::string& attr_name);

  inline bool HasAttr(const std::string& attr_name) const {
    return attrs.count(attr_name) != 0;
  }

  /*!
   * \brief Get a move copy of the attribute, implement copy on write semantics.
   *  The content is moved if the reference counter of shared_ptr is 1.
   *  The attribute is erased from attrs after the call.
   *
   * \param attr_name the name of the attribute
   * \return a new copy of the corresponding attribute.
   * \tparam T the type of the attribute.
   */
  template<typename T>
  inline T MoveCopyAttr(const std::string& attr_name);

  template<typename T>
  inline const T& GetGraphAttr(const std::string& attr_name) const {
    return this->GetAttr<T>(attr_name);
  }

  // Note: NOT THREAD SAFE!
  template<typename T>
  inline void SetGraphAttr(const std::string& attr_name, const T& value) {
    attrs[attr_name] = value;
  }

  template<typename T>
  inline const T& GetNodeAttr(const std::string& attr_name, uint32_t nodeid) const;

  // Note: NOT THREAD SAFE!
  template<typename T>
  inline void SetNodeAttr(const std::string& attr_name,
                          uint32_t nodeid, const T& value);

  // Note: NOT THREAD SAFE!
  template<typename T>
  inline void SetNodeAttr(const std::string& attr_name, const std::vector<T>& attrs) {
    this->SetGraphAttr(kNodeAttrPrefix + attr_name, attrs);
  }

  template<typename T>
  inline const T& GetNodeEntryAttr(const std::string& attr_name, uint32_t entryid) const;

  // Note: NOT THREAD SAFE!
  template<typename T>
  inline void SetNodeEntryAttr(const std::string& attr_name,
                               uint32_t entryid, const T& value);

  // Note: NOT THREAD SAFE!
  template<typename T>
  inline void SetNodeEntryAttr(const std::string& attr_name, const std::vector<T>& attrs) {
    this->SetGraphAttr(kNodeEntryAttrPrefix + attr_name, attrs);
  }

  /*!
   * \brief get a indexed graph of current graph, if not exist, create it on demand
   * \return The indexed graph.
   * \sa IndexedGraph
   */
  const IndexedGraph& indexed_graph();

 private:
  // internal structure of indexed graph
  std::shared_ptr<const IndexedGraph> indexed_graph_;
};

/*!
 * \brief Auxililary data structure to index a graph.
 *  It maps Nodes in the graph to consecutive integers node_id.
 *  It also maps IndexedGraph::NodeEntry to consecutive integer entry_id.
 *  This allows storing properties of Node and NodeEntry into
 *  compact vector and quickly access them without resorting to hashmap.
 *
 *  The node_id and entry_rptr are the same as the JSON graph produced by SaveJSON Pass.
 */
class IndexedGraph {
 public:
  /*! \brief represents a data in the graph */
  struct NodeEntry {
    /*! \brief the source node id in the computation graph */
    uint32_t node_id;
    /*! \brief index of output from the source. */
    uint32_t index;
    /*! \brief version of the node */
    uint32_t version;
  };
  /*! \brief Node data structure in IndexedGraph */
  struct Node {
    /*! \brief pointer to the source node */
    const nnvm::Node* source;
    /*! \brief inputs to the node */
    array_view<NodeEntry> inputs;
    /*! \brief control flow dependencies to the node */
    array_view<uint32_t> control_deps;
  };
  /*! \return number of nodes in the graph */
  inline size_t num_nodes() const {
    return nodes_.size();
  }
  /*! \return total number of NodeEntry in the graph */
  inline size_t num_node_entries() const {
    return entry_rptr_.back();
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given IndexedGraph::NodeEntry
   * \param node_id The node index
   * \param index the output index
   * \return the unique index.
   */
  inline uint32_t entry_id(uint32_t node_id, uint32_t index) const {
    return entry_rptr_[node_id] + index;
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given IndexedGraph::NodeEntry
   * \param e The entry to query for index.
   * \return the unique index.
   */
  inline uint32_t entry_id(const NodeEntry& e) const {
    return entry_rptr_[e.node_id] + e.index;
  }
  /*!
   * \brief Get a unique entry id between 0 to num_node_entries()
   *  for a given NodeEntry.
   * \param e The entry to query for index.
   * \return the unique index.
   */
  inline uint32_t entry_id(const nnvm::NodeEntry& e) const {
    return entry_rptr_[node_id(e.node.get())] + e.index;
  }
  /*!
   * \brief Get the corresponding node id for a given Node in the IndexedGraph.
   * \param node The Node to query for index.
   * \return the node index.
   */
  inline uint32_t node_id(const nnvm::Node* node) const {
    return node2index_.at(node);
  }
  /*!
   * \brief Get the corresponding Node structure for a given node_id.
   * \param node_id The node id
   * \return const reference to the corresponding IndexedGraph::Node
   */
  inline const Node& operator[](uint32_t node_id) const {
    return nodes_[node_id];
  }
  /*!
   * \brief Get the corresponding Node structure
   * \param node The pointer to the Node structure
   * \return const reference to the corresponding IndexedGraph::Node
   */
  inline const Node& operator[](const nnvm::Node* node) const {
    return nodes_[node_id(node)];
  }
  /*! \return list of argument nodes */
  inline const std::vector<uint32_t>& input_nodes() const {
    return input_nodes_;
  }
  /*! \return list of mutable nodes */
  inline const std::unordered_set<uint32_t>& mutable_input_nodes() const {
    return mutable_input_nodes_;
  }
  /*! \return list of output entries */
  inline const std::vector<NodeEntry>& outputs() const {
    return outputs_;
  }
  // disalllow copy assign
  IndexedGraph(const IndexedGraph&) = delete;

 private:
  friend class Graph;
  /*!
   * \brief Constructor an IndexedGraph from normal Graph
   * \param other The source graph.
   */
  explicit IndexedGraph(const Graph& other);
  // Node pointers in CSR structure.
  std::vector<Node> nodes_;
  // Index to all input nodes.
  std::vector<uint32_t> input_nodes_;
  // Index to all mutable input nodes.
  std::unordered_set<uint32_t> mutable_input_nodes_;
  // space to store the outputs entries
  std::vector<NodeEntry> outputs_;
  // mapping from node to index.
  std::unordered_map<const nnvm::Node*, uint32_t> node2index_;
  // CSR pointer of node entries
  std::vector<size_t> entry_rptr_;
  // space to store input entries of each
  std::vector<NodeEntry> input_entries_;
  // control flow dependencies
  std::vector<uint32_t> control_deps_;
};

/*!
 * \brief perform a Post Order DFS visit to each node in the graph.
 *  This order is deterministic and is also topoligical sorted.
 * \param heads The heads in the graph.
 * \param fvisit a function of type std::function<void(const std::shared_ptr<Node>&)>
 * \tparam FVisit The function type to perform the visit.
 */
template<typename FVisit>
inline void DFSVisit(const std::vector<NodeEntry>& heads, FVisit fvisit);

// inline function implementations
template<typename T>
inline const T& Graph::GetAttr(const std::string& attr_name) const {
  auto it = attrs.find(attr_name);
  CHECK(it != attrs.end())
      << "Cannot find attribute " << attr_name << " in the graph";
  return nnvm::get<T>(*it->second);
}

template<typename T>
inline T& Graph::GetAttr(const std::string& attr_name) {
  auto it = attrs.find(attr_name);
  CHECK(it != attrs.end())
      << "Cannot find attribute " << attr_name << " in the graph";
  return nnvm::get<T>(*it->second);
}

template<typename T>
inline T Graph::MoveCopyAttr(const std::string& attr_name) {
  auto it = attrs.find(attr_name);
  CHECK(it != attrs.end())
      << "Cannot find attribute " << attr_name << " in the graph";
  std::shared_ptr<any> sptr = it->second;
  attrs.erase(it);
  if (sptr.unique()) {
    return std::move(nnvm::get<T>(*sptr));
  } else {
    return nnvm::get<T>(*sptr);
  }
}
template<typename T>
inline const T& Graph::GetNodeAttr(
    const std::string& attr_name, uint32_t nodeid) const {
  const std::vector<T>& node_attrs = this->GetAttr<std::vector<T>>(
      kNodeAttrPrefix + attr_name);
  CHECK(nodeid < node_attrs.size());
  return node_attrs[nodeid];
}

template<typename T>
inline void Graph::SetNodeAttr(
  const std::string& attr_name, uint32_t nodeid, const T& value) {
  std::vector<T>& node_attrs = this->GetAttr<std::vector<T>>(
      kNodeAttrPrefix + attr_name);
  if (nodeid >= node_attrs.size()) {
    node_attrs.resize(nodeid + 1);
  }
  node_attrs[nodeid] = value;
}

template<typename T>
inline const T& Graph::GetNodeEntryAttr(
    const std::string& attr_name, uint32_t entryid) const {
  const std::vector<T>& entry_attrs = this->GetAttr<std::vector<T>>(
      kNodeEntryAttrPrefix + attr_name);
  CHECK(entryid < entry_attrs.size());
  return entry_attrs[entryid];
}

template<typename T>
inline void Graph::SetNodeEntryAttr(
    const std::string& attr_name, uint32_t entryid, const T& value) {
  std::vector<T>& entry_attrs = this->GetAttr<std::vector<T>>(
      kNodeEntryAttrPrefix + attr_name);
  if (entryid >= entry_attrs.size()) {
    entry_attrs.resize(entryid + 1);
  }
  entry_attrs[entryid] = value;
}


template <typename GNode, typename HashType,
           typename FVisit, typename HashFunc,
          typename InDegree, typename GetInput>
void PostOrderDFSVisit(const std::vector<GNode>& heads,
                       FVisit fvisit,
                       HashFunc hash,
                       InDegree indegree,
                       GetInput getinput) {
  std::vector<std::pair<GNode, uint32_t> > stack;
  std::unordered_set<HashType> visited;
  for (auto& head : heads) {
    HashType head_hash = hash(head);
    if (visited.count(head_hash) == 0) {
      stack.push_back(std::make_pair(head, 0));
      visited.insert(head_hash);
    }
    while (!stack.empty()) {
      std::pair<GNode, uint32_t>& back = stack.back();
      if (back.second == indegree(back.first)) {
        fvisit(back.first);
        stack.pop_back();
      } else {
        const GNode& input = getinput(back.first, back.second++);
        HashType input_hash = hash(input);
        if (visited.count(input_hash) == 0) {
          stack.push_back(std::make_pair(input, 0));
          visited.insert(input_hash);
        }
      }
    }
  }
}

template<typename FVisit>
inline void DFSVisit(const std::vector<NodeEntry>& heads,
                     FVisit fvisit) {
  typedef const NodePtr* GNode;
  std::vector<GNode> head_nodes(heads.size());
  std::transform(heads.begin(), heads.end(), head_nodes.begin(),
                 [](const NodeEntry& e)->GNode {
                   return &e.node;
                 });
  PostOrderDFSVisit<GNode, Node*>(
      head_nodes,
      [fvisit](GNode n) { fvisit(*n); },  // FVisit
      [](GNode n)->Node* { return n->get(); },  // HashFunc
      [](GNode n)->uint32_t {  // InDegree
        return (*n)->inputs.size() + (*n)->control_deps.size();
      },
      [](GNode n, uint32_t index)->GNode {  // GetInput
        if (index < (*n)->inputs.size()) {
          return &(*n)->inputs.at(index).node;
        } else {
          return &(*n)->control_deps.at(index - (*n)->inputs.size());
        }
      });
}

}  // namespace nnvm

#endif  // NNVM_GRAPH_H_
