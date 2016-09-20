/*!
 *  Copyright (c) 2016 by Minjie Wang
 * \file partition_pass.cc
 * \brief Auto-partition dataflow graph
 */
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>

#include "./partition.h"

using namespace std;

namespace nnvm {
namespace pass {
namespace {

// Visitor function for PrintPass.
void PrintPassVisitor(const NodePtr& n) {
  if (n->op()) {
    ostringstream oss;
    for (const auto& map_pair : n->attrs.dict) {
      oss << map_pair.first << " : " << map_pair.second << ", ";
    }
    LOG(INFO) << "PrintPass: Node: \"" << n->attrs.name << "\"; Op \""
              << n->op()->name << "\"; Attrs: {" << oss.str() << "}";
  }
}

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

  // Cut algorithm.
  CutAlgorithm algo(&src, bfs);
  //algo.OneCut();
  //algo.Print();
  cost_t total_cost = algo.KCuts(3);
  algo.Print();
  LOG(INFO) << "Total K-cuts cost: " << total_cost;

  return src;
}

NNVM_REGISTER_PASS(PartitionPass)
.describe("Partition tensors in graph and place them to multiple devices.")
.set_body(PartitionPass)
.depend_graph_attr("shape")  // Shape information from InferShapePass.
.depend_graph_attr("forward2backward")  // Gradient information from GradientPass.
.depend_graph_attr("backward2forward")  // Gradient information from GradientPass.
.depend_op_attr("FAlignedSchemes")  // Require op to provide aligned schemes.
.set_change_graph(true);


}  // namespace pass
}  // namespace nnvm
