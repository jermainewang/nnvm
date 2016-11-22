/*!
 *  Copyright (c) 2016 by Contributors
 * \file place_device.cc
 * \brief Inference the device of each operator given known information.
 *  Insert a copy node automatically when there is a cross device.
 */
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/op_attr_types.h>
#include <nnvm/graph_attr_types.h>

namespace nnvm {
namespace pass {
namespace {

static const int kUnknownDevice = -1;

/*
 * \brief Pass that assigns nodes to the given devices.
 *
 * The pass decides placement in following descending priorities:
 * 1. The value (under key specified by `device_group_attr_key`) in the nodes' attrs
 *    dictionary. The value represents which virtual "group" the node belongs to. Another
 *    map `device_assign_map` given in the argument specifies how each "group" is assigned
 *    to which "device" (represented by an integer). The pass will simply assign the
 *    node to the device its group belongs to.
 * 2. Put the node on the same device as *any* of its preceding nodes.
 */
class PlaceDevicePass : public Pass {
 public:
  PassResult RunOnGraph(Graph src, const PassArgument& pargs) {
    PassResult ret;
    const PlaceDevicePassArgs& args = nnvm::get<PlaceDevicePassArgs>(pargs.value);
    const IndexedGraph& idx = src.indexed_graph();
    const Op* copy_op = Op::Get(args.device_copy_op);
    DeviceVector device(idx.num_nodes(), kUnknownDevice);

    // Forward pass
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const auto& inode = idx[nid];
      auto it = inode.source->attrs.dict.find(args.device_group_attr_key);
      if (it != inode.source->attrs.dict.end()) {
        // Assign device according to the group information.
        const std::string& device_group = it->second;
        auto dit = args.device_assign_map.find(device_group);
        CHECK_NE(dit, args.device_assign_map.end())
            << "The device assignment not found for group " << device_group;
        device[nid] = dit->second;
      } else {
        // Assign device according to *any* of its preceding nodes.
        for (const IndexedGraph::NodeEntry& e : inode.inputs) {
          if (device[e.node_id] != kUnknownDevice) {
            device[nid] = device[e.node_id];
            break;
          }
        }
      }
    }
    // backward pass
    for (uint32_t i = idx.num_nodes(); i != 0; --i) {
      const uint32_t nid = i - 1;
      const auto& inode = idx[nid];
      if (device[nid] == kUnknownDevice) {
        continue;
      }
      for (const IndexedGraph::NodeEntry& e : inode.inputs) {
        if (device[e.node_id] == kUnknownDevice) {
          // Assign device according to *any* of its preceding nodes.
          device[e.node_id] = device[nid];
        }
      }
    }

    // Detect whether there are multiple devices assigned in the graph.
    bool has_multiple_devices = false;
    for (size_t i = 0; i < device.size(); ++i) {
      if (device[i] == kUnknownDevice) {
        // For all unknown devices, simply assign them to device 0.
        device[i] = 0;
      }
      if (device[i] != device[0]) {
        has_multiple_devices = true;
      }
    }

    if (!has_multiple_devices) {
      // Only one device in the graph, no need to instrument copy nodes.
      ret.graph = src;
      ret.graph.SetNodeAttr("device", std::move(device));
      return ret;
    }

    // Instrument copy nodes between nodes that are assigned to different devices.
    std::map<std::tuple<uint32_t, uint32_t, int>, NodePtr> copy_map;
    std::vector<NodePtr> new_node_map(idx.num_nodes(), nullptr);
    std::unordered_map<const Node*, int> new_device_map;
    static auto& fmutate_inputs = Op::GetAttr<FMutateInputs>("FMutateInputs");

    // insert copy node
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      int dev_id = device[nid];
      const auto& inode = idx[nid];
      // check if mutation is needed
      bool need_mutate = false;
      if (!inode.source->is_variable() && fmutate_inputs.count(inode.source->op())) {
        for (uint32_t index : fmutate_inputs[inode.source->op()](inode.source->attrs)) {
          auto e = inode.inputs[index];
          if (new_node_map[e.node_id] != nullptr || dev_id != device[e.node_id]) {
            LOG(FATAL) << " mutable state cannot go across device"
                       << " op=" << inode.source->op()->name
                       << " input_state_index=" << index;
          }
        }
      }
      for (const IndexedGraph::NodeEntry& e : inode.inputs) {
        if (new_node_map[e.node_id] != nullptr || dev_id != device[e.node_id]) {
          need_mutate = true; break;
        }
      }
      if (!need_mutate) {
        for (const uint32_t cid : inode.control_deps) {
          if (new_node_map[cid] != nullptr)  {
            need_mutate = true; break;
          }
        }
      }
      if (inode.source->is_variable()) {
        CHECK(!need_mutate) << "consistency check";
      }
      if (need_mutate) {
        NodePtr new_node = Node::Create();
        new_node->attrs = inode.source->attrs;
        new_node->inputs.reserve(inode.inputs.size());
        for (size_t i = 0; i < inode.inputs.size(); ++i) {
          const IndexedGraph::NodeEntry& e = inode.inputs[i];
          if (dev_id != device[e.node_id]) {
            auto copy_key = std::make_tuple(e.node_id, e.index, dev_id);
            auto it = copy_map.find(copy_key);
            if (it != copy_map.end() && it->first == copy_key) {
              new_node->inputs.emplace_back(
                  NodeEntry{it->second, 0, 0});
            } else {
              NodePtr copy_node = Node::Create();
              std::ostringstream os;
              os << inode.source->inputs[i].node->attrs.name << "_" << e.index <<"_copy";
              copy_node->attrs.op = copy_op;
              copy_node->attrs.name = os.str();
              if (new_node_map[e.node_id] != nullptr) {
                copy_node->inputs.emplace_back(
                  NodeEntry{new_node_map[e.node_id], e.index, 0});
              } else {
                copy_node->inputs.push_back(inode.source->inputs[i]);
              }
              if (copy_node->attrs.op->attr_parser != nullptr) {
                copy_node->attrs.op->attr_parser(&(copy_node->attrs));
              }
              copy_map[copy_key] = copy_node;
              new_device_map[copy_node.get()] = dev_id;
              new_node->inputs.emplace_back(
                  NodeEntry{std::move(copy_node), 0, 0});
            }
          } else {
            if (new_node_map[e.node_id] != nullptr) {
              new_node->inputs.emplace_back(
                  NodeEntry{new_node_map[e.node_id], e.index, 0});
            } else {
              new_node->inputs.push_back(inode.source->inputs[i]);
            }
          }
        }
        new_node->control_deps.reserve(inode.control_deps.size());
        for (size_t i = 0; i < inode.control_deps.size(); ++i) {
          uint32_t cid = inode.control_deps[i];
          if (new_node_map[cid] != nullptr) {
            new_node->control_deps.push_back(new_node_map[cid]);
          } else {
            new_node->control_deps.push_back(inode.source->control_deps[i]);
          }
        }
        new_device_map[new_node.get()] = dev_id;
        new_node_map[nid] = std::move(new_node);
      } else {
        new_device_map[inode.source] = dev_id;
      }
    }
    // Make the new graph.
    for (const NodeEntry& e : src.outputs) {
      if (new_node_map[idx.node_id(e.node.get())] != nullptr) {
        ret.graph.outputs.emplace_back(
            NodeEntry{new_node_map[idx.node_id(e.node.get())], e.index, e.version});
      } else {
        ret.graph.outputs.emplace_back(e);
      }
    }
    // Make the device attribute.
    const IndexedGraph& new_idx = ret.graph.indexed_graph();
    DeviceVector new_device_vec(new_idx.num_nodes());
    for (uint32_t nid = 0; nid < new_idx.num_nodes(); ++nid) {
      auto source = new_idx[nid].source;
      if (new_device_map.count(source) == 0) {
        LOG(FATAL) << "canot find " << source;
      }
      new_device_vec[nid] = new_device_map.at(source);
    }
    ret.graph.SetNodeAttr("device", std::move(new_device_vec));
    return ret;
  }
};

NNVM_REGISTER_PASS_CLASS(PlaceDevicePass)
.describe("Infer the device type of each operator."\
          "Insert a copy node when there is cross device copy")
.set_change_graph(true)
.provide_node_attr("device")
.preserve_all(attr::kNode | attr::kNodeEntry)
;

DMLC_JSON_ENABLE_ANY(DeviceAssignMap, dict_str_int);

}  // namespace
}  // namespace pass
}  // namespace nnvm
