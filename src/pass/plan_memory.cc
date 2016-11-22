/*!
 *  Copyright (c) 2016 by Contributors
 * \file plan_memory.cc
 * \brief Assign memory tag to each of the data entries.
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/pass_functions.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <memory>
#include "./graph_algorithm.h"

namespace nnvm {
namespace pass {
namespace {

// simple graph based allocator.
class GraphAllocator {
 public:
  // storage id equals integer.
  using StorageID = int;
  // bad storage id
  static const StorageID kBadStorageID = -1;
  // request a free storage
  StorageID Request(int dev_id, int dtype, TShape shape, uint32_t node_id) {
    if (shape.ndim() == 0) return kBadStorageID;
    // search memory block in [size / match_range_, size * match_range_)
    // TODO(tqchen) add size of the dtype, assume 4 bytes for now
    size_t size = shape.Size() * 4;
    if (match_range_ == 0) return this->Alloc(dev_id, size);
    auto begin = free_.lower_bound(size / match_range_);
    auto mid = free_.lower_bound(size);
    auto end = free_.upper_bound(size * match_range_);
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;
      // Use exect matching strategy
      e->max_bytes = std::max(size, e->max_bytes);
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // cannot find anything return a new one.
    return this->Alloc(dev_id, size);
  }
  // release a memory space.
  void Release(StorageID id, uint32_t node_id) {
    CHECK_NE(id, kBadStorageID);
    StorageEntry *e = data_[id].get();
    e->released_by_node = node_id;
    free_.insert({e->max_bytes, e});
  }
  // totoal number of bytes allocated
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (auto &p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // constructor
  explicit GraphAllocator(const IndexedGraph* idx) : idx_(idx) {
    this->Init(dmlc::GetEnv("NNVM_EXEC_MATCH_RANGE", 16),
               dmlc::GetEnv("NNVM_EXEC_NUM_TEMP", 1));
  }

 private:
  // initialize the graph allocator
  void Init(size_t match_range, uint32_t num_match_color) {
    match_range_ = match_range;
    num_match_color_ = num_match_color;
    if (num_match_color_ > 1) {
      std::vector<uint32_t> importance(idx_->num_nodes(), 0);
      for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
        if ((*idx_)[nid].source->is_variable()) continue;
        importance[nid] = 1;
      }
      num_match_color_ = pass::ColorNodeGroup(
          *idx_, importance, num_match_color_, &node_color_);
    }
  }

  StorageID Alloc(int dev_id, size_t size) {
    StorageID id = static_cast<StorageID>(data_.size());
    std::unique_ptr<StorageEntry> ptr(new StorageEntry());
    ptr->id = id;
    ptr->device_id = dev_id;
    ptr->max_bytes = size;
    data_.emplace_back(std::move(ptr));
    return id;
  }
  // internal storage entry
  struct StorageEntry {
    // the id of the entry.
    StorageID id;
    // the device id of the storage.
    int device_id;
    // maximum size of storage requested.
    size_t max_bytes{0};
    // node index that released it last time
    uint32_t released_by_node{0};
  };
  // scale used for rough match
  size_t match_range_;
  // whether use color based match algorithm
  uint32_t num_match_color_{1};
  // the size of each dtype
  std::vector<size_t> dtype_size_dict_;
  // free list of storage entry
  std::multimap<size_t, StorageEntry*> free_;
  // all the storage resources available
  std::vector<std::unique_ptr<StorageEntry> > data_;
  // color of nodes in the graph, used for auxiliary policy making.
  std::vector<uint32_t> node_color_;
  // internal indexed graph
  const IndexedGraph* idx_;
};

// function to plan memory
class PlanMemoryPass : public Pass {
 public:
  PassResult RunOnGraph(Graph src, const PassArgument& pargs) override {
    PassResult ret;
    ret.graph = src;
    // setup ref counter
    const IndexedGraph& idx = src.indexed_graph();
    // reference counter of each node
    std::vector<uint32_t> ref_count(idx.num_node_entries(), 0);
    // step 1: initialize reference count
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      for (const auto& e : idx[nid].inputs) {
        ++ref_count[idx.entry_id(e)];
      }
    }
    for (const auto& e : idx.outputs()) {
      ++ref_count[idx.entry_id(e)];
    }
    // step 2: allocate memory.
    StorageVector storage(idx.num_node_entries(), GraphAllocator::kBadStorageID);
    // TODO(minjie): define this constant.
    std::vector<int> storage_inplace_index(idx.num_node_entries(), -1);
    static auto& finplace_option = Op::GetAttr<FInplaceOption>("FInplaceOption");

    // The memory allocation algorithm.
    GraphAllocator allocator(&idx);
    // Number of entries that are not statically allocated.
    size_t num_not_allocated = 0;

    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const auto& inode = idx[nid];
      if (inode.source->is_variable()) {
        // No storage planning for variable nodes since they should be pre-allocated.
        continue;
      }
      // Check inplace option.
      if (finplace_option.count(inode.source->op()) != 0) {
        auto inplace_pairs = finplace_option[inode.source->op()](inode.source->attrs);
        for (const auto& kv : inplace_pairs) {
          const uint32_t eid_out = idx.entry_id(nid, kv.second);
          const uint32_t eid_in = idx.entry_id(inode.inputs[kv.first]);
          const TShape& out_shape = src.GetNodeEntryAttr<TShape>("shape", eid_out);
          const TShape& in_shape = src.GetNodeEntryAttr<TShape>("shape", eid_in);
          const int out_dtype = src.GetNodeEntryAttr<int>("dtype", eid_out);
          const int in_dtype = src.GetNodeEntryAttr<int>("dtype", eid_in);
          if (ref_count[eid_in] == 1 &&
              ref_count[eid_out] != 0 &&
              storage[eid_out] == GraphAllocator::kBadStorageID &&
              storage[eid_in] != GraphAllocator::kBadStorageID &&
              out_shape.Size() == in_shape.Size() &&
              out_dtype == in_dtype) {
            // inplace optimization
            storage[eid_out] = storage[eid_in];
            ref_count[eid_in] = 0;
            storage_inplace_index[eid_out] = kv.first;
          }
        }
      }
      // normal allocation
      const int dev_id = src.GetNodeAttr<int>("device", nid);
      // allocate output
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        const uint32_t eid = idx.entry_id(nid, index);
        const TShape& shape = src.GetNodeEntryAttr<TShape>("shape", eid);
        const int dtype = src.GetNodeEntryAttr<int>("dtype", eid);
        if (storage[eid] == GraphAllocator::kBadStorageID) {
          storage[eid] = allocator.Request(dev_id, dtype, shape, nid);
        }
      }
      // then free inputs
      for (const auto& e : inode.inputs) {
        const uint32_t eid = idx.entry_id(e);
        // temp_ref_count == 0 means it is taken by inplace op
        if (ref_count[eid] == 0) continue;
        // if we decrease it to zero, means we are ready to relase
        --ref_count[eid];
        if (ref_count[eid] == 0 && storage[eid] != GraphAllocator::kBadStorageID) {
          allocator.Release(storage[eid], nid);
        }
      }
      // check if there are outputs that can be freeded immediately
      // these output are not referenced by any operator.
      for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
        const uint32_t eid = idx.entry_id(nid, index);
        if (ref_count[eid] == 0 && storage[eid] != GraphAllocator::kBadStorageID) {
          allocator.Release(storage[eid], nid);
          // use -2 to indicate that the node was never touched.
          // TODO(minjie): define this constant.
          storage_inplace_index[eid] = -2;
        }
        if (storage[eid] == GraphAllocator::kBadStorageID) {
          ++num_not_allocated;
        }
      }
    }
    ret.graph.SetNodeEntryAttr("storage_id", std::move(storage));
    ret.graph.SetNodeEntryAttr("storage_inplace_index", std::move(storage_inplace_index));
    ret.graph.SetGraphAttr("storage_allocated_bytes", allocator.TotalAllocBytes());
    ret.graph.SetGraphAttr("storage_num_not_allocated", num_not_allocated);
    return ret;
  }
};

NNVM_REGISTER_PASS_CLASS(PlanMemoryPass)
.describe("Plan the memory allocation of each node entries.")
.set_change_graph(false)
.depend_op_attr("FInplaceOption")
.depend_entry_attr("dtype")
.depend_entry_attr("shape")
.depend_node_attr("device")
.provide_entry_attr("storage_id")
.provide_entry_attr("storage_inplace_index")
.provide_graph_attr("storage_allocated_bytes")
.provide_graph_attr("storage_num_not_allocated")
.preserve_all();

}  // namespace
}  // namespace pass
}  // namespace nnvm
