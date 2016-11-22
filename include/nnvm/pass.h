/*!
 *  Copyright (c) 2016 by Contributors
 * \file pass.h
 * \brief Pass that can be applied to a graph.
 */
#ifndef NNVM_PASS_H_
#define NNVM_PASS_H_

#include <vector>
#include <functional>
#include <memory>
#include "./base.h"
#include "./graph.h"

namespace nnvm {

/*!
 * \brief Apply a series of pass transformations on the input graph.
 * \param src The graph to be transformed.
 * \param passes A list of pass names to be applied.
 * \return The transformed graph
 */
//Graph ApplyPasses(Graph src,
                  //const std::vector<std::string>& passes);

/*!
 * \brief Apply one pass to the graph.
 * \param src The graph to be transformed.
 * \param pass The name of pass to be applied.
 * \return The transformed graph.
 */
//inline Graph ApplyPass(Graph src, const std::string& pass) {
  //return ApplyPasses(src, {pass});
//}

struct PassResult {
  Graph graph;
};

struct PassArgument {
  std::shared_ptr<any> value;
};

/*!
 * \brief A PassCreator is an "Operator on Graph".
 *  It takes a source graph and return a graph that may or may
 *  not be the same as the input one.
 *
 *  A pass function can either change the graph structure (thus,
 *  generating a new Graph), or add new attributes to the graph.
 *
 * \param src The graph to be transformed.
 * \return The generated graph.
 */
class Pass {
 public:
  virtual void Setup() {}
  virtual PassResult RunOnGraph(Graph src, const PassArgument& args) = 0;
  virtual void Finalize() {}
};

/*! brief Function type for creating a pass */
typedef std::function<std::unique_ptr<Pass> ()> PassCreator;

/*!
 * \brief Registry entry for DataIterator factory functions.
 */
struct PassReg
    : public dmlc::FunctionRegEntryBase<PassReg, PassCreator> {
  /*!
   * \brief Whether the pass will change graph structure
   *  If this is false, the pass will only change attributes.
   */
  bool change_graph{false};
  /*! \brief Pass dependencies. */
  std::vector<std::string> pass_dependency;
  /*! \brief Operator attributes that the pass depends on. */
  std::vector<std::string> op_attr_dependency;
  /*! \brief Graph attributes that the pass depends on. */
  std::vector<std::string> graph_attr_dependency;
  /*! \brief Graph attributes that the pass will provide. */
  std::vector<std::string> graph_attr_targets;
  /*! \brief Graph attributes that the pass preserves. */
  std::vector<std::string> graph_attr_preserved;
  /*!
   * \brief Flags for which attribute categories are preserved.
   *  Example:
   *    .preserve_all(attr::kNode | attr::kNodeEntry);
   *    // this will preserve all node and entry attributes.
   */
  int preserve_flag = 0;
  /*!
   * \brief Declare this pass requires certain pass to be scheduled ahead.
   * \param pass_name Name of the pass.
   * \return Reference to self.
   */
  PassReg& depend_pass(const std::string& pass_name) {
    pass_dependency.push_back(pass_name);
    return *this;
  }
  /*!
   * \brief Set whether this pass will change graph structure.
   * \param v If true, the pass will change graph structure.
   * \return Reference to self.
   */
  PassReg& set_change_graph(bool v) {  // NOLINT(*)
    change_graph = v;
    return *this;
  }
  /*!
   * \brief Declare this pass requires the given operator attribute to be 
   *        available before being applied on the graph.
   * \param attr_name Name of the attribute.
   * \return Reference to self.
   */
  PassReg& depend_op_attr(const std::string& attr_name) {  // NOLINT(*)
    op_attr_dependency.push_back(attr_name);
    return *this;
  }
  /*!
   * \brief Declare this pass requires the given graph attribute to be
   *        available before being applied on the graph.
   * \param attr_name Name of the attribute.
   * \return Reference to self.
   */
  PassReg& depend_graph_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_dependency.push_back(attr_name);
    return *this;
  }
  /*!
   * \brief Declare that this pass will generate the given graph attribute name
   *        once it is applied on the graph.
   * \param attr_name Name of the graph attribute.
   * \return Reference to self.
   */
  PassReg& provide_graph_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_targets.push_back(attr_name);
    return *this;
  }
  /*!
   * \brief Declare that this pass will preserve the graph attributes.
   * \param attr_name Name of the graph attribute.
   * \return Reference to self.
   */
  PassReg& preserve_graph_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_preserved.push_back(attr_name);
    return *this;
  }
  /*!
   * \brief Declare this pass requires the given node attribute to be
   *        available before being applied on the graph.
   * \param attr_name Name of the attribute.
   * \return Reference to self.
   */
  PassReg& depend_node_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_dependency.push_back(Graph::kNodeAttrPrefix + attr_name);
    return *this;
  }
  /*!
   * \brief Declare that this pass will generate the given node attribute name
   *        once it is applied on the graph.
   * \param attr_name Name of the graph attribute.
   * \return Reference to self.
   */
  PassReg& provide_node_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_targets.push_back(Graph::kNodeAttrPrefix + attr_name);
    return *this;
  }
  /*!
   * \brief Declare that this pass will preserve the node attribute.
   * \param attr_name Name of the graph attribute.
   * \return Reference to self.
   */
  PassReg& preserve_node_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_preserved.push_back(Graph::kNodeAttrPrefix + attr_name);
    return *this;
  }
  /*!
   * \brief Declare this pass requires the given entry attribute to be
   *        available before being applied on the graph.
   * \param attr_name Name of the attribute.
   * \return Reference to self.
   */
  PassReg& depend_entry_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_dependency.push_back(Graph::kNodeEntryAttrPrefix + attr_name);
    return *this;
  }
  /*!
   * \brief Declare that this pass will generate the given node attribute name
   *        once it is applied on the graph.
   * \param attr_name Name of the graph attribute.
   * \return Reference to self.
   */
  PassReg& provide_entry_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_targets.push_back(Graph::kNodeEntryAttrPrefix + attr_name);
    return *this;
  }
  /*!
   * \brief Declare that this pass will preserve the node attribute.
   * \param attr_name Name of the graph attribute.
   * \return Reference to self.
   */
  PassReg& preserve_entry_attr(const std::string& attr_name) {  // NOLINT(*)
    graph_attr_preserved.push_back(Graph::kNodeEntryAttrPrefix + attr_name);
    return *this;
  }
  /*!
   * \brief Declare that this pass will preserve all the attributes in the
   *        given categories.
   * \return Reference to self.
   */
  PassReg& preserve_all(int flag) {
    preserve_flag = flag;
    return *this;
  }
  /*!
   * \brief Declare that this pass will preserve all attributes.
   */
  PassReg& preserve_all() {
    preserve_flag = attr::kGraph | attr::kNode | attr::kNodeEntry;
    return *this;
  }
};

/*!
 * \def NNVM_REGISTER_PASS
 * \brief Macro to register pass class.
 *
 * \code
 * // example of registering a shape inference pass
 * NNVM_REGISTER_PASS(InferShape)
 * .describe("Shape Inference function, generate graph attributes")
 * .set_body( ... )
 * .provide_graph_attr("data_shape")
 * .depend_graph_attr("indexed_graph")
 * .depend_op_attr("infer_shape");
 * \endcode
 */
#define NNVM_REGISTER_PASS(name)                         \
  DMLC_REGISTRY_REGISTER(::nnvm::PassReg, PassReg, name)

#define NNVM_REGISTER_PASS_CLASS(clsname)                                       \
  DMLC_REGISTRY_REGISTER(::nnvm::PassReg, PassReg, clsname) \
    .set_body([]() { return std::unique_ptr<clsname>(new clsname); })

class PassManager {
 public:
  PassManager& Enable(const std::string& pass) {
    return this->Enable(std::vector<std::string>({pass}));
  }

  PassManager& Enable(const std::vector<std::string>& passes) {
    enabled_passes_.insert(enabled_passes_.end(), passes.begin(), passes.end());
    return *this;
  }

  PassManager& SetPassArguments(const std::string& pass,
                                const std::shared_ptr<any>& arg_value) {
    CHECK(pass_arguments_.find(pass) == pass_arguments_.end())
      << "Arguments for pass \"" << pass << "\" have already been set.";
    pass_arguments_[pass] = arg_value;
    return *this;
  }

  PassResult Run(Graph src);

  static std::unique_ptr<PassManager> Create();

 private:
  PassManager();

  std::vector<std::string> enabled_passes_;
  std::unordered_map<std::string, std::shared_ptr<any>> pass_arguments_;

  std::vector<std::unique_ptr<Pass>> ordered_passes_;
};

}  // namespace nnvm

#endif  // NNVM_PASS_H_
