/*!
 *  Copyright (c) 2016 by Contributors
 * \file pass.cc
 * \brief Support for pass registry.
 */
#include <nnvm/pass.h>
#include <algorithm>

namespace dmlc {
// enable registry
DMLC_REGISTRY_ENABLE(nnvm::PassReg);
}  // namespace dmlc

namespace nnvm {
/*const PassReg* FindPassDep(const std::string&attr_name) {
  for (auto* r : dmlc::Registry<PassReg>::List()) {
    for (auto& s : r->graph_attr_targets) {
      if (s == attr_name) return r;
    }
  }
  return nullptr;
}

Graph ApplyPasses(Graph g,
                  const std::vector<std::string>& pass) {
  std::vector<const PassReg*> fpass;
  for (auto& name : pass) {
    auto* reg = dmlc::Registry<PassReg>::Find(name);
    CHECK(reg != nullptr)
        << "Cannot find pass " << name << " in the registry";
    fpass.push_back(reg);
  }

  for (auto r : fpass) {
    for (auto& dep : r->graph_attr_dependency) {
      if (g.attrs.count(dep) == 0) {
        auto* pass_dep = FindPassDep(dep);
        std::string msg;
        if (pass_dep != nullptr) {
          msg = " The attribute is provided by pass " + pass_dep->name;
        }
        LOG(FATAL) << "Graph attr dependency " << dep
                   << " is required by pass " << r->name
                   << " but is not available "
                   << msg;
      }
    }
    g = r->body(std::move(g));
  }

  return g;
}*/

Graph ApplyPasses(Graph src,
                  const std::vector<std::string>& passes,
                  const std::vector<std::shared_ptr<any>>& args) {
  CHECK_EQ(passes.size(), args.size())
    << "The number of passes and arguments should be the same";
  std::unique_ptr<PassManager> pm = PassManager::Create();
  pm->Enable(passes);
  for (size_t i = 0; i < passes.size(); ++i) {
    pm->SetPassArguments(passes[i], args[i]);
  }
  return pm->Run(src).graph;
}

}  // namespace nnvm
