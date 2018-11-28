#include "dead_code_elimination.h"

#include "torch/csrc/jit/passes/alias_analysis.h"

#include <unordered_map>

namespace torch {
namespace jit {

class DeadCodeEliminator {
 public:
  // If given a top-level graph, DCE will construct an aliasDb that allows for
  // "smarter" dead code elimination (we will eliminate mutable ops if we can
  // prove the mutated values are not used).
  //
  // Otherwise, we will not allow DCE to eliminate mutable ops.
  explicit DeadCodeEliminator(std::shared_ptr<Graph> graph)
      : aliasDb_(AliasAnalysis(graph)) {}
  DeadCodeEliminator(){};


  // The algorithm is an inverse mark-and-sweep. Starting from the return node,
  // we mark "live" nodes based on
  void runMarkSweep(Block* block, bool recurse) {
    // Add return node to work list
    addToWorkList(block->return_node());

    // Mark all nodes with side effects as well.
    for (auto node : block->nodes()) {
      if (hasSideEffects(node)) {
        addToWorkList(node);
      }
    }

    while (!workList_.empty()) {
      auto node = workList_.front();
      workList_.pop_front();

      // Mark this node
      marked_.insert(node);

      for (auto subBlock : node->blocks()) {
        runMarkSweep(subBlock, recurse);
      }

      // If this node is in a sub-block of `block`, traverse the blockchain
      // upwards until we find the containing node in `block`.
      if (node->owningBlock() != block) {
        auto topLevelNode = node;
        while (topLevelNode) {
          if (!topLevelNode->owningBlock()) {
            break;
          }

          if (topLevelNode->owningBlock() == block) {
            addToWorkList(topLevelNode);
            break;
          }
          topLevelNode = topLevelNode->owningBlock()->owningNode();
        }
      }

      // Find preceding writers for node, add to work list
      if (aliasDb_) {
        for (auto writer : aliasDb_->getWritersForNode(node)) {
          if (writer->isBefore(node)) {
            addToWorkList(writer);
          }
        }
      }

      // Find producers for all inputs, add to work list
      for (auto input : node->inputs()) {
        addToWorkList(input->node());
      }
    }



    // Sweep:
    // In reverse order:
    //   If the node is not marked, delete it.
    auto nodes = block->nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      // note these occur before the recursion because we want to uncover
      // dead code in the blocks used to calculate the output
      removeDeadIfOutputs(node);
      removeDeadLoopOutputs(node);
      if (recurse) {
        for (Block* block : node->blocks())
          run(block, true);
      }
      if (!marked_.count(node)) {
        it.destroyCurrent();
      }
    }
  }



  void run(Block* block, bool recurse) {
    auto nodes = block->nodes().reverse();
    for (auto it = nodes.begin(); it != nodes.end(); it++) {
      auto node = *it;
      // note these occur before the recursion because we want to uncover
      // dead code in the blocks used to calculate the output
      removeDeadIfOutputs(node);
      removeDeadLoopOutputs(node);
      if (recurse) {
        for (Block* block : node->blocks())
          run(block, true);
      }
      if (!node->hasUses() && !hasSideEffects(node))
        it.destroyCurrent();
    }
  }

 private:
  std::unordered_set<Node*> marked_;
  void addToWorkList(Node* n) {
    if (!marked_.count(n)) {
      workList_.push_back(n);
    }
  }

  std::list<Node*> workList_;

  bool isMutable(Node* node) {
    if (!node->kind().is_aten())
      return false;
    // onnx export calls EliminateDeadCode but sometimes passes invalid
    // aten operators. So we call maybeSchema so we handle the cases when
    // there is no valid schema for a node
    auto schema = node->maybeSchema();
    return schema && schema->is_mutable();
  }

  bool hasSideEffects(Node* node) {
    // FIXME: PythonOp should be treated as having side effects as well!
    //        Unfortunately ONNX depends on it getting removed in this pass, so
    //        it's not a simple change.
    auto it = memo_.find(node);
    if (it != memo_.end())
      return it->second;
    bool has_side_effects = node->kind() == prim::Print ||
        node->kind() == prim::RaiseException ||
        std::any_of(node->blocks().begin(),
                    node->blocks().end(),
                    [&](Block* b) {
                      return std::any_of(
                          b->nodes().begin(), b->nodes().end(), [&](Node* n) {
                            return hasSideEffects(n);
                          });
                    });

    if (!aliasDb_) {
      // If we don't have aliasing information, we have to disallow DCE for all
      // mutable ops, since we're not sure what they affect.
      has_side_effects |= isMutable(node);
    }

    memo_.emplace(node, has_side_effects);
    return has_side_effects;
  }

  void removeDeadIfOutputs(Node* node) {
    if (node->kind() != prim::If)
      return;
    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses()) {
        node->eraseOutput(i);
        for (Block* b : node->blocks()) {
          b->eraseOutput(i);
        }
      }
    }
  }

  void removeDeadLoopOutputs(Node* node) {
    if (node->kind() != prim::Loop)
      return;
    auto loop_body = node->blocks().at(0);
    auto loop_input_offset = 2; // offset of loop carried deps in input list
    auto loop_body_offset =
        1; // offset to the loop carried dependencies in block inputs/outputs

    for (size_t i_1 = node->outputs().size(); i_1 > 0; --i_1) {
      size_t i = i_1 - 1;
      if (!node->outputs().at(i)->hasUses() &&
          !loop_body->inputs().at(loop_body_offset + i)->hasUses()) {
        node->eraseOutput(i);
        node->removeInput(loop_input_offset + i);
        loop_body->eraseInput(loop_body_offset + i);
        loop_body->eraseOutput(loop_body_offset + i);
      }
    }
  }

  c10::optional<AliasDb> aliasDb_;
  std::unordered_map<Node*, bool> memo_;
};

void EliminateDeadCode(const std::shared_ptr<Graph>& graph) {
  DeadCodeEliminator(graph).runMarkSweep(graph->block(), true);
}

void EliminateDeadCode(Block* block, bool recurse) {
  DeadCodeEliminator().runMarkSweep(block, recurse);
}

} // namespace jit
} // namespace torch
