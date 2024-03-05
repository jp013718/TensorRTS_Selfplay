import os

from TensorRTS import TensorRTS
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from enn_trainer import load_checkpoint, RogueNetAgent

class TensorRTS_SelfPlay(TensorRTS):
  def __init__(
      self,
      mapsize: int=32,
      nclusters: int = 6,
      ntensors: int = 2,
      maxdots: int = 9,
      checkpoint_dir: str = "checkpoints"):
    super().__init__(mapsize, nclusters, ntensors, maxdots)
    # Include the checkpoint directory to use for the opponent AI
    self.checkpoint_dir = checkpoint_dir

  """
  While training, the AI learns how to play from the left side. As we now want to use the trained AI as an opponent
  (playing from the right side), we need to trick the AI into thinking that it is on the left. We can do this by giving
  it the reward and done signals that it should be used to and by reversing the order of the game map given to it.
  """
  def opp_observe(self) -> Observation:
    done = self.tensors[1][0] >= self.tensors[0][0]
    if done:
      reward = 10 if self.tensor_power(1) > self.tensor_power(0) else 0 if self.tensor_power(1) == self.tensor_power(0) else -10
    else:
      reward = 1.0 if self.tensors[1][1] > self.tensors[0][1] else 0.0
    
    opp_clusters = [[self.mapsize - i - 1, j] for i,j in self.clusters]
    for cluster in opp_clusters:
      cluster[0] = self.mapsize - cluster[0] - 1
    opp_tensors = [[self.mapsize - i - 1, j, k, l] for i,j,k,l in self.tensors]
    for tensor in opp_tensors:
      tensor[0] = self.mapsize - tensor[0] - 1

    return Observation(
      entities={
        "Cluster": (
          opp_clusters,
          [("Cluster", i) for i in range(len(opp_clusters))]
        ),
        "Tensor": (
          opp_tensors,
          [("Tensor", i) for i in range(len(opp_tensors))]
        )
      },
      actions={
        "Move": GlobalCategoricalActionMask(),
      },
      done=done,
      reward=reward,
    )
  
  """
  The new reset needs to load in the opponent AI before calling the inherited reset.
  Currently, we just load in the first AI that appears in the checkpoint directory. This
  isn't the best option, but the ENN libraries overwrite old checkpoints, anyway. If this
  feature of the ENN libraries is ever changed, a better option could be found.
  """
  def reset(self) -> Observation:
    checkpoint = load_checkpoint(os.path.join(self.checkpoint_dir, os.listdir(self.checkpoint_dir)[0]))
    self.opp_ai = RogueNetAgent(checkpoint.state.agent)
    return super().reset()

  """
  The overridden opponent_act function implements all four possible actions and queries the 
  opponent AI for the best action.
  """
  def opponent_act(self):
    action, predicted_return = self.opp_ai.act(self.opp_observe())
    print(action)
    action = action["Move"]
    assert isinstance(action, GlobalCategoricalAction)

    if action.label == "advance":
      for _ in range(self.attack_speed):
        if self.tensors[1][0] > 0:
          self.tensors[1][0] -= 1
          self.tensors[1][2] += self.collect_dots(self.tensors[1][0])
    elif action.label == "retreat" and self.tensors[1][0] < self.mapsize:
      self.tensors[1][0] += 1
      self.tensors[1][2] += self.collect_dots(self.tensors[1][0])
    elif action.label == "boom":
      if int(self.boom_factor * self.tensors[1][2]) > 1:
        self.tensors[1][2] += int(self.boom_factor * self.tensors[1][2])
      else:
        self.tensors[1][2] += 1
    elif action.label == "rush":
      if self.tensors[1][2] >= 1:
        if int(self.e_to_m * self.tensors[1][2]) > 1:
          self.tensors[1][1] = 2
          self.tensors[1][2] -= int(self.e_to_m * self.tensors[1][2])
          self.tensors[1][3] += int(self.e_to_m * self.tensors[1][2])
        else:
          self.tensors[1][1] = 2
          self.tensors[1][2] -= 1
          self.tensors[1][3] += 1

    return self.observe()
  
if __name__ == "__main__":
  env = TensorRTS_SelfPlay()
  CliRunner(env).run()
