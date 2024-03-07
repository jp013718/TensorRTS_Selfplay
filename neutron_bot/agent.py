import os

from TensorRTS import Agent
from typing import Dict, Mapping
from entity_gym.env import Observation
from entity_gym.runner import CliRunner
from entity_gym.env import *
from enn_trainer import load_checkpoint, RogueNetAgent

class NeutronBot(Agent):
  def __init__(self, initial_observation: Observation, action_space: Dict[ActionName, ActionSpace], bot_dir="neutron_checkpoint"):
    super().__init__(initial_observation, action_space)
    checkpoint = load_checkpoint(os.path.join(os.getcwd(), "bots/neutron_bot", bot_dir))
    self.agent = RogueNetAgent(checkpoint.state.agent)
    self.kd_ratio = 0
    self.games = 0

  def take_turn(self, current_game_state: Observation) -> Mapping[ActionName, Action]:
    if self.is_player_one:
      action, predicted_return = self.agent.act(current_game_state)
      return action
    elif self.is_player_two:
      entities = current_game_state.entities
      actions = current_game_state.actions
      done = current_game_state.done
      reward = current_game_state.reward

      clusters = entities["Cluster"]
      tensors = entities["Tensor"]

      opp_clusters = [[32-i-1, j] for i, j in clusters]
      opp_tensors = [[32-i-1, j, k, l] for i, j, k, l in tensors]

      opp_obs = Observation(
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
        actions=actions,
        done=done,
        reward=reward
      )

      action, predicted_return = self.agent.act(current_game_state)
  
  def on_game_start(self) -> None:
    return super().on_game_start()
  
  def on_game_over(self, did_i_win: bool, did_i_tie: bool) -> None:
    self.games += 1
    if did_i_win:
      self.kd_ratio = (self.kd_ratio*(self.games-1)+1)/self.games
    return super().on_game_over(did_i_win, did_i_tie)
  
def agent_hook(init_observation: Observation, action_space: Dict[ActionName, ActionSpace]) -> Agent:
  return NeutronBot(init_observation, action_space)
  
def student_name_hook() -> str:
  return "James Pennington"