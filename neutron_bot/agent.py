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
    self.win_ratio = 0
    self.games = 0

  def take_turn(self, current_game_state: Observation) -> Mapping[ActionName, Action]:
    """
    GameRunner seems to flip the observation for player two, so the flipping shouldn't be
    necessary anymore
    """
    # if self.is_player_one:
    action, predicted_return = self.agent.act(current_game_state)
    return action
    # elif self.is_player_two:
    #   entities = current_game_state.features
    #   actions = current_game_state.actions
    #   done = current_game_state.done
    #   reward = current_game_state.reward

    #   clusters = entities["Cluster"]
    #   tensors = entities["Tensor"]

    #   opp_clusters = [[32-i-1, j] for i, j in clusters]
    #   opp_tensors = [[32-i-1, j, k, l] for i, j, k, l in tensors]

    #   opp_obs = Observation(
    #     entities={
    #       "Cluster": (
    #         opp_clusters,
    #         [("Cluster", i) for i in range(len(opp_clusters))]
    #       ),
    #       "Tensor": (
    #         opp_tensors,
    #         [("Tensor", i) for i in range(len(opp_tensors))]
    #       )
    #     },
    #     actions=actions,
    #     done=done,
    #     reward=reward
    #   )

    #   action, predicted_return = self.agent.act(opp_obs)
    #   return action
  
  def on_game_start(self, is_player_one:bool, is_player_two:bool) -> None:
    # Load from a checkpoint on game start. This should allow training on selfplay
    # using the TensorRTS_Selfplay class
    checkpoint = load_checkpoint(os.path.join(__file__.strip("agent.py"), bot_dir))
    self.agent = RogueNetAgent(checkpoint.state.agent)
    return super().on_game_start(is_player_one, is_player_two)
  
  def on_game_over(self, did_i_win: bool, did_i_tie: bool) -> None:
    self.games += 1
    if did_i_win:
      self.win_ratio = (self.win_ratio*(self.games-1)+1)/self.games
    return super().on_game_over(did_i_win, did_i_tie)
  
def agent_hook(init_observation: Observation, action_space: Dict[ActionName, ActionSpace]) -> Agent:
  return NeutronBot(init_observation, action_space)
  
def student_name_hook() -> str:
  return "James Pennington"

if __name__ == "__main__":
  print("\n---Testing Neutron Bot---\n")
  from TensorRTS import GameRunner

  runner = GameRunner(enable_printouts=True, trace_file="NeutronTest.txt")
  init_observation = runner.get_game_observation(is_player_two=False)
  neutron_bot = NeutronBot(init_observation, runner.game.action_space())

  runner.assign_players(neutron_bot)
  runner.run(max_game_turns=300)
