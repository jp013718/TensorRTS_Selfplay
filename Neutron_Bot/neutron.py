from TensorRTS import Agent
from typing import Dict, Mapping
from entity_gym.env import *
from enn_trainer import load_checkpoint, RogueNetAgent

class NeutronBot(Agent):
  def __init__(self, initial_observation: Observation, action_space: Dict[ActionName, ActionSpace], bot_dir="neutron_checkpoint"):
    super().__init__(initial_observation, action_space)
    checkpoint = load_checkpoint(bot_dir)
    self.agent = RogueNetAgent(checkpoint.state.agent)
    self.kd_ratio = 0

  def take_turn(self, current_game_state: Observation) -> Mapping[ActionName, Action]:
    action, predicted_return = self.agent.act(current_game_state)
    return action
    