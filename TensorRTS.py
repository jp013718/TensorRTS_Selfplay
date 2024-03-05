import random
import abc
from typing import Dict, List, Mapping, Tuple, Set
from entity_gym.env import Observation

from entity_gym.runner import CliRunner
from entity_gym.env import *

class TensorRTS(Environment):
    """
LinearRTS, the first epoch of TensorRTS, is intended to be the simplest RTS game.
    """

    def __init__(
        self,
        mapsize: int = 32,
        nclusters: int = 6,
        ntensors: int = 2,
        maxdots: int = 9,
        attack_speed: int = 2,
        e_to_m: float = 0.2,
        boom_factor: float = 0.2,
        attack_adv: float = 0.8
    ):
        print(f"LinearRTS -- Mapsize: {mapsize}")
        self.mapsize = mapsize
        self.maxdots = maxdots
        self.nclusters = nclusters
        self.clusters: List[List[int]] = []  # The inner list has a size of 2 (position, number of dots).
        self.tensors: List[List[int ]] = [] # The inner list has a size of 4 (position, dimension, x, y).
        # Adjustable parameters for game balance. Default values are as given in PA4 on GitHub
        self.attack_speed = attack_speed
        self.e_to_m = e_to_m
        self.boom_factor = boom_factor
        self.attack_adv = attack_adv

    def obs_space(cls) -> ObsSpace:
        return ObsSpace(
            entities={
                "Cluster": Entity(features=["position", "dot"]),
                "Tensor": Entity(features=["position", "dimension", "x", "y"]),
            }
        )

    def action_space(cls) -> Dict[ActionName, ActionSpace]:
        return {
            "Move": GlobalCategoricalActionSpace(
                ["advance", "retreat", "rush", "boom"],
            ),
        }

    def reset(self) -> Observation:
        positions = set()
        while len(positions) < self.nclusters // 2:
            position, b = random.choice(
                [[position, b] for position in range(self.mapsize // 2) for b in range(1, self.maxdots)]
            )
            if position not in positions:
                positions.add(position)
                self.clusters.append([position, b])
                self.clusters.append([self.mapsize - position - 1, b])
        self.clusters.sort()
 
        position = random.randint(0, self.mapsize // 2)
        self.tensors = [[position, 1, 2, 0], [self.mapsize - position - 1, 1, 2, 0]]
        # Starting positions are added for TP calculation
        self.starts = (self.tensors[0][0], self.tensors[1][0])
        self.print_universe()

        return self.observe()
    
    def tensor_power(self, tensor_index) -> float :
        # A better tensor power calculation may be possible that doesn't depend heavily on whether the unit starts on the left or right
        if tensor_index == 0:
            f = self.tensors[tensor_index][3] * (1 + (self.tensors[tensor_index][0]-(self.starts[1]-self.starts[0])/2)/self.mapsize*self.attack_adv)
        else:
            f = self.tensors[tensor_index][3] * (1 + ((self.starts[1]-self.starts[0])/2-self.tensors[tensor_index][0])/self.mapsize*self.attack_adv)
        print(f"TP({tensor_index})=TP({self.tensors[tensor_index]})={f}")
        return f

    def observe(self) -> Observation:
        done = self.tensors[0][0] >= self.tensors[1][0]
        if done:
            reward = 10 if self.tensor_power(0) > self.tensor_power(1) else 0 if self.tensor_power(0) == self.tensor_power(1) else -10
        else:
            reward = 1.0 if self.tensors[0][1] > self.tensors[1][1] else 0.0
        return Observation(
            entities={
                "Cluster": (
                    self.clusters,
                    [("Cluster", i) for i in range(len(self.clusters))],
                ),
                "Tensor": (
                    self.tensors,
                    [("Tensor", i) for i in range(len(self.tensors))],
                ),
            },
            actions={
                "Move": GlobalCategoricalActionMask(),
            },
            done=done,
            reward=reward,
        )

    def act(self, actions: Mapping[ActionName, Action], trigger_default_opponent_action : bool = True) -> Observation:
        action = actions["Move"]
        assert isinstance(action, GlobalCategoricalAction)
        if action.label == "advance":
            # Move forward as many times as the attack speed is set to
            for _ in range(self.attack_speed):
              if self.tensors[0][0] < self.mapsize:
                self.tensors[0][0] += 1
                self.tensors[0][2] += self.collect_dots(self.tensors[0][0])
        elif action.label == "retreat" and self.tensors[0][0] > 0:
            self.tensors[0][0] -= 1
            self.tensors[0][2] += self.collect_dots(self.tensors[0][0])
        # Conversion rates are applied to boom and rush if the conversion rate is greater than 1
        elif action.label == "boom":
            if int(self.boom_factor * self.tensors[0][2]) > 1:
                self.tensors[0][2] += int(self.boom_factor * self.tensors[0][2])
            else:
                self.tensors[0][2] += 1
        elif action.label == "rush":
            if self.tensors[0][2] >= 1:
                if int(self.e_to_m * self.tensors[0][2]) > 1:
                    self.tensors[0][1] = 2
                    self.tensors[0][2] -= int(self.e_to_m * self.tensors[0][2])
                    self.tensors[0][3] += int(self.e_to_m * self.tensors[0][2])
                else:
                    self.tensors[0][1] = 2 # the number of dimensions is now 2
                    self.tensors[0][2] -= 1
                    self.tensors[0][3] += 1

        if trigger_default_opponent_action:
            self.opponent_act()
        
        self.print_universe()

        return self.observe()

    def opponent_act(self):         # This is the rush AI.
        self.advantage = [1, 1]
        if self.tensors[1][2]>0 :   # Rush if possile
            if int(self.e_to_m * self.tensors[1][2]) > 1:
                self.tensors[1][1] = 2
                self.tensors[1][2] -= int(self.e_to_m * self.tensors[1][2])
                self.tensors[1][3] += int(self.e_to_m * self.tensors[1][2])
            else:
                self.tensors[1][2] -= 1
                self.tensors[1][3] += 1
                self.tensors[1][1] = 2      # the number of dimensions is now 2
        else:                       # Otherwise Advance.
            self.advantage[0] = self.attack_adv
            for _ in range(self.attack_speed):
                if self.tensors[1][0] > 0:
                    self.tensors[1][0] -= 1
                    self.tensors[1][2] += self.collect_dots(self.tensors[1][0])

        return self.observe()

    def collect_dots(self, position):
        low, high = 0, len(self.clusters) - 1

        while low <= high:
            mid = (low + high) // 2
            current_value = self.clusters[mid][0]

            if current_value == position:
                dots = self.clusters[mid][1]
                self.clusters[mid][1] = 0
                return dots
            elif current_value < position:
                low = mid + 1
            else:
                high = mid - 1

        return 0        

    def print_universe(self):
        #    print(self.clusters)
        #    print(self.tensors)
        for j in range(self.mapsize):
            print(f" {j%10}", end="")
        print(" #")
        position_init = 0
        for i in range(len(self.clusters)):
            for j in range(position_init, self.clusters[i][0]):
                print("  ", end="")
            print(f" {self.clusters[i][1]}", end="")
            position_init = self.clusters[i][0]+1
        for j in range(position_init, self.mapsize):
            print("  ", end="")
        print(" ##")

        position_init = 0
        for i in range(len(self.tensors)):
            for j in range(position_init, self.tensors[i][0]):
                print("  ", end="")
            print(f"{self.tensors[i][2]}", end="")
            if self.tensors[i][3]>=0:
                print(f"-{self.tensors[i][3]}", end="")
            position_init = self.tensors[i][0]+1
        for j in range(position_init, self.mapsize):
            print("  ", end="")
        print(" ##")

class Interactive_TensorRTS(TensorRTS): 
    def __init__(self,
        mapsize: int = 32,
        nclusters: int = 6,
        ntensors: int = 2,
        maxdots: int = 9): 
        self.is_game_over = False

        super().__init__(mapsize, nclusters, ntensors, maxdots)

    def act(self, actions: Mapping[ActionName, Action],  trigger_default_opponent_action : bool = True) -> Observation:
        obs_result = super().act(actions, False)

        if (obs_result.done == True):
            self.is_game_over = True

        return obs_result

class Agent(metaclass=abc.ABCMeta):
    def __init__(self, initial_observation : Observation, action_space : Dict[ActionName, ActionSpace]):
        self.previous_game_state = initial_observation
        self.action_space = action_space

    @abc.abstractmethod
    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]: 
        """Pure virtual function in which an agent should return the move that they will make on this turn.

        Returns:
            str: name of the action that will be taken
        """
        pass

    @abc.abstractmethod
    def on_game_start(self) -> None: 
        """Function which is called for the agent before the game begins.
        """
        pass

    @abc.abstractmethod
    def on_game_over(self) -> None:
        """Function which is called for the agent once the game is over.

        Args:
            did_i_win (bool): set to True if this agent won the game.
        """
        pass

class Random_Agent(Agent):
    def __init__(self, init_observation : Observation, action_space : Dict[ActionName, ActionSpace]) -> None: 
        super().__init__(init_observation, action_space)

    def take_turn(self, current_game_state : Observation) -> Mapping[ActionName, Action]:
        mapping = {}

        action_choice = random.randrange(0, 2)
        if (action_choice == 1): 
            mapping['Move'] = GlobalCategoricalAction(0, self.action_space['Move'].index_to_label[0])
        else: 
            mapping["Move"] = GlobalCategoricalAction(1, self.action_space['Move'].index_to_label[1])
        
        return mapping
    
    def on_game_start(self) -> None:
        return super().on_game_start()
    
    def on_game_over(self) -> None:
        return super().on_game_over()

class GameRunner(): 
    def __init__(self, environment = None):
        self.game = Interactive_TensorRTS()

        self.player_one = None
        self.player_two = None

    def set_new_game(self) -> Observation: 
        """Resets the environment for a new game

        Returns:
            Observation: Initial observation for the new game
        """

        return self.game.reset()
    
    def assign_players(self, first_agent : Agent, second_agent : Agent = None):
        self.player_one = first_agent

        if second_agent is not None:
            self.player_two = second_agent

    def run(self): 
        assert(self.player_one is not None)

        game_state = self.game.observe()
        self.player_one.on_game_start()

        while(self.game.is_game_over is False):
            #take moves and pass updated environments to agents
            game_state = self.game.act(self.player_one.take_turn(game_state))
            
            if (self.game.is_game_over is False):
                if self.player_two is None: 
                    game_state = self.game.opponent_act()
                else:
                    #future player_two code
                    pass

        self.player_one.on_game_over()

# if __name__ == "__main__":  # This is to run wth agents
#     runner = GameRunner()
#     init_observation = runner.set_new_game()
#     random_agent = Random_Agent(init_observation, runner.game.action_space())

#     runner.assign_players(random_agent)
#     runner.run()
    
if __name__ == "__main__":  #this is to run cli
    env = TensorRTS()
    # The `CliRunner` can run any environment with a command line interface.
    CliRunner(env).run()    