import os
from typing import Dict, List, Mapping, Tuple, Set

from entity_gym.env import *
from enn_trainer import TrainConfig, State, init_train_state, train, load_checkpoint, RogueNetAgent

from TensorRTS_Selfplay import TensorRTS_SelfPlay

import hyperstate

checkpoint = load_checkpoint(os.path.join('checkpoints', os.listdir('checkpoints')[0]))
agent = RogueNetAgent(checkpoint.state.agent)

@hyperstate.stateful_command(TrainConfig, State, init_train_state)
def main(state_manager: hyperstate.StateManager) -> None:
    train(state_manager=state_manager, env=TensorRTS_SelfPlay, agent=agent)

if __name__ == "__main__":  # This is to train
    main()