# TensorRTS_Selfplay
An implementation of TensorRTS from PA4 in the RL4SE repo from drchangliu

# Setting Up:
After cloning the repo, make sure to run on Python 3.8. This can be done with the following command if conda is installed:

`conda create -n ENN python=3.8 \n
conda activate ENN`

Once the environment is activated, install entity-gym and enn-trainer following the steps [here](https://github.com/entity-neural-network/entity-gym) and [here](https://github.com/entity-neural-network/enn-trainer).

# Playing
In order to play TensorRTS, you can run the following command to play against a built-in Rush AI:

`python TensorRTS.py`

If there is a checkpoint saved in the directory `checkpoints`, you can also run the following command to play against a trained AI:

`python TensorRTS_Selfplay.py`

# Training:
The initial round of training will be performed against a built-in Rush AI. The length of the training can be changed by editing the file `config.ron`. In order to train against this AI, run the following command:

`python train.py --config=config.ron --checkpoint-dir=checkpoints`

As you will see, checkpoints are automatically overwritten by the hyperstate library. As of the writing of this, hyperstate has no ability to save persistent checkpoints. To work around this for selfplay, we will save subsequent checkpoints in a different directory and manually move new checkpoints in between rounds of training. To run a new round of training against the previously trained AI, we can now run the following:

`python train_selfplay.py --config=config.ron --checkpoint-dir=selfplay_checkpoints`

This script by default loads in the first checkpoint found in the directory `checkpoints` to be used as the opponent. To ensure the most recently trained agent is used for the opponent, it is best to leave only the desired checkpoint in that directory. If you wish to save previous checkpoints, they can safely be stored in any directory other than the directory set with the `--checkpoint-dir` tag.
