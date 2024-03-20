# TensorRTS_Selfplay
An implementation of TensorRTS from PA4 in the RL4SE repo from drchangliu

# Setting Up
After cloning the repo, make sure to run on Python 3.8. This can be done with the following command if conda is installed:
```
conda create -n ENN python=3.8
conda activate ENN
```
Once the environment is activated, install entity-gym and enn-trainer following the steps [here](https://github.com/entity-neural-network/entity-gym) and [here](https://github.com/entity-neural-network/enn-trainer).

# Playing
In order to play TensorRTS, you can run the following command to play against a built-in Rush AI:
```
python TensorRTS.py
```
If there is a checkpoint saved in the directory `checkpoints`, you can also run the following command to play against a trained AI:
```
python TensorRTS_Selfplay.py
```
# Training (Important: Read all of this before beginning training)
Before training can begin, you will first need to update the file `config.ron`. The contents of this file will look as follows:
```
Config(
    total_timesteps: 50000,
    rollout: (
        steps: 32,
        num_envs: 64,
        processes: 4,
    ),
    optim: (
        bs: 512,
        lr: 0.005,
    ),
    track: true,
    wandb_project_name: "TensorRTS_ENN",
    wandb_entity: "jp4"
)
```
First off, you can change the `total_timesteps` attribute. This specifies the total number of steps that will be used in training. For the initial training against the Rush AI, a (relatively) small number, around 10000, should be sufficient. Next, you will need to edit the values of `track`, `wandb_project_name`, and `wandb_entity`. The value of `track` will depend on whether or not you want to send the results of your training to WandB. If so, leave it set to `true`. Otherwise, set it to `false`. Next, the value of `wandb_project_name` will specify the project under which your training will be saved to WandB. You can leave it as is or change it as you desire. Finally, it will be necessary to change the value of `wandb_entity`. This should be your username in WandB and is case sensitive. Now, before training, if you wish to use WandB, ensure that you are logged in to WandB by inputting `wandb login` to your terminal. If you are not logged in, the training will hang after instantiating the environments.
Finally, you will be ready to begin training. The initial round of training will be performed against a built-in Rush AI.
```
python train.py --config=config.ron --checkpoint-dir=checkpoints
```
As you will see, checkpoints are automatically overwritten by the hyperstate library. As of the writing of this, hyperstate has no ability to save persistent checkpoints. There are several approaches for handling this for selfplay. The first is to load a constant checkpoint as the opponent player. To do so, run the following command. Notice that we change the `--checkpoint-dir` tag, so we are saving our trained bot to a new directory.
```
python train_selfplay.py --config=config.ron --checkpoint-dir=selfplay_checkpoints
```
The above script will also train a new bot from scratch. If you instead wish to continue training the previously trained bot, you will need to edit the `config.ron` file stored in the checkpoint directory and increase the `total_timesteps` attribute. The number of timesteps is tracked during the previous iteration of training, so the number of steps you want to train with selfplay will need to be added to the number already there. After doing this, you can run the following command: 
```
python train_from_checkpoint.py --config=[path_to_checkpoint_config] --checkpoint-dir=selfplay_checkpoints
```
Both of the above scripts by default load in the first checkpoint found in the directory `checkpoints` to be used as the opponent. If you have saved your checkpoints elsewhere, you can edit what is passed to the `load_checkpoint` function in either file.
If you instead wish to train against an opponent that is loaded from the most recently saved model of your bot, you can simply run either of the commands above, replacing the directory given to `--checkpoint-dir` with the same directory that `train_selplay.py` or `train_from_checkpoint.py` use to load checkpoints from.
In order to prepare a bot usable by the tournament runner, follow the directions laid out in `neutron_bot/README.md`.
