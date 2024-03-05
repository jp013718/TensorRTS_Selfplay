# Neutron Bot
This is a bot for TensorRTS trained on selfplay using ENN. It inherits from the class `Agent` in the original TensorRTS.py (stored in parent directory here). In order to run, simply import the class `NeutronBot` from the file `neutron.py`. The bot loads a trained algorithm checkpoint from `neutron_checkpoint`, which is expected to be found in the same directory as `neutron.py`.
