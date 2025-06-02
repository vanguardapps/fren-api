# Import the config of the algorithm of your choice.
from ray.rllib.algorithms.ppo import PPOConfig

# Print out the abstract APIs, you need to subclass from and whose
# abstract methods you need to implement, besides the ``setup()`` and ``_forward_..()``
# methods.
print(PPOConfig().get_default_learner_class().rl_module_required_apis())
