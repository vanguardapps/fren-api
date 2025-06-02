import os
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from pprint import pprint
from ppo_classes import HFTextRLModule, HFTokenizedTextEnv
from reward import reward_estimator_fn
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from data_df import data_df


ray.init()


def env_creator(config):
    return HFTokenizedTextEnv(config)


register_env("HFTokenizedTextEnv-v0", env_creator)


model_name = "gpt2"

max_query_len = 128
max_response_len = 128

config = (
    PPOConfig()
    .environment(
        "HFTokenizedTextEnv-v0",
        env_config={
            "dataframe": data_df,
            "model_name": model_name,
            "reward_estimator": reward_estimator_fn,
            "max_query_len": max_query_len,
            "max_response_len": max_response_len,
        },
    )
    .env_runners(num_env_runners=2)
    .framework("torch")
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=HFTextRLModule,
            # inference_only=False,
            model_config={
                "model_name": model_name,
                "strip_prompt_from_response": True,
                "max_query_len": max_query_len,
                "max_response_len": max_response_len,
                # "top_p": 0.95,
                # "top_k": 0,
            },
        )
    )
    .training(lr=0.0002, train_batch_size_per_learner=1000, num_epochs=2)
    # .debugging(log_level="INFO")
    # .resources(num_gpus=1)
)

ppo = config.build_algo()

for _ in range(4):
    pprint(ppo.train())

save_path = os.path.abspath("../checkpoints")
save_dir = ppo.save_to_path(f"file://{save_path}")
print(save_dir)
