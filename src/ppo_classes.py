import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.utils.typing import TensorType
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.torch import TorchRLModule
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)
from typing import Callable, Optional, TypedDict, Union
from ray.rllib.utils.annotations import override


class TextGenEnvConfig(TypedDict):
    dataframe: pd.DataFrame
    max_query_len: int
    max_response_len: int
    reward_estimator: Callable[[str, str], float]


class TextGenEnv(gym.Env):
    @override(gym.Env)
    def __init__(self, config: TextGenEnvConfig):
        self._validate_config(config)

        self.dataframe = config["dataframe"]
        self.reward_estimator = config["reward_estimator"]
        self.max_query_len = config["max_query_len"]
        self.max_response_len = config["max_response_len"]

        self.observation_space = gym.spaces.Text(max_length=self.max_query_len)
        self.action_space = gym.spaces.Text(max_length=self.max_response_len)

        self.np_random, _ = gym.utils.seeding.np_random(0)

    def _validate_config(self, config: dict):
        required_keys = ["reward_estimator", "max_query_len", "max_response_len"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        if not callable(config["reward_estimator"]):
            raise TypeError("reward_estimator must be a function or callable module")

    def _generate_observation(self) -> str:
        idx = self.np_random.integers(len(self.dataframe))
        return self.dataframe.iloc[idx]["query"]

    def _get_obs(self) -> str:
        return self.observation

    def _get_reward(self, observation: str, action: str) -> float:
        return self.reward_estimator(observation, action)

    def _get_terminated(self) -> bool:
        return True

    def _get_truncated(self) -> bool:
        return False

    def _get_info(self) -> dict:
        return {}

    @override(gym.Env)
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[str, str]:
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        super().reset(seed=seed)
        self.observation = self._generate_observation()
        return self._get_obs(), self._get_info()

    @override(gym.Env)
    def step(self, action: str) -> tuple[str, float, bool, bool, str]:
        current_observation = self._get_obs()
        self.observation = self._generate_observation()
        return (
            self._get_obs(),
            self._get_reward(current_observation, action),
            self._get_terminated(),
            self._get_truncated(),
            self._get_info(),
        )


class HFTokenizedTextEnvConfig(TextGenEnvConfig):
    model_name: str


class HFTokenizedTextEnv(TextGenEnv):
    @override(TextGenEnv)
    def __init__(self, config: HFTokenizedTextEnvConfig):
        super().__init__(config)

        self._validate_config(config)

        self.model_name = config["model_name"]

        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"], use_fast=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.observation_space = gym.spaces.MultiDiscrete(
            [self.tokenizer.vocab_size] * self.max_query_len
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [self.tokenizer.vocab_size] * self.max_response_len
        )

    def _validate_config(self, config: dict):
        super()._validate_config(config)

        required_keys = ["model_name"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        if not isinstance(config["model_name"], str):
            raise TypeError("model_name must be a string")

    def _encode_query(self, query: Union[str, list[str]]) -> np.ndarray:
        encodings = self.tokenizer(
            query,
            return_tensors="np",
            max_length=self.max_query_len,
            padding="max_length",
            truncation=True,
        )["input_ids"]

        # Return single array if input was a string, otherwise batch
        return encodings[0] if isinstance(query, str) else encodings

    def _decode_response(
        self, response: Union[np.ndarray, list[np.ndarray]]
    ) -> Union[str, list[str]]:
        if isinstance(response, np.ndarray) and response.ndim == 1:
            return self.tokenizer.decode(response, skip_special_tokens=True)
        else:
            return self.tokenizer.batch_decode(response, skip_special_tokens=True)

    @override(TextGenEnv)
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[np.ndarray, str]:
        observation, info = super().reset(seed, options)
        observation = self._encode_query(observation)
        assert observation.shape == (self.max_query_len,)
        return observation, info

    @override(TextGenEnv)
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, str]:
        assert action.shape[0] <= self.max_response_len

        action = self._decode_response(action)
        observation, reward, terminated, truncated, info = super().step(action)
        observation = self._encode_query(observation)

        assert observation.shape == (self.max_query_len,)
        return observation, reward, terminated, truncated, info


class HFTextRLModule(TorchRLModule):

    # @staticmethod
    # def _model_has_task_specific_head(model) -> bool:
    #     head_candidates = ["lm_head", "classifier", "score", "qa_outputs", "heads"]
    #     for attr in head_candidates:
    #         if hasattr(model, attr):
    #             return True
    #     return False

    @staticmethod
    def _compute_attention_mask(
        token_ids: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        return torch.where(token_ids != pad_token_id, 1, 0)

    @override(TorchRLModule)
    def setup(self):
        self._validate_model_config()

        self.model_name = self.model_config["model_name"]
        self.strip_prompt_from_response = self.model_config[
            "strip_prompt_from_response"
        ]
        self.max_query_len = self.model_config["max_query_len"]
        self.max_response_len = self.model_config["max_response_len"]
        self.top_p = self.model_config.get("top_p", 0.95)
        self.top_k = self.model_config.get("top_k", 0)

        self._policy_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        policy_config = AutoConfig.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        hidden_dim = policy_config.hidden_size

        policy_tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.pad_token_id = (
            policy_tokenizer.pad_token_id
            if policy_tokenizer.pad_token_id is not None
            else policy_tokenizer.eos_token_id
        )

        # TODO: Figure this out once you have the basic thing working. We will have to programmatically link the policy head to the model.
        # input_dim = self.observation_space.shape[0]
        # if not HFTextRLModule._model_has_task_specific_head(self._model):
        #     # TODO: Make this default policy head configurable
        #     self._policy_head = nn.Sequential(
        #         nn.Linear(input_dim, hidden_dim),
        #         nn.ReLU(),
        #         nn.Linear(hidden_dim, output_dim),
        #     )

        # TODO: Make this default value head configurable
        self._value_model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        self._generation_config = GenerationConfig(
            max_new_tokens=self.max_response_len,
            top_p=self.top_p,
            top_k=self.top_k,
            do_sample=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_logits=True,
        )

    def _validate_model_config(self):
        required_keys = [
            "model_name",
            "strip_prompt_from_response",
            "max_query_len",
            "max_response_len",
        ]
        for key in required_keys:
            if key not in self.model_config:
                raise ValueError(f"Missing required self.model_config key: {key}")

    @override(TorchRLModule)
    def forward_inference(self, batch, **kwargs) -> dict:
        """Forward pass through the RL module."""
        obs = batch[Columns.OBS].long()
        attention_mask = self._compute_attention_mask(obs, self.pad_token_id)

        with torch.no_grad():
            policy_outputs = self._policy_model.generate(
                input_ids=obs,
                attention_mask=attention_mask,
                generation_config=self._generation_config,
            )

            actions = (
                policy_outputs.sequences[:, obs.shape[1] :]
                if self.strip_prompt_from_response
                else policy_outputs.sequences
            )

            log_probs = torch.nn.functional.log_softmax(policy_outputs.logits, dim=-1)
            log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(
                -1
            )
            log_probs = log_probs.sum(dim=1)

        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: log_probs,
        }

    @override(TorchRLModule)
    def forward_train(self, batch, **kwargs) -> dict:
        """Forward pass through the RL module."""
        obs = batch[Columns.OBS].long()
        attention_mask = self._compute_attention_mask(obs, self.pad_token_id)

        policy_outputs = self._policy_model.generate(
            input_ids=obs,
            attention_mask=attention_mask,
            generation_config=self._generation_config,
        )

        actions = (
            policy_outputs.sequences[:, obs.shape[1] :]
            if self.strip_prompt_from_response
            else policy_outputs.sequences
        )

        log_probs = torch.nn.functional.log_softmax(policy_outputs.logits, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        log_probs = log_probs.sum(dim=1)

        last_hidden_state = policy_outputs.hidden_states[-1]
        values = self._value_model(last_hidden_state[:, -1, :]).squeeze(-1)

        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: log_probs,
            Columns.VF_PREDS: values,
        }

    @override(TorchRLModule)
    def forward_exploration(self, batch, **kwargs) -> dict:
        """Forward exploration pass through the RLModule."""
        obs = batch[Columns.OBS].long()
        attention_mask = self._compute_attention_mask(obs, self.pad_token_id)

        policy_outputs = self._policy_model.generate(
            input_ids=obs,
            attention_mask=attention_mask,
            generation_config=self._generation_config,
        )

        actions = (
            policy_outputs.sequences[:, obs.shape[1] :]
            if self.strip_prompt_from_response
            else policy_outputs.sequences
        )

        logits = torch.stack(policy_outputs.logits, dim=1)

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs = log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
        log_probs = log_probs.sum(dim=1)

        return {
            Columns.ACTIONS: actions,
            Columns.ACTION_LOGP: log_probs,
        }

    @override(TorchRLModule)
    def value_function(self, batch, **kwargs) -> dict:
        """Forward pass through the RL module."""
        obs = batch[Columns.OBS].long()
        attention_mask = self._compute_attention_mask(obs, self.pad_token_id)

        with torch.no_grad():
            policy_outputs = self._policy_model.generate(
                input_ids=obs,
                attention_mask=attention_mask,
                generation_config=self._generation_config,
            )

        last_hidden_state = policy_outputs.hidden_states[-1]
        values = self._value_model(last_hidden_state[:, -1, :]).squeeze(-1)

        return {
            Columns.VF_PREDS: values,
        }
