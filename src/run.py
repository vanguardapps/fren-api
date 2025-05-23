import os
import torch
from data_df import data_df
from datasets import Dataset
from device import get_device
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
)
from trl import (
    PPOTrainer,
    PPOConfig,
)


# def base_object():
#     return type("basic_object", (object,), {})()


max_new_tokens = 20

load_dotenv()

token = os.getenv("HF_TOKEN")

device = get_device()
model_name = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
reward_tokenizer = AutoTokenizer.from_pretrained("roberta-base", token=token)

# Create a reference model for PPO
ref_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    generation_config=GenerationConfig(
        min_length=-1,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
    ),
)
ref_model.eval()
for param in ref_model.parameters():
    param.requires_grad = False

# Create a trainable model for PPO
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    generation_config=GenerationConfig(
        min_length=-1,
        top_k=0.0,
        top_p=1.0,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        output_scores=True,
    ),
)
model.train()
for param in model.parameters():
    param.requires_grad = True


value_model_name = "roberta-base"
value_model = AutoModelForSequenceClassification.from_pretrained(
    value_model_name, token=token
)
value_model.train()
for param in value_model.parameters():
    param.requires_grad = True

reward_model_name = "roberta-base"
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name, token=token
)
reward_model.eval()
for param in reward_model.parameters():
    param.requires_grad = False

config = PPOConfig(
    batch_size=4,
    mini_batch_size=1,
    learning_rate=1.41e-5,
    num_ppo_epochs=2,
    num_train_epochs=3,
)

train_dataset = Dataset.from_pandas(data_df)

tokenizer.pad_token = tokenizer.eos_token


def tokenize(example):
    return tokenizer(example["query"], padding=True)


train_dataset = train_dataset.map(tokenize, batched=True)


def data_collator(batch):
    input_ids = [example["input_ids"] for example in batch]
    return {"input_ids": torch.tensor(input_ids)}


ppo_trainer = PPOTrainer(
    config,
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)


ppo_trainer.train()
