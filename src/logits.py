import torch
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
)


def compute_attention_mask(token_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    return torch.where(token_ids != pad_token_id, 1, 0)


model_name = "gpt2"
max_query_len = 128
max_response_len = 128
top_p = 0.95
top_k = 0

model = AutoModelForCausalLM.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)
hidden_dim = config.hidden_size
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

pad_token_id = (
    tokenizer.pad_token_id
    if tokenizer.pad_token_id is not None
    else tokenizer.eos_token_id
)

generation_config = GenerationConfig(
    max_new_tokens=max_response_len,
    top_p=top_p,
    top_k=top_k,
    do_sample=True,
    output_hidden_states=True,
    return_dict_in_generate=True,
    output_logits=True,
)

query = ["Hello, how are you today?", "Is your name George?"]

tokenizer.pad_token_id = tokenizer.eos_token_id

input_ids = tokenizer(
    query,
    return_tensors="np",
    max_length=max_query_len,
    padding="max_length",
    truncation=True,
)["input_ids"]

input_ids = torch.tensor(input_ids)
attention_mask = compute_attention_mask(input_ids, tokenizer.pad_token_id)

output = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    generation_config=generation_config,
)

logits = torch.stack(output.logits, dim=1)

print("logits.size()", logits.size())

output_ids = output.sequences[:, 128:]
computed_sequences = torch.argmax(logits, dim=2)

print("output_ids.size()", output_ids.size())
print("computed_sequences.size()", computed_sequences.size())

for i in range(128):
    if output_ids[0, i].item() == computed_sequences[0, i].item():
        print("same!")
    else:
        print("different!")

# Return single array if input was a string, otherwise batch
# return encodings[0] if isinstance(query, str) else encodings
