# epochs = 3

# prompts = data_df["prompt"].tolist()

# print(device)

# for epoch in range(epochs):
#     for prompt in prompts:
#         print("prompt", prompt)
#         input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
#         output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
#         output = tokenizer.decode(input_ids[0], skip_special_tokens=True)
#         print("output", output)
#         reward_input_ids = reward_tokenizer(output, return_tensors="pt").input_ids.to(
#             device
#         )
#         reward = reward_model(input_ids=reward_input_ids)
#         ppo_trainer.train()
#         # ppo_trainer.step([input_ids[0]], [output], [reward])
#         print(f"Prompt: {prompt}")
#         print(f"Response: {output}")
#         print(f"Reward: {reward}")
#     print(f"Epoch {epoch+1} complete.")


# messages = [
#     {"role": "user", "content": "Hello, how are you?"},
#     {"role": "assistant", "content": "I'm fine, thank you!"},
#     {"role": "user", "content": "What is the capital of France?"},
# ]

# input_text = tokenizer.apply_chat_template(
#     messages, tokenize=True, add_generation_prompt=True
# )
