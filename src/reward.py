from data_df import data_df


def reward_fn(prompt: str, output: str):
    expected_output = output = data_df.loc[data_df["query"] == prompt]
    return 1 if output.trim() == expected_output.trim() else 0
