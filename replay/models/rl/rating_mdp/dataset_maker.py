import pandas as pd
train_df = pd.read_csv("../experiments/train_df.csv", sep = "\t")
test_df = pd.read_csv("../experiments/test_df.csv", sep = "\t")
train_df.loc[:,"dataset"]="train"
test_df.loc[:,"dataset"]="test"
df = train_df.append(test_df, ignore_index=True)
df = df.rename(columns={"user_idx": "user_id", "item_idx": "item_id", "relevance":"rating"})
df = df[['user_id', 'item_id', 'rating', 'timestamp', 'dataset']]
df.to_csv("ml_prepared.csv")