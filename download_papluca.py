from datasets import load_dataset
import pandas as pd

print("Downloading Papluca Language Identification dataset...")
dataset = load_dataset("papluca/language-identification")

train = dataset["train"]
valid = dataset["validation"]
test  = dataset["test"]

df_train = pd.DataFrame(train)
df_valid = pd.DataFrame(valid)
df_test  = pd.DataFrame(test)

df = pd.concat([df_train, df_valid, df_test])

# Rename columns
df = df.rename(columns={"text": "Text", "labels": "language"})

df.to_csv("dataset.csv", index=False)

print("Saved combined dataset as dataset.csv")
print(df["language"].value_counts())
