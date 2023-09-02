import pandas as pd
from sklearn.model_selection import train_test_split
import re


if __name__ == "__main__":
    base = "data/OLIDv1/"
    df = pd.read_csv("data/olid-training-v1.0.tsv", sep="\t")
    df["label"] = df["subtask_a"].replace({"OFF": 1, "NOT": 0})
    df.drop(labels=["id", "subtask_b", "subtask_c", "subtask_a"], inplace=True, axis=1)

    # Remove usernames starting with @
    df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'@\w+', '', x))
    # Remove hashtags
    df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'#', '', x))
    # Remove URLs
    df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))
    # Remove extra whitespace
    df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    train, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
    train, valid = train_test_split(train, test_size=0.1, random_state=42, stratify=train["label"])

    print(f"{len(df)    = }")
    print(f"{len(train) = }")
    print(f"{len(valid) = }")
    print(f"{len(test)  = }")

    train[["label", "tweet"]].to_csv(base + "OLIDv1-train.txt", index=False, header=False, sep='\t')
    test[["label", "tweet"]].to_csv(base + "OLIDv1-test.txt", index=False, header=False, sep='\t')
    valid[["label", "tweet"]].to_csv(base + "OLIDv1-dev.txt", index=False, header=False, sep='\t')




