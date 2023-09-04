import pandas as pd
from sklearn.model_selection import train_test_split
import re
import nltk
import ssl
import emoji
import os

ssl._create_default_https_context = ssl._create_unverified_context


emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)


def convert_emoji_to_text(text):
    return emoji.demojize(text, delimiters=(" ", " "))


def remove_emojis(data):
    return re.sub(emoj, '', data)


if __name__ == "__main__":
    nltk.download('punkt')

    base = "data/OLIDv2/"
    df = pd.read_csv("data/olid-training-v1.0.tsv", sep="\t")
    df["label"] = df["subtask_a"].replace({"OFF": 1, "NOT": 0})
    df.drop(labels=["id", "subtask_b", "subtask_c", "subtask_a"], inplace=True, axis=1)

    # Remove usernames starting with @
    df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'@\w+', '', x))
    # Remove hashtags
    df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'#', '', x))
    # Remove URLs
    df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))
    # Handle emoji
    # df["tweet"] = df["tweet"].apply(lambda x: remove_emojis(x))
    df["tweet"] = df["tweet"].apply(convert_emoji_to_text)
    # Remove extra whitespace
    df["tweet"] = df["tweet"].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    train, test = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])
    train, valid = train_test_split(train, test_size=0.1, random_state=42, stratify=train["label"])

    print(f"{len(df)    = }")
    print(f"{len(train) = }")
    print(f"{len(valid) = }")
    print(f"{len(test)  = }")

    if not os.path.exists(base):
        os.mkdir(base)

    train[["label", "tweet"]].to_csv(base + "OLIDv2-train.txt", index=False, header=False, sep='\t')
    test[["label", "tweet"]].to_csv(base + "OLIDv2-test.txt", index=False, header=False, sep='\t')
    valid[["label", "tweet"]].to_csv(base + "OLIDv2-dev.txt", index=False, header=False, sep='\t')

    vocab = set()
    for t in train['tweet'].apply(nltk.word_tokenize):
        vocab = vocab.union(set(t))
    vocab.add('UNK')

    with open(base + "OLIDv1-vocab.txt", "w") as f:
        f.write("\n".join(vocab))





