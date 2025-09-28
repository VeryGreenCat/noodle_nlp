import pandas as pd
from pythainlp.tokenize import word_tokenize

# Load dataset
def load_data(path="data/orders.csv"):
    df = pd.read_csv(path)
    return df["input"].tolist(), df["output"].tolist()

# Tokenize Thai text
def tokenize(text):
    return word_tokenize(text, engine="newmm")

if __name__ == "__main__":
    X, y = load_data()
    print("Tokenized:", tokenize(X[0]))
