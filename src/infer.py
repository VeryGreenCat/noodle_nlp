import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from preprocess import tokenize

# ------------------------
# Load model & tokenizer
# ------------------------
model = load_model("D:/Python/noodle_nlp/model.h5", custom_objects={})
tokenizer = pickle.load(open("D:/Python/noodle_nlp/tokenizer.pkl", "rb"))

max_seq_len = model.input_shape[0][1]  # encoder input length

# ------------------------
# Decode greedy
# ------------------------
def predict(text):
    # Encode input
    seq = tokenizer.texts_to_sequences([" ".join(tokenize(text))])
    seq = pad_sequences(seq, maxlen=max_seq_len, padding="post")

    # Start decoding with <start> token
    start_token = tokenizer.word_index["<start>"]
    end_token = tokenizer.word_index["<end>"]
    decoder_input = np.array([[start_token]])

    output_tokens = []

    for _ in range(max_seq_len):
        decoder_input_pad = pad_sequences(decoder_input, maxlen=max_seq_len, padding="post")
        pred = model.predict([seq, decoder_input_pad], verbose=0)
        next_token = pred[0, len(decoder_input[0])-1].argmax()

        if next_token == end_token:
            break

        output_tokens.append(next_token)
        decoder_input = np.concatenate([decoder_input, [[next_token]]], axis=1)

    # Convert token IDs to text
    output_text = tokenizer.sequences_to_texts([output_tokens])[0]
    return output_text

# ------------------------
# Parse normalized string to JSON
# ------------------------
def parse_to_json(normalized_text):
    """
    Example: "เส้นเล็ก น้ำ ไม่ใส่พริก"
    Output:
    {
        "noodle_type": "เส้นเล็ก",
        "soup": "น้ำ",
        "chili": False,
        "quantity": 1
    }
    """
    data = {
        "noodle_type": None,
        "soup": None,
        "chili": True,
        "quantity": 1
    }

    words = normalized_text.split()
    for i, w in enumerate(words):
        if "เส้น" in w:
            data["noodle_type"] = w
            # Check if next word is a number
            if i + 1 < len(words) and words[i+1].isdigit():
                data["quantity"] = int(words[i+1])
        if "น้ำ" in w or "แห้ง" in w:
            data["soup"] = w
        if "ไม่ใส่พริก" in w:
            data["chili"] = False

    return data

# ------------------------
# Run example
# ------------------------
if __name__ == "__main__":
    text = "หมีตกไม่งอกพริก"  # input slang text
    normalized = predict(text)
    print("Normalized string:", normalized)
    print("Parsed JSON:", parse_to_json(normalized))
