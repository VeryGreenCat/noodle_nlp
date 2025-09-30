import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from preprocess import tokenize
import json
import re

model = load_model("D:/Code/NLP/noodle_nlp/model1.h5", custom_objects={})
tokenizer = pickle.load(open("D:/Code/NLP/noodle_nlp/tokenizer1.pkl", "rb"))

max_seq_len = model.input_shape[0][1]  # encoder input length

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


# ---------- Step 2: โหลดไฟล์แล้ว test ----------

def clean_json_text(text):
    # ตัด token <start>, <end>
    text = text.replace("<start>", "").replace("<end>", "").strip()

    # JS style true/false → Python style
    text = re.sub(r'\btrue\b', 'True', text)
    text = re.sub(r'\bfalse\b', 'False', text)

    try:
        # แปลงเป็น Python dict
        obj = eval(text)

        # ฟังก์ชันลบช่องว่างใน value
        def strip_spaces(val):
            if isinstance(val, str):
                return val.replace(" ", "")  # ลบทุกช่องว่างใน string
            elif isinstance(val, dict):
                return {k: strip_spaces(v) for k, v in val.items()}
            elif isinstance(val, list):
                return [strip_spaces(v) for v in val]
            else:
                return val

        obj = strip_spaces(obj)

        # แปลงกลับเป็น JSON แบบ canonical
        return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    except Exception:
        # fallback ถ้า parse ไม่ได้
        return text

import json


import json

def evaluate_from_file(filename="../data/test_cases.txt"):
    total = 0
    correct = 0

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")  # 🔹 ตัด comma ออก
            if not line:
                continue
            try:
                case = json.loads(line)
                inp = case["input"]
                expected = case["expected"]
            except Exception as e:
                print(f"⚠️ Error parsing line: {line}")
                continue

            pred = predict(inp)
            pred_clean = clean_json_text(pred)

            try:
                pred_json = json.loads(pred_clean)
                exp_json = expected
                is_correct = (pred_json == exp_json)
            except Exception:
                is_correct = (pred_clean == json.dumps(expected, ensure_ascii=False))

            if is_correct:
                correct += 1
            total += 1

            print(f"Input   : {inp}")
            print(f"Expected: {json.dumps(expected, ensure_ascii=False)}")
            print(f"Pred    : {pred_clean}")
            print(f"{'✅ Correct' if is_correct else '❌ Wrong'}")
            print("-" * 50)

    acc = correct / total if total > 0 else 0
    print(f"🎯 Accuracy: {acc:.2%} ({correct}/{total})")
    return acc



# ---------- Example run ----------
if __name__ == "__main__":
    evaluate_from_file()