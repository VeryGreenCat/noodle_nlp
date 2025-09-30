import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import numpy as np
from preprocess import tokenize
import json
import re
import deepcut

custom_stopwords = [
    "ครับ","คับ","คั้บ", "ค่ะ", "จ้า", "จ๊ะ", "จ๋า", "ฮะ", "ผม", "ดิฉัน", "ฉัน",
    "นะ", "น่ะ", "เนอะ", "น้า", "นะคะ",
    "ด้วย", "ที", "หน่อย", "จัดให้หน่อย",
    "เอา", "ขอ", "จัด", "ทำ", "เอามา", "อยาก",
    "แบบ", "ใส่ให้", "เอาเป็นว่า", "เอาสัก",
    "ชาม", "ถ้วย", "ที่", "จาน",
    "และ", "กับ", ",", "|", "หรือ",
    "เอาด้วย", "อีก",
    "เลย", "แหละ", "ล่ะ", "เท่านั้น", "ก็พอ", "ก็แล้วกัน",
    "โคตร", "มาก", "เยอะ", "หน่อย", "หนัก", "เบา", "นิด", "น้อย", "แปป",
]

# ------------------------
# Load model & tokenizer
# ------------------------
model = load_model("D:/Code/NLP/noodle_nlp/model1.h5", custom_objects={})
tokenizer = pickle.load(open("D:/Code/NLP/noodle_nlp/tokenizer1.pkl", "rb"))

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


options_map_reverse = {
    "no_chili": "ไม่ใส่พริก",
    "no_bean_sprout": "ไม่ใส่ถั่วงอก",
    "no_peanut": "ไม่ใส่ถั่ว",
    "no_vegetable": "ไม่ใส่ผัก"
}

def normalize_order(output_json_str):
    """
    รับ string JSON จากโมเดล แล้วเรียงเป็น: 
    food + meat + style + option_text + quantity
    """
    # ลบ <start> หรือช่องว่างออกก่อน
    clean_str = output_json_str.strip()
    if clean_str.startswith("<start>"):
        clean_str = clean_str.replace("<start>", "").strip()
    if clean_str.startswith("<end>"):
        clean_str = clean_str.replace("<end>", "").strip()

    # แปลง string → dict
    try:
        order = json.loads(clean_str)
    except json.JSONDecodeError:
        # ถ้าเป็น list ก็ลองอีกแบบ
        order = json.loads(clean_str.replace("'", "\""))

    # ถ้ามีหลาย order (list)
    if isinstance(order, list):
        normalized_list = [format_one(o) for o in order]
        return " / ".join(normalized_list)
    else:
        return format_one(order)

def format_one(order_dict):
    """จัดเรียง 1 order"""
    food = order_dict.get("food", "").strip()
    meat = order_dict.get("meat", "").strip()
    style = order_dict.get("style", "").strip()
    qty = order_dict.get("quantity", "")
    opt_dict = order_dict.get("options", None)

    option_text = ""
    if isinstance(opt_dict, dict):
        # แปลง key → คำเต็มภาษาไทย
        for k, v in opt_dict.items():
            if v and k in options_map_reverse:
                option_text += options_map_reverse[k]

    # ประกอบเป็นประโยคตามลำดับ
    parts = [food, meat, style, option_text, str(qty)]
    # เอาเฉพาะที่ไม่ว่าง
    return "".join([p for p in parts if p])


def remove_stopword(text: str) -> str:
    # ตัดคำด้วย deepcut
    tokens = deepcut.tokenize(text)
    
    # ลบ stopwords
    filtered_tokens = [t for t in tokens if t.strip() and t not in custom_stopwords]
    
    # รวมเป็น string
    return "".join(filtered_tokens)


# ------------------------
# Run example
# ------------------------
if __name__ == "__main__":
    i = 1
    normalized_list = [] 
    while True:
        str_input = input(f"เมนูที่ {i} : ")
        if str_input.lower() == "e":
            break
        else:
            pred = predict(remove_stopword(str_input))
            normalized = normalize_order(pred)
            print("Raw order:", normalized)
            normalized_list.append(normalized)  
            i += 1
    # Print all normalized data after break
    print("\nAll orders:")
    for idx, item in enumerate(normalized_list, 1):
        print(f"{idx}: {item}")