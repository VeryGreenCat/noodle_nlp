import pandas as pd
from pythainlp.tokenize import word_tokenize
import re

# Load dataset
def load_data(path="data/noodle_data.csv"):
    df = pd.read_csv(path)
    return df["input"].tolist(), df["output"].tolist()

# 1. Normalize Thai text (reduce repeated characters only)
def normalize_text(text):
    # ลดตัวอักษรซ้ำ เช่น "ครับบบ" -> "ครับ"
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    return text

# 2. Tokenize Thai text
def tokenize(text):
    return word_tokenize(text, engine="newmm")

# 3. Remove custom stopwords form chatGPT
def remove_stopwords(tokens):
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
    "โคตร", "มาก", "เยอะ", "หน่อย", "หนัก", "เบา", "นิด", "น้อย", "แปป",]  # ใส่คำที่ไม่สำคัญสำหรับ Noodel
    return [t for t in tokens if t not in custom_stopwords]

# for testing
if __name__ == "__main__":
    X, y = load_data()

    text = "เอาเล็สส้มยำไม่กัก2ที่ครับบบบบ"
    print("Original:", text)

    norm = normalize_text(text)
    print("Normalized:", norm)

    tokens = tokenize(norm)
    print("Tokenized:", tokens)

    clean_tokens = remove_stopwords(tokens)
    print("After Stopword Removal:", clean_tokens)
