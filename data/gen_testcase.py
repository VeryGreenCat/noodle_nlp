import random
import json

# ---- Config ----
noodles_map = {
    "เล็ก": "เส้นเล็ก",
    "ใหญ่": "เส้นใหญ่",
    "หมี่": "หมี่ขาว",
    "บะหมี่": "บะหมี่",
    "วุ้นเส้น": "วุ้นเส้น"
}
styles_map = {
    "ตก": "น้ำตก",
    "ใส": "น้ำใส",
    "แห้ง": "แห้ง",
    "ต้มยำ": "ต้มยำ",
    "เย็นตาโฟ": "เย็นตาโฟ"
}
meats = ["หมู", "เนื้อ"]
options_map = {
    "ไม่พริก": {"no_chili": True},
    "ไม่งอก": {"no_bean_sprout": True},
    "ไม่ถั่ว": {"no_peanut": True},
    "ไม่ผัก": {"no_vegetable": True}
}
quantities = list(range(1, 6))

# ---- Generator ----
def gen_case():
    noodle_slang, noodle_full = random.choice(list(noodles_map.items()))
    style_slang, style_full = random.choice(list(styles_map.items()))
    meat = random.choice(meats)
    qty = random.choice(quantities)

    if random.random() < 0.5:
        opt_slang, opt_json = random.choice(list(options_map.items()))
    else:
        opt_slang, opt_json = None, None

    # ----- Build input string -----
    if opt_slang:
        # option ก่อน meat
        input_text = noodle_slang + style_slang + opt_slang + meat + str(qty)
    else:
        # ไม่มี option
        input_text = noodle_slang + style_slang + meat + str(qty)

    # ----- Build expected JSON -----
    expected = {
        "food": noodle_full,
        "quantity": qty,
        "style": style_full,
        "meat": meat
    }
    if opt_json:
        expected["options"] = opt_json

    return {"input": input_text, "expected": expected}

# ---- Save file ----
def save_cases(filename="test_cases.txt", n=20):
    with open(filename, "w", encoding="utf-8") as f:
        for _ in range(n):
            case = gen_case()
            f.write(json.dumps(case, ensure_ascii=False) + ",\n")
    print(f"✅ Saved {n} cases to {filename}")

if __name__ == "__main__":
    save_cases("test_cases.txt", n=100)
