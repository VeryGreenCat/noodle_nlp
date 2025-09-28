import random
import pandas as pd
import os
import re

# ----- Ensure data folder exists -----
os.makedirs("../data", exist_ok=True)

# ----- Config -----
noodles = ["เส้นเล็ก", "เส้นใหญ่", "บะหมี่", "วุ้นเส้น", "หมี่ขาว"]
styles = ["น้ำตก", "น้ำใส", "แห้ง", "ต้มยำ", "เย็นตาโฟ"]
options_dict = {
    "ไม่พริก": {"no_chili": True},
    "ไม่งอก": {"no_bean_sprout": True},
    "ไม่ถั่ว": {"no_peanut": True},
    "ไม่ผัก": {"no_vegetable": True},
}
quantities = list(range(1, 11))  # 1 to 10

# ----- Build JSON output -----
def build_json(food, style=None, qty=1, opts=None):
    obj = {"food": food, "quantity": qty}
    if style:
        obj["style"] = style
    if opts:
        obj["options"] = opts
    return obj

# ----- Generate typo/slang variations -----
def add_typos(input_text):
    variations = [input_text]
    # remove spaces
    variations.append(input_text.replace(" ", ""))
    # drop one letter randomly
    if len(input_text) > 3:
        idx = random.randint(0, len(input_text)-2)
        variations.append(input_text[:idx] + input_text[idx+1:])
    # swap adjacent letters randomly
    if len(input_text) > 3:
        idx = random.randint(0, len(input_text)-2)
        lst = list(input_text)
        lst[idx], lst[idx+1] = lst[idx+1], lst[idx]
        variations.append("".join(lst))
    return list(set(variations))  # unique

# ----- Generate single or multi orders -----
def generate_order():
    rows = []
    for _ in range(1000):  # change to how many base orders you want
        if random.random() < 0.7:
            # single order
            noodle = random.choice(noodles)
            style = random.choice(styles)
            qty = random.choice(quantities)
            opt_text, opt_json = random.choice(list(options_dict.items())) if random.random() < 0.3 else (None, None)

            # slang input
            input_text = noodle.replace("เส้น", "") + style
            if qty > 1:
                input_text += str(qty)
            if opt_text:
                input_text += opt_text

            output_json = build_json(noodle, style, qty, opt_json)
            typo_versions = add_typos(input_text)
            for typo_input in typo_versions:
                rows.append((typo_input, str(output_json).replace("'", '\"')))

        else:
            # multi-order (2 orders)
            noodle1, noodle2 = random.sample(noodles, 2)
            style1, style2 = random.sample(styles, 2)
            qty1, qty2 = random.choice(quantities), random.choice(quantities)
            opt1 = random.choice(list(options_dict.values())) if random.random() < 0.3 else None
            opt2 = random.choice(list(options_dict.values())) if random.random() < 0.3 else None

            input_text = noodle1.replace("เส้น", "") + style1 + str(qty1) + " " + noodle2.replace("เส้น", "") + style2 + str(qty2)
            if opt1:
                input_text += "ไม่พริก" if "no_chili" in opt1 else ""
            if opt2:
                input_text += "ไม่งอก" if "no_bean_sprout" in opt2 else ""

            output_json = [
                build_json(noodle1, style1, qty1, opt1),
                build_json(noodle2, style2, qty2, opt2)
            ]
            typo_versions = add_typos(input_text)
            for typo_input in typo_versions:
                rows.append((typo_input, str(output_json).replace("'", '\"')))
    return rows

# ----- Save CSV -----
def save_to_csv(filename="orders.csv", n_rows=100000):
    rows = generate_order()
    # Trim or duplicate to reach n_rows
    if len(rows) < n_rows:
        times = n_rows // len(rows) + 1
        rows = (rows * times)[:n_rows]
    else:
        rows = rows[:n_rows]

    df = pd.DataFrame(rows, columns=["input", "output"])
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"✅ Saved {len(df)} rows to {filename}")

# ----- Main -----
if __name__ == "__main__":
    save_to_csv(n_rows=100000)  # adjust number as needed
