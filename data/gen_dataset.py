import random
import pandas as pd
import os
import json

os.makedirs("../data", exist_ok=True)

# ----- Config -----
noodles = ["เส้นเล็ก", "เส้นใหญ่", "บะหมี่", "วุ้นเส้น", "หมี่ขาว"]
styles = ["น้ำตก", "น้ำใส", "แห้ง", "ต้มยำ", "เย็นตาโฟ"]
meats = ["หมู", "เนื้อ"]
options_dict = {
    "ไม่พริก": {"no_chili": True},
    "ไม่งอก": {"no_bean_sprout": True},
    "ไม่ถั่ว": {"no_peanut": True},
    "ไม่ผัก": {"no_vegetable": True},
}
quantities = list(range(1, 11))  # 1 to 10

abbr_map = {
    "เส้นเล็ก": "เล็ก",
    "เส้นใหญ่": "ใหญ่",
    "วุ้นเส้น": "วุ้น",
    "วุ้นเส้น": "วุ้นเส้น",
    "หมี่ขาว": "หมี่",
    "น้ำตก": "ตก",
    "น้ำใส": "ใส",
}

# จำนวน typo แบบสุ่มต่อ variant (ปรับได้)
TYPO_PER_VARIANT = 3

def build_json(food, style=None, qty=1, opts=None, meat=None):
    obj = {"food": food, "quantity": qty}
    if style:
        obj["style"] = style
    if opts:
        obj["options"] = opts
    if meat:
        obj["meat"] = meat
    return obj

# สร้าง typo แบบสุ่ม 1 เวอร์ชัน (ลบตัวอักษรหรือสลับตัวติดกัน)
def make_random_typo(s):
    if len(s) <= 1:
        return s
    choice = random.choice(["delete", "swap", "drop_space"])
    if choice == "delete":
        idx = random.randint(0, len(s)-1)
        return s[:idx] + s[idx+1:]
    elif choice == "swap" and len(s) > 2:
        idx = random.randint(0, len(s)-2)
        lst = list(s)
        lst[idx], lst[idx+1] = lst[idx+1], lst[idx]
        return "".join(lst)
    else:  # drop_space: remove any spaces (if present)
        return s.replace(" ", "")

# สร้าง up to k unique typo จากสตริง s
def generate_k_typos(s, k):
    typos = set()
    attempts = 0
    max_attempts = k * 10  # ป้องกัน loop ไม่รู้จบ
    while len(typos) < k and attempts < max_attempts:
        t = make_random_typo(s)
        if t != s:
            typos.add(t)
        attempts += 1
    return list(typos)

def generate_all(keep_abbr=True):
    rows = []
    # สร้างทุกความเป็นไปได้ที่ถูกต้อง (full + abbr) แบบ brute-force
    for noodle in noodles:
        noodle_token = noodle.replace("เส้น", "")  # token ใช้ใน input
        noodle_abbr = abbr_map.get(noodle, noodle_token)

        for style in styles:
            style_abbr = abbr_map.get(style, style)
            for meat in meats:
                for opt_key in list(options_dict.keys()) + [None]:  # None = no option
                    for qty in quantities:
                        # สลับตำแหน่ง style<->meat ได้ 2 แบบ
                        for swapped in [False, True]:
                            if not swapped:
                                parts = [noodle_token, style, meat]
                                parts_abbr = [noodle_abbr, style_abbr, meat]
                            else:
                                parts = [noodle_token, meat, style]
                                parts_abbr = [noodle_abbr, meat, style_abbr]

                            if opt_key:
                                parts.append(opt_key)
                                parts_abbr.append(opt_key)
                            parts.append(str(qty))
                            parts_abbr.append(str(qty))

                            normal = "".join(parts)
                            abbr = "".join(parts_abbr) if keep_abbr else None

                            # output json (ใช้ชื่อเต็ม)
                            output_json = build_json(noodle, style, qty, options_dict.get(opt_key), meat)

                            # เก็บ original normal
                            rows.append((normal, json.dumps(output_json, ensure_ascii=False)))

                            # เก็บ abbr ถ้ามี
                            if abbr:
                                rows.append((abbr, json.dumps(output_json, ensure_ascii=False)))

                            # สร้าง typo แบบสุ่ม k ชุดสำหรับ normal และ abbr
                            for base in ([normal, abbr] if abbr else [normal]):
                                if base is None:
                                    continue
                                typos = generate_k_typos(base, TYPO_PER_VARIANT)
                                for t in typos:
                                    rows.append((t, json.dumps(output_json, ensure_ascii=False)))
    return rows

if __name__ == "__main__":
    rows = generate_all(keep_abbr=True)
    # ถ้าต้องการลบ duplicates:
    unique = {}
    for inp, out in rows:
        unique[inp] = out
    rows_final = list(unique.items())
    print("Raw rows:", len(rows), "Unique rows after dedupe:", len(rows_final))
    df = pd.DataFrame(rows_final, columns=["input", "output"])
    df.to_csv("orders_with_typos.csv", index=False, encoding="utf-8-sig")
    print("Saved", len(df), "rows to orders_with_typos.csv")
