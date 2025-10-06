import random
import pandas as pd
import os
import json
import itertools

random.seed(42)
os.makedirs("../data", exist_ok=True)

# ----- Config -----
noodles = ["เส้นเล็ก", "เส้นใหญ่", "บะหมี่เหลือง", "วุ้นเส้น", "หมี่ขาว","มาม่า","เกาเหลา"]
styles = ["น้ำตก", "น้ำใส", "แห้ง", "ต้มยำ", "เย็นตาโฟ"]
meats = ["หมูชิ้น","หมูตุ๋น","เนื้อสด","เนื้อตุ๋น","เนื้อเปื่อย","ปลา"]
options_dict = {
    "ไม่พริก": {"no_chili": True},
    "ไม่งอก": {"no_bean_sprout": True},
    "ไม่ถั่ว": {"no_peanut": True},
    "ไม่ผัก": {"no_vegetable": True},
    "ไม่กระเทียมเจียว": {"no_garlic": True},
    "ไม่ตับ": {"no_liver": True},
    "ไม่เครื่องใน": {"no_offal": True},
}
quantities = list(range(1, 11))  # 1 to 10

abbr_map = {
    "เส้นเล็ก": "เล็ก",
    "เส้นใหญ่": "ใหญ่",
    "บะหมี่เหลือง": "บะหมี่",
    "เกาเหลา": "เหลา",
    "วุ้นเส้น": "วุ้น",
    "หมี่ขาว": "หมี่",
    "น้ำตก": "ตก",
    "น้ำใส": "ใส",
    "ต้มยำ": "ยำ",
    "เย็นตาโฟ": "โฟ",
    "หมูชิ้น": "หมู",
    "เนื้อสด": "สด",
    "เนื้อเปื่อย": "เปื่อย",
    
}

TYPO_PER_VARIANT = 3


def limited_permutations(tokens, limit=8):
    perms = list(itertools.permutations(tokens))
    if len(perms) > limit:
        perms = random.sample(perms, limit)
    return perms


def build_json(food, style=None, qty=1, opts=None, meat=None):
    obj = {"food": food, "quantity": qty}
    if style:
        obj["style"] = style
    if opts:
        obj["options"] = opts
    if meat:
        obj["meat"] = meat
    return obj


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
    else:
        return s.replace(" ", "")


def generate_k_typos(s, k):
    typos = set()
    attempts = 0
    max_attempts = k * 10
    while len(typos) < k and attempts < max_attempts:
        t = make_random_typo(s)
        if t != s:
            typos.add(t)
        attempts += 1
    return list(typos)


def generate_all(keep_abbr=True):
    rows = []
    for noodle in noodles:
        noodle_token = noodle.replace("เส้น", "")
        noodle_abbr = abbr_map.get(noodle, noodle_token)

        for style in styles:
            style_abbr = abbr_map.get(style, style)
            for meat in meats:
                for opt_key in list(options_dict.keys()) + [None]:
                    for qty in quantities:

                        # component เต็ม
                        parts_full = [noodle_token, style, meat]
                        parts_abbr = [noodle_abbr, style_abbr, meat]

                        if opt_key:
                            parts_full.append(opt_key)
                            parts_abbr.append(opt_key)

                        parts_full.append(str(qty))
                        parts_abbr.append(str(qty))

                        # output json
                        output_json = build_json(noodle, style, qty, options_dict.get(opt_key), meat)

                        # สลับตำแหน่งทุกแบบ (permutations)
                        for parts, parts_abbr in [(parts_full, parts_abbr)]:
                            for perm in limited_permutations(parts, limit=8):
                                normal = "".join(perm)
                                rows.append((normal, json.dumps(output_json, ensure_ascii=False)))

                            if keep_abbr:
                                for perm in limited_permutations(parts_abbr, limit=8):
                                    abbr = "".join(perm)
                                    rows.append((abbr, json.dumps(output_json, ensure_ascii=False)))
                                    
                        # เพิ่ม typo หรือคำผิด
                        for base in [ "".join(parts_full), "".join(parts_abbr) if keep_abbr else None]:
                            if base:
                                typos = generate_k_typos(base, TYPO_PER_VARIANT)
                                for t in typos:
                                    rows.append((t, json.dumps(output_json, ensure_ascii=False)))

    return rows


if __name__ == "__main__":
    rows = generate_all(keep_abbr=True)
    unique = {}
    for inp, out in rows:
        unique[inp] = out
    rows_final = list(unique.items())
    print("Raw rows:", len(rows), "Unique rows after dedupe:", len(rows_final))
    df = pd.DataFrame(rows_final, columns=["input", "output"])
    df.to_csv("orders_with_typos_and_perms.csv", index=False, encoding="utf-8-sig")
    print("Saved", len(df), "rows to orders_with_typos_and_perms.csv")
