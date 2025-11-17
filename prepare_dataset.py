import json, re, random
from pathlib import Path

DATA_DIR = Path("content/data")
RAW_FILE = DATA_DIR / "complaints.txt"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# --- Heuristic parsers ---
def detect_issue(text):
    t = text.lower()
    if any(w in t for w in ["broken", "cracked", "damaged", "leak", "faulty"]): return "damaged_item"
    if any(w in t for w in ["missing", "not received", "lost"]): return "missing_item"
    if any(w in t for w in ["wrong", "incorrect", "different"]): return "wrong_item"
    if any(w in t for w in ["late", "delay"]): return "delayed_delivery"
    return "other_issue"

def detect_action(text):
    t = text.lower()
    if "refund" in t: return "refund"
    if any(w in t for w in ["replace", "resend", "send again", "exchange"]): return "replacement"
    return "null"

def detect_order(text):
    m = re.search(r"(?:order\s*#?\s*)(\d{3,6})", text, re.I)
    return m.group(1) if m else "null"

def detect_item(text):
    match = re.search(r"for the ([\w\s]+?)(?:,|\.|\sorder)", text.lower())
    return match.group(1).strip() if match else "item"

# --- Convert ---
with open(RAW_FILE, "r", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]

random.shuffle(lines)
split = int(0.8 * len(lines))
train_lines, val_lines = lines[:split], lines[split:]

def to_jsonl(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for msg in lines:
            issue = detect_issue(msg)
            action = detect_action(msg)
            oid = detect_order(msg)
            item = detect_item(msg)

            sql = (
                "INSERT INTO complaints (order_id, item_name, issue, requested_action)\n"
                f"VALUES ('{oid}', '{item}', '{issue}', '{action}');"
            )
            record = {
                "text": f"<|system|>\nYou are a parser. Convert the given message into SQL INSERT for table complaints(order_id, item_name, issue, requested_action).\n<|user|>\n{msg}\n<|assistant|>\n{sql}"
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

to_jsonl(train_lines, TRAIN_FILE)
to_jsonl(val_lines, VAL_FILE)
print(f"✅ Created {len(train_lines)} train + {len(val_lines)} val samples.")
