import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from difflib import SequenceMatcher

# ===================== CONFIG ======================
MODEL_DIR = "merged_model_4bit"       # your quantized model path
GOLDEN_FILE = "golden_data.jsonl"     # golden dataset
MAX_NEW_TOKENS = 150
# ===================================================

# ---- Must match training prompt exactly ----
SYSTEM_PROMPT = """<|system|>
You are a parser. Convert the given customer message into a SQL INSERT statement 
for the 'complaints' table with columns: order_id, item_name, issue, requested_action.
No extra text. If a field is unknown, use null."""

# ---- Load model & tokenizer ----
print("🚀 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)

# ---- Utility functions ----
def extract_sql(text: str):
    """Extract only SQL statement from model output"""
    sql_match = re.search(r"insert\s+into.*?;", text, re.IGNORECASE | re.DOTALL)
    return sql_match.group(0).strip() if sql_match else text.strip()

def normalize_sql(sql: str):
    """Normalize SQL text, remove noise, and unify synonyms."""
    sql = sql.lower()
    sql = re.sub(r"\s+", " ", sql.strip())
    sql = sql.replace("damaged", "damaged_item")
    sql = sql.replace("broken", "damaged_item")
    sql = sql.replace("cracked", "damaged_item")
    sql = sql.replace("leaked", "damaged_item")
    sql = sql.replace("defective", "defective_item")
    sql = sql.replace("wrong color", "wrong_item")
    sql = sql.replace("wrong size", "wrong_item")
    sql = sql.replace("wrong product", "wrong_item")
    sql = sql.replace("missing", "missing_item")
    sql = sql.replace("send back", "exchange")
    sql = sql.replace("resend", "replacement")
    sql = sql.replace("replace", "replacement")
    sql = sql.replace("return", "refund")
    sql = sql.replace("refund pls", "refund")
    return sql.strip()

def semantic_match(expected_sql, generated_sql):
    """Check if generated SQL is semantically correct."""
    expected_norm = normalize_sql(expected_sql)
    generated_norm = normalize_sql(generated_sql)

    # Extract string fields from VALUES(...)
    expected_fields = re.findall(r"'(.*?)'", expected_norm)
    generated_fields = re.findall(r"'(.*?)'", generated_norm)

    # Field-level matching
    field_matches = 0
    for g in generated_fields:
        for e in expected_fields:
            if e in g or g in e:
                field_matches += 1
                break

    field_ratio = field_matches / max(1, len(expected_fields))
    semantic_ratio = SequenceMatcher(None, expected_norm, generated_norm).ratio()

    # consider semantically same if ratio high or 75% field match
    return semantic_ratio > 0.85 or field_ratio > 0.75, semantic_ratio * 100

# ---- Evaluation ----
print("🧠 Evaluating model against golden dataset...")

total = 0
strict_correct = 0
semantic_correct = 0
results = []

with open(GOLDEN_FILE, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line)
        complaint = sample["complaint"]
        expected_sql = sample["expected_sql"]

        # match your training structure
        prompt = f"""{SYSTEM_PROMPT}
<|user|>
{complaint}
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to("cuda")

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_sql = extract_sql(decoded)

        # Compare results
        strict_match = normalize_sql(generated_sql) == normalize_sql(expected_sql)
        sem_match, sim_score = semantic_match(expected_sql, generated_sql)

        total += 1
        if strict_match:
            strict_correct += 1
        if sem_match:
            semantic_correct += 1

        results.append({
            "complaint": complaint,
            "expected": expected_sql,
            "generated": generated_sql,
            "strict_match": strict_match,
            "semantic_match": sem_match,
            "similarity": round(sim_score, 2)
        })

        print(f"\n🧾 Complaint: {complaint}")
        print(f"Expected: {expected_sql}")
        print(f"Got:      {generated_sql}")
        print(f"Strict: {'✅' if strict_match else '❌'} | Semantic: {'✅' if sem_match else '❌'} | Similarity: {sim_score:.2f}%")
        print("-" * 90)

# ---- Final scores ----
strict_acc = (strict_correct / total) * 100 if total > 0 else 0
semantic_acc = (semantic_correct / total) * 100 if total > 0 else 0

print(f"\n🎯 Strict Accuracy: {strict_acc:.2f}% ({strict_correct}/{total})")
print(f"💡 Semantic Accuracy: {semantic_acc:.2f}% ({semantic_correct}/{total})")

# ---- Save report ----
with open("evaluation_report.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("\n📊 Evaluation report saved to evaluation_report.json ✅")
