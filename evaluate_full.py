import os
import re
import csv
import json
import torch
import difflib
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional GPT-based judging
USE_GPT_JUDGE = True  # set to True if you have OpenAI API access

if USE_GPT_JUDGE:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ====================================================
# CONFIG
# ====================================================
MODEL_PATH = "merged_model_4bit"
DATA_PATH = "data/golden_data.jsonl"  # Your golden test data
REPORT_PATH = "evaluation_report.csv"

# ====================================================
# HELPERS
# ====================================================
def normalize_sql(sql: str) -> str:
    """Clean and normalize SQL string for comparison."""
    sql = sql.strip().lower()
    sql = re.sub(r"\s+", " ", sql)
    return sql

def parse_sql(sql: str):
    """Parse SQL into structured dictionary for semantic matching."""
    try:
        sql = normalize_sql(sql)
        cols_part = re.search(r"\((.*?)\)", sql).group(1)
        vals_part = re.search(r"values\s*\((.*?)\)", sql).group(1)
        cols = [c.strip() for c in cols_part.split(",")]
        vals = [v.strip().strip("'") for v in vals_part.split(",")]
        return {c: v for c, v in zip(cols, vals)}
    except Exception:
        return {}

def compute_similarity(a: str, b: str) -> float:
    """Compute fuzzy similarity score."""
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100

def gpt_judge(expected_sql, generated_sql):
    """Ask GPT to evaluate correctness."""
    prompt = f"""
    You are a SQL evaluation assistant.
    Compare the generated SQL with the expected one.
    Rate from 0 to 1 how correct it is (structure + meaning).
    Return JSON: {{"score": <float>, "reason": "<short reason>"}}.

    Expected:
    {expected_sql}

    Generated:
    {generated_sql}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"score": 0.0, "reason": f"Error: {e}"}

# ====================================================
# LOAD MODEL
# ====================================================
print("🚀 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ====================================================
# LOAD DATA
# ====================================================
print(f"📂 Loading test data from {DATA_PATH} ...")
dataset = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        ex = json.loads(line)
        dataset.append(ex)

# ====================================================
# EVALUATION LOOP
# ====================================================
results = []
strict_matches = 0
semantic_matches = 0
fuzzy_scores = []
gpt_scores = []

for i, ex in enumerate(tqdm(dataset, desc="Evaluating")):
    text = ex.get("complaint") or ex.get("message") or ex.get("input") or ex.get("text")
    expected_sql = ex.get("expected_sql") or ex.get("expected") or ex.get("sql")

    if not text or not expected_sql:
        continue

    # Prompt like training
    prompt = f"""<|system|>
You are a parser. Convert the given customer message into a SQL INSERT statement 
for the 'complaints' table with columns: order_id, item_name, issue, requested_action.
No extra text. If a field is unknown, use null.

<|user|>
{text}
"""

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "insert into" in full_output.lower():
        gen_sql = full_output[full_output.lower().find("insert into"):]
    else:
        gen_sql = full_output.strip()

    # -------------------------------
    # Metrics
    # -------------------------------
    strict = normalize_sql(gen_sql) == normalize_sql(expected_sql)
    fuzzy = compute_similarity(gen_sql, expected_sql)
    sem_exp, sem_gen = parse_sql(expected_sql), parse_sql(gen_sql)
    sem_match = sem_exp == sem_gen

    gpt_score = None
    if USE_GPT_JUDGE:
        gpt_eval = gpt_judge(expected_sql, gen_sql)
        gpt_score = gpt_eval.get("score", 0)
        gpt_reason = gpt_eval.get("reason", "")
    else:
        gpt_score = 0.0
        gpt_reason = "Disabled"

    # Update stats
    if strict:
        strict_matches += 1
    if sem_match:
        semantic_matches += 1
    fuzzy_scores.append(fuzzy)
    gpt_scores.append(gpt_score)

    results.append({
        "complaint": text,
        "expected": expected_sql,
        "generated": gen_sql,
        "strict_match": strict,
        "semantic_match": sem_match,
        "fuzzy_similarity": round(fuzzy, 2),
        "gpt_score": round(gpt_score, 3),
        "gpt_reason": gpt_reason,
    })

# ====================================================
# FINAL REPORT
# ====================================================
strict_acc = (strict_matches / len(results)) * 100
sem_acc = (semantic_matches / len(results)) * 100
avg_fuzzy = sum(fuzzy_scores) / len(fuzzy_scores)
avg_gpt = sum(gpt_scores) / len(gpt_scores)

print("\n📊 Evaluation Summary")
print(f"✅ Strict Accuracy: {strict_acc:.2f}%")
print(f"🧩 Semantic Accuracy: {sem_acc:.2f}%")
print(f"🔤 Fuzzy Similarity: {avg_fuzzy:.2f}%")
if USE_GPT_JUDGE:
    print(f"🤖 GPT Judge Score: {avg_gpt:.2f}")
else:
    print("🤖 GPT Judge: Skipped (set USE_GPT_JUDGE=True to enable)")

# Save CSV
print(f"\n💾 Saving detailed report to {REPORT_PATH}")
with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("\n✅ Done. Report ready!")

