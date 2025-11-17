import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

# ==========================================
# CONFIG
# ==========================================
BASE_MODEL = "microsoft/phi-2"  # or your fine-tuned model path
OUTPUT_FILE = "data/train_augmented.jsonl"
NUM_VARIANTS_PER_SAMPLE = 10
MAX_NEW_TOKENS = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================================
# BASE CANONICAL EXAMPLES (seed set)
# ==========================================
base_examples = [
    {
        "complaint": "my phone (order 1001) screen broken. please replace.",
        "sql": "INSERT INTO complaints (order_id, item_name, issue, requested_action) VALUES ('1001', 'phone', 'damaged_item', 'replacement');"
    },
    {
        "complaint": "order id 1002, blender leaking from the bottom. refund pls.",
        "sql": "INSERT INTO complaints (order_id, item_name, issue, requested_action) VALUES ('1002', 'blender', 'damaged_item', 'refund');"
    },
    {
        "complaint": "my order 1003 arrived but one cup was missing. resend pls.",
        "sql": "INSERT INTO complaints (order_id, item_name, issue, requested_action) VALUES ('1003', 'cup', 'missing_item', 'replacement');"
    },
    {
        "complaint": "received wrong color t-shirt (order 1004). need correct one.",
        "sql": "INSERT INTO complaints (order_id, item_name, issue, requested_action) VALUES ('1004', 't-shirt', 'wrong_item', 'exchange');"
    },
    {
        "complaint": "my earphones (order 1005) not charging properly. replacement pls.",
        "sql": "INSERT INTO complaints (order_id, item_name, issue, requested_action) VALUES ('1005', 'earphones', 'defective_item', 'replacement');"
    },
    {
        "complaint": "order id 1006 delayed again. update delivery status.",
        "sql": "INSERT INTO complaints (order_id, item_name, issue, requested_action) VALUES ('1006', NULL, 'late_delivery', 'status_update');"
    },
]

# ==========================================
# LOAD MODEL
# ==========================================
print("🚀 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")

# ==========================================
# FUNCTION: generate paraphrases
# ==========================================
def generate_paraphrases(text, num_variants=5):
    """Generate multiple natural language variations of a complaint."""
    prompt = f"""Paraphrase the following complaint in {num_variants} different ways.
Keep the meaning identical, but vary wording, tone, and style.
Output one variation per line. Do NOT include numbering.

Complaint: "{text}"."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.9,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    lines = [l.strip() for l in response.split("\n") if l.strip()]
    # clean unwanted echoes
    variations = [l for l in lines if not l.lower().startswith("complaint:")]
    return variations[:num_variants]

# ==========================================
# MAIN LOOP
# ==========================================
augmented_data = []
order_counter = 2000

for base in tqdm(base_examples, desc="Generating augmented samples"):
    for _ in range(NUM_VARIANTS_PER_SAMPLE):
        order_counter += 1
        # Generate paraphrases
        paraphrases = generate_paraphrases(base["complaint"], num_variants=1)
        for ptext in paraphrases:
            sql = base["sql"].replace(base["complaint"].split("order")[0], "").replace(
                base["sql"].split("'")[1], str(order_counter)
            )
            # Slightly modify SQL to new order_id
            sql = sql.replace(base["sql"].split("'")[1], str(order_counter))
            entry = {
                "text": f"<|system|>\nYou are a parser. Convert the given customer message into a SQL INSERT statement for the 'complaints' table with columns: order_id, item_name, issue, requested_action. No extra text.\n<|user|>\n{ptext}\n<|assistant|>\n{sql}"
            }
            augmented_data.append(entry)

# ==========================================
# SAVE OUTPUT
# ==========================================
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for ex in augmented_data:
        f.write(json.dumps(ex) + "\n")

print(f"\n✅ Generated {len(augmented_data)} augmented samples and saved to {OUTPUT_FILE}")
