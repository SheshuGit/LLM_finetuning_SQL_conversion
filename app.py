import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, re

# === Model Path ===
MODEL_DIR = "merged_model_4bit"  # Change if needed

# === Load Model & Tokenizer ===
print("🚀 Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)

# === Define Prompt Template ===
SYSTEM_PROMPT = """<|system|>
You are an SQL generator. Given a user complaint, output a single SQL INSERT statement 
into 'complaints(order_id, item_name, issue, requested_action)'.
No explanations or code. Output only the SQL statement.
"""

# === SQL Generation Function ===
def generate_sql(complaint):
    prompt = f"{SYSTEM_PROMPT}\n<|user|>\n{complaint}"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract SQL
    sql_match = re.search(r"INSERT\s+INTO.*?;", result, re.IGNORECASE | re.DOTALL)
    clean_sql = sql_match.group(0).strip() if sql_match else result.strip()

    return clean_sql


# === Gradio UI ===
title = "🧠 SQL Complaint Parser"
description = "Enter a user complaint below — the fine-tuned LLM will generate a SQL INSERT query automatically."

iface = gr.Interface(
    fn=generate_sql,
    inputs=gr.Textbox(
        label="Enter Customer Complaint",
        placeholder="e.g. Hi, my order 2234 arrived but the phone case is cracked. Please replace it.",
        lines=4,
    ),
    outputs=gr.Textbox(label="Generated SQL Query", lines=4),
    title=title,
    description=description,
    theme="soft",
    examples=[
        ["Hi, my order 2234 arrived but the phone case is cracked. Please replace it."],
        ["My perfume order 4455 leaked inside the box, need replacement."],
        ["Order 7782 was delayed by 5 days. I want a refund."],
        ["The laptop I ordered (order 2020) came without the charger. Please resend it."],
    ],
)

if __name__ == "__main__":
    iface.launch(share=True)
