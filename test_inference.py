from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, re

MODEL_DIR = "merged_model_4bit"

print("🚀 Loading fine-tuned model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# ✅ Fix padding issue
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    device_map="auto",
)

prompt = """<|system|>
You are an SQL generator. Given a user complaint, output a single SQL INSERT statement 
into 'complaints(order_id, item_name, issue, requested_action)'.
No explanations or code. Output only the SQL statement.

<|user|>
Hi, my order 2234 arrived but the phone case is cracked. please replace it.
"""

# ✅ Padding and truncation now safe
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")

print("🧠 Generating SQL query...")
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.2,
        do_sample=False,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)

sql_match = re.search(r"INSERT\s+INTO.*?;", result, re.IGNORECASE | re.DOTALL)
clean_sql = sql_match.group(0).strip() if sql_match else result.strip()

print("\n✅ Final SQL Output:\n", clean_sql)
