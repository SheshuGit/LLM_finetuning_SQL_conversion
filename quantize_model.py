# quantize_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_PATH = "merged_model"
QUANTIZED_PATH = "./merged_model_4bit"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("⚙️ Quantizing merged model to 4-bit...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model.save_pretrained(QUANTIZED_PATH)
tokenizer.save_pretrained(QUANTIZED_PATH)
print(f"✅ Quantized model saved to {QUANTIZED_PATH}")
