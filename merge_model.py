from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os, gc

BASE_MODEL = "microsoft/Phi-2"
ADAPTER_DIR = "./output/checkpoint-6"
MERGED_DIR = "./merged_model"
OFFLOAD_DIR = "./offload_tmp"   # 👈 temporary folder for layer offloading

os.makedirs(OFFLOAD_DIR, exist_ok=True)

print("🔄 Merging LoRA adapter into base model (with offloading)...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    offload_folder=OFFLOAD_DIR,   # 👈 key line
)

peft_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    device_map="auto",
    offload_folder=OFFLOAD_DIR,   # 👈 key line
)

merged = peft_model.merge_and_unload()
merged.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.save_pretrained(MERGED_DIR)

print(f"✅ Merged model saved successfully at: {MERGED_DIR}")

# cleanup
del base_model, peft_model, merged
gc.collect()
torch.cuda.empty_cache()
