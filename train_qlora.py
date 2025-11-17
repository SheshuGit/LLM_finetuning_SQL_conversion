import torch, gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from pathlib import Path

# 🧠 lightweight model
BASE_MODEL = "microsoft/Phi-2"

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading Phi-2 with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.config.use_cache = False

train = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"))["train"]
val = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"))["train"]

peft_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)

train_cfg = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    bf16=False,
    fp16=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_total_limit=2,
    report_to=["tensorboard"],
    group_by_length=True,
    gradient_checkpointing=True,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    eval_dataset=val,
    peft_config=peft_cfg,
    args=train_cfg,
    dataset_text_field="text",
)

trainer.train()
print("✅ Training completed on RTX 3050 (4 GB)")
