
# 🧠 SQL Complaint Parser using LLM Fine-tuning (QLoRA)

> 🚀 An AI-powered system that converts **unstructured customer complaints** into **structured SQL INSERT statements** automatically.

This project fine-tunes a lightweight **language model (Phi-2 / LLaMA)** using **QLoRA (Quantized Low-Rank Adaptation)** to understand natural-language customer complaints and generate accurate SQL queries that can be directly inserted into a database.

---

## 📘 Project Overview

Modern e-commerce platforms, logistics systems, and customer support centers receive thousands of customer complaints daily — most written in **unstructured text** like:

> “Hi, my order 2234 arrived but the phone case is cracked. Please replace it.”

This project automates the process by **parsing complaints → extracting details → generating SQL insert queries** for structured storage in the backend system.

---

## ⚙️ System Workflow

```
User Complaint  →  Fine-tuned LLM  →  SQL INSERT Query  →  Database (complaints table)
```

---

## ✨ Features Implemented

✅ **1. Data Generation**

* 300+ realistic complaint samples covering various product issues.
* Each sample includes text + expected SQL structure.
* Data stored in `.jsonl` format for training and golden evaluation.

✅ **2. Model Fine-Tuning (QLoRA)**

* Used **Phi-2 (2.7B)** and **LLaMA 3.2 (optional)** with 4-bit quantization.
* Trained using **Hugging Face TRL `SFTTrainer`**.
* Optimized for GPU efficiency (runs on RTX 3050 4GB 💪).

✅ **3. Model Merging**

* Adapter weights merged with base model using `PeftModel.merge_and_unload()`.
* Quantized 4-bit merged model saved for inference.

✅ **4. Inference Pipeline**

* Converts any user complaint into SQL query:

```sql
INSERT INTO complaints (order_id, item_name, issue, requested_action)
VALUES ('2234', 'phone case', 'damaged_item', 'replacement');
```

✅ **5. Gradio UI**

* User-friendly interface for real-time complaint-to-SQL generation.
* Supports instant preview of generated SQL queries.

✅ **6. Evaluation Framework**

* Golden dataset-based evaluation (`golden_data.jsonl`).
* Metrics:

  * Strict Accuracy ✅
  * Semantic Accuracy ✅
  * Fuzzy Similarity ✅
  * GPT-based Judgment (optional with OpenAI key) 🤖
* Detailed CSV report generation with accuracy per case.

✅ **7. Model Evaluation Testing**

* Added script `evaluate_model.py` for structured evaluation.
* Outputs visual comparison between expected and generated SQL queries.

✅ **8. Error Analysis**

* Automatically calculates similarity %
* Identifies consistent patterns like “leaked → damaged_item”
* Guided retraining suggestions for normalization.

---

## 🧰 Tech Stack

| Component                 | Technology                                    |
| ------------------------- | --------------------------------------------- |
| **Model Base**            | Phi-2 (Microsoft) / LLaMA 3.2                 |
| **Fine-tuning Framework** | 🤗 Hugging Face Transformers + TRL            |
| **Quantization**          | BitsAndBytes (4-bit QLoRA)                    |
| **Dataset Format**        | JSONL (instruction-style data)                |
| **UI Interface**          | Gradio                                        |
| **Evaluation Metrics**    | FuzzyWuzzy, GPT Judge, Semantic Normalization |
| **Environment**           | Python 3.10+, PyTorch, CUDA, RTX 3050         |

---

## 📂 Repository Structure

```
sql_complaint_parser/
│
├── data/
│   ├── train.jsonl                # Training data
│   ├── golden_data.jsonl          # Evaluation dataset
│
├── train_qlora.py                 # Fine-tuning script (QLoRA)
├── merge_model.py                 # Merge adapter + base model
├── test_inference.py              # Test single complaint
├── evaluate_model.py              # Evaluate against golden data
├── gradio_app.py                  # Interactive complaint-to-SQL UI
├── evaluate_full.py               # Full evaluator with GPT + fuzzy metrics
│
├── merged_model_4bit/             # Quantized merged model (inference ready)
│
└── README.md                      # (this file)
```

---

## 🧩 Sample Workflow

### 🧠 Input

```
Hi, my order 2234 arrived but the phone case is cracked. Please replace it.
```

### 🤖 Model Output

```sql
INSERT INTO complaints (order_id, item_name, issue, requested_action)
VALUES ('2234', 'phone case', 'damaged_item', 'replacement');
```

---

## 🧪 Evaluation Results (Example Run)

| Metric            | Score |
| ----------------- | ----- |
| Strict Accuracy   | 48.5% |
| Semantic Accuracy | 87.2% |
| Fuzzy Similarity  | 95.3% |
| GPT Judge Score   | 91.7% |

🔍 Most mismatches come from synonyms:

* `"leaked"` vs `"damaged_item"`
* `"wrong color"` vs `"wrong_item"`
* `"resend"` vs `"replacement"`

---

## 🖥️ Gradio UI Preview

```python
import gradio as gr

def complaint_to_sql(complaint):
    # returns generated SQL from fine-tuned model
    ...

gr.Interface(fn=complaint_to_sql, 
             inputs="text", 
             outputs="text", 
             title="🧠 SQL Complaint Generator",
             description="Enter your customer complaint below").launch()
```

💡 Allows direct testing of unseen complaints.

---

## 🌍 Real-Time Application Scope

### 🎯 **1. E-commerce Platforms**

Automatically logs complaints into SQL databases to trigger refunds/replacements.

### 💬 **2. Customer Support Automation**

Integrate with chatbots (Zendesk, Freshdesk, etc.) to classify complaints and auto-generate support tickets.

### 🏦 **3. Banking / Insurance**

Parse customer transaction issues into SQL for fraud or refund investigation systems.

### 🏥 **4. Healthcare Platforms**

Detect and log patient complaints regarding prescriptions or medical orders.

### 🚚 **5. Logistics / Delivery**

Extract “delay”, “damage”, or “missing” events from messages and log in shipment systems.

---

## 🔮 Future Enhancements

| Area                       | Description                                                                            |
| -------------------------- | -------------------------------------------------------------------------------------- |
| **Synonym Normalization**  | Improve accuracy by mapping words like “leaked”, “cracked”, “broken” → `damaged_item`. |
| **Multi-lingual Support**  | Extend model to handle Hindi, Telugu, Tamil complaints.                                |
| **REST API Integration**   | Create Flask/FastAPI endpoint for production integration.                              |
| **RAG Integration**        | Retrieve similar historical complaints for consistency.                                |
| **Confidence Scoring**     | Output a “reliability” percentage for each query.                                      |
| **Continuous Fine-tuning** | Retrain periodically with new complaint logs.                                          |

---

## 🧠 Example Commands

**Fine-tune the model**

```bash
python train_qlora.py
```

**Merge adapters with base model**

```bash
python merge_model.py
```

**Run inference**

```bash
python test_inference.py
```

**Evaluate on golden data**

```bash
python evaluate_model.py
```

**Launch Gradio UI**

```bash
python app.py
```

---

## 📊 Example Golden Dataset (Excerpt)

```json
{"complaint": "my perfume (order 3011) leaked. refund pls.",
 "expected_sql": "INSERT INTO complaints (order_id, item_name, issue, requested_action) VALUES ('3011', 'perfume', 'damaged_item', 'refund');"}

{"complaint": "order id 4747, wrong size shoes",
 "expected_sql": "INSERT INTO complaints (order_id, item_name, issue, requested_action) VALUES ('4747', 'shoes', 'wrong_item', 'exchange');"}
```

---

## 🧩 Citation / Acknowledgements

* Microsoft [Phi-2 Model](https://huggingface.co/microsoft/phi-2)
* Hugging Face Transformers + TRL
* PEFT + BitsAndBytes for QLoRA
* Gradio for interactive UI
* FuzzyWuzzy / GPT Judge for evaluation

---

## 👨‍💻 Contributors

**Developer:** [Sheshu Enabothula](https://github.com/)
**Role:** Machine Learning & AI Developer
**Stack:** Python | PyTorch | Hugging Face | LLMs | Flask | SQL
