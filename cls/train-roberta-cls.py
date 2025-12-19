# ============================================================
# 0. IMPORT LIBRARIES
# ============================================================

import numpy as np
import torch
import wandb
import huggingface_hub

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import precision_recall_fscore_support


# ============================================================
# 1. AUTHENTICATION (HuggingFace & WandB)
# ============================================================

huggingface_hub.login("hf_key")
wandb.login(key="wandb_key")


# ============================================================
# 2. LOAD DATASET
# ============================================================

DATASET_NAME = "Kudod/Cognitive-Distortions"

dataset = load_dataset(
    DATASET_NAME,
    cache_dir=f"./cache/dataset/{DATASET_NAME}"
)


# ============================================================
# 3. LABEL MAPPING
# ============================================================

label2id = {
    "All-or-Nothing Thinking": 0,
    "Emotional Reasoning": 1,
    "Fortune-telling": 2,
    "Labeling": 3,
    "Magnification": 4,
    "Mental Filter": 5,
    "Mind Reading": 6,
    "No Distortion": 7,
    "Overgeneralization": 8,
    "Personalization": 9,
    "Should Statements": 10,
}

id2label = {v: k for k, v in label2id.items()}
NUM_LABELS = len(label2id)


# ============================================================
# 4. TOKENIZER
# ============================================================

MODEL_NAME = "FacebookAI/roberta-large"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=f"./cache/tokenizer/{MODEL_NAME}",
)

# Optional (if pad_token is missing)
# tokenizer.pad_token = tokenizer.eos_token


# ============================================================
# 5. PREPROCESSING FUNCTION
# ============================================================

def preprocess_function(examples):
    """
    Tokenize input texts and keep labels unchanged.
    Includes fallback handling for problematic samples.
    """
    input_ids = []
    attention_masks = []

    for text in examples["text"]:
        try:
            encoding = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
        except Exception as e:
            print(f"[Warning] Tokenization failed for text: {text}")
            print(f"Error: {e}")
            encoding = tokenizer(
                "No",
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )

        input_ids.append(encoding["input_ids"].squeeze().tolist())
        attention_masks.append(encoding["attention_mask"].squeeze().tolist())

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": examples["label"],
    }


tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)


# ============================================================
# 6. DATA COLLATOR
# ============================================================

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ============================================================
# 7. METRICS
# ============================================================

def compute_metrics(eval_pred):
    """
    Compute weighted Precision, Recall, and F1-score.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# 8. MODEL INITIALIZATION
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    label2id=label2id,
    id2label=id2label,
    torch_dtype=torch.float16,
    cache_dir=f"./cache/model/{MODEL_NAME}",
)


# ============================================================
# 9. TRAINING ARGUMENTS
# ============================================================

training_args = TrainingArguments(
    output_dir="Cognitive-Distortions-Robertta-Large-v1",
    num_train_epochs=30,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    gradient_accumulation_steps=4,
    warmup_steps=500,
    weight_decay=0.01,
    optim="adamw_torch_fused",

    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,

    fp16=True,
    gradient_checkpointing=True,

    logging_dir="./logs",
    report_to="wandb",

    push_to_hub=True,
)


# ============================================================
# 10. EARLY STOPPING
# ============================================================

early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.01,
)


# ============================================================
# 11. TRAINER
# ============================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["dev"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping],
)


# ============================================================
# 12. TRAIN & PUSH TO HUB
# ============================================================

trainer.train()
trainer.push_to_hub()
