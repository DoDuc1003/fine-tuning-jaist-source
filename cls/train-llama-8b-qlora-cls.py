# Import necessary libraries and modules
import huggingface_hub
import wandb
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import DataCollatorWithPadding
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, LoraModel

# Log in to Hugging Face and WandB for experiment tracking
huggingface_hub.login("hf_api_key_here")  # Replace with your Hugging Face API key
wandb.login(key="wandb_api_key_here")     # Replace with your WandB API key

# 1. Load dataset from Hugging Face Datasets
id_dataset = "Kudod/Cognitive-Distortions"  # Dataset name
dataset = load_dataset(id_dataset, cache_dir=f'./cache/dataset/{id_dataset}')  # Load dataset

# Define the mapping of labels to integers and vice versa for classification tasks
label2id = {
    'All-or-Nothing Thinking': 0, 'Emotional Reasoning': 1, 'Fortune-telling': 2, 
    'Labeling': 3, 'Magnification': 4, 'Mental Filter': 5, 'Mind Reading': 6, 
    'No Distortion': 7, 'Overgeneralization': 8, 'Personalization': 9, 'Should Statements': 10
}
id2label = {v: k for k, v in label2id.items()}  # Reverse mapping for inference

# Model name: Using Llama model for sequence classification
model_name = "meta-llama/Llama-3.1-8B"  # Replace with the model you want to use

# Initialize tokenizer and set pad_token (important for padding)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=f'./cache/tokenizer/{model_name}')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token to eos_token if it's not defined

# 2. Tokenize the dataset
# This function is used to preprocess the text data by tokenizing the text and padding it to a fixed length
def preprocess_function(examples):
    tokenized = {
        'input_ids': [],
        'attention_mask': [],
        'labels': examples['label']  # Retain the labels from the dataset
    }
    
    # Iterate over the texts in the dataset and tokenize them
    for text in examples["text"]:
        try:
            encoding = tokenizer(
                text, 
                truncation=True, 
                padding='max_length',  # Pad to max length (512 tokens)
                max_length=512,  # Maximum sequence length
                return_tensors='pt'  # Return tensors in PyTorch format
            )
            tokenized['input_ids'].append(encoding['input_ids'].squeeze().tolist())  # Squeeze to remove extra dimensions
            tokenized['attention_mask'].append(encoding['attention_mask'].squeeze().tolist())  # Similarly for attention_mask
        except Exception as e:
            # If an error occurs during tokenization, handle it and use a default text
            print(f"Error processing text: {text}")
            print(f"Error: {e}")
            encoding = tokenizer(
                "No",  # Default text in case of an error
                truncation=True, 
                padding='max_length', 
                max_length=512,
                return_tensors='pt'
            )
            tokenized['input_ids'].append(encoding['input_ids'].squeeze().tolist())
            tokenized['attention_mask'].append(encoding['attention_mask'].squeeze().tolist())
    
    return tokenized

# Apply preprocessing to the entire dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)  # Tokenize the dataset in batches

# 3. Set up DataCollator for padding during batching
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # This will pad input sequences during training

# 4. Define the compute_metrics function for evaluation (precision, recall, f1 score)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # Get the predicted class for each example
    
    # Calculate precision, recall, and F1 score using weighted average for imbalance handling
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 5. BitsAndBytes configuration for 4-bit quantization (to reduce model size)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization for faster inference and smaller model size
    bnb_4bit_use_double_quant=True,  # Enable double quantization
    bnb_4bit_quant_type="nf4",  # Choose the quantization type (e.g., nf4)
    bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 precision for computation
)

# 6. Load the model with quantization configuration
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(label2id),  # Set the number of output labels
    label2id=label2id,  # Map label names to ids
    id2label=id2label,  # Reverse mapping for inference
    torch_dtype=torch.float16,  # Use float16 precision for training
    cache_dir=f'./cache/model/{model_name}',  # Cache directory for model files
    quantization_config=bnb_config  # Apply the quantization configuration
)

# 7. Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning of large models
peft_config = LoraConfig(
    lora_alpha=16,  # LoRA alpha scaling factor
    lora_dropout=0.1,  # Dropout rate for LoRA layers
    r=64,  # Rank of LoRA layers, can be adjusted depending on memory and model size
    bias="none",  # No bias in LoRA layers
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Apply LoRA to attention layers
    task_type="SEQ_CLS",  # Sequence classification task type
)

# Get the PEFT (Parameter Efficient Fine-Tuning) model with LoRA applied
model = get_peft_model(model, peft_config)

# 8. Check the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters
print(f"Total number of parameters: {total_params}")
print(f"Number of trainable parameters: {trainable_params}")

# 9. Define training arguments
training_args = TrainingArguments(
    output_dir='Cognitive-Distortions-Llama-3.1-8B-v2',  # Output directory for training results
    num_train_epochs=30,  # Number of training epochs
    per_device_train_batch_size=64,  # Batch size for training
    per_device_eval_batch_size=64,  # Batch size for evaluation
    warmup_steps=500,  # Number of warmup steps for learning rate scheduler
    weight_decay=0.01,  # Weight decay for regularization
    optim='adamw_torch_fused',  # Optimizer to use (AdamW)
    eval_strategy="epoch",  # Evaluate the model after each epoch
    save_strategy="epoch",  # Save the model after each epoch
    load_best_model_at_end=True,  # Load the best model after training
    metric_for_best_model="f1",  # Metric to use for selecting the best model
    greater_is_better=True,  # If F1 score increases, consider the model better
    push_to_hub=True,  # Push the model to Hugging Face Hub after training
    fp16_opt_level="O1",  # Use mixed precision training
    gradient_accumulation_steps=4,  # Gradient accumulation steps
    gradient_checkpointing=True,  # Use gradient checkpointing to save memory
)

# 10. Configure early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,  # Stop if the model doesn't improve after 2 epochs
    early_stopping_threshold=0.01,  # Minimum improvement threshold for early stopping
)

# 11. Set up the Trainer with all configurations
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],  # Training dataset
    eval_dataset=tokenized_dataset['dev'],  # Evaluation dataset
    tokenizer=tokenizer,  # Tokenizer for encoding data
    data_collator=data_collator,  # Data collator for padding
    compute_metrics=compute_metrics,  # Compute metrics during evaluation
    callbacks=[early_stopping_callback],  # Add early stopping callback
)

# 12. Fine-tune the model
trainer.train()  # Start training the model

# 13. Push the model to Hugging Face Hub
trainer.push_to_hub()  # Push the trained model to the Hugging Face Hub

