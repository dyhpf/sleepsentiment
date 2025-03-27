import pandas as pd
import random
from datasets import Dataset
from transformers import (
    AutoModelWithLMHead,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer
import nltk

# Setup
nltk.download("punkt")

# Optional: pip installation commands to run manually
# pip install transformers datasets trl peft

# File paths (replace with your actual local paths)
train_path = "data/train_for_sft.tsv"
val_path = "data/val_for_sft.tsv"
model_save_path = "models/german_gpt2_sft_3_epochs"
pretrained_model_name = "benjamin/gpt2-wechsel-german"

# Load and prepare data
def sentiment_map(x):
    return "positive" if x == 1 else "negative"

def create_instruction(sent, text):
    """
    Format training input as "[sentiment] text", truncated to 512 words.
    """
    text = f"[{sent}] {text}"
    words = text.split()
    return " ".join(words[:512])

# Load and process training data
data = pd.read_csv(train_path, sep='\t')[["preprocessed_text", "sentiment"]]
data["sentiment"] = data["sentiment"].apply(sentiment_map)
data["instructions"] = data.apply(lambda row: create_instruction(row['sentiment'], row['preprocessed_text']), axis=1)

# Create Hugging Face dataset
dataset = Dataset.from_pandas(data)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
model = AutoModelWithLMHead.from_pretrained(pretrained_model_name)
tokenizer.pad_token = tokenizer.eos_token

# Training configuration
training_args = TrainingArguments(
    output_dir="outputs",  # temporary output during training
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    logging_steps=50,
    save_strategy="no"
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    dataset_text_field="instructions",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args
)

# Train
trainer.train()

# Save the trained model
trainer.save_model(model_save_path)

# ========================
# Inference Example
# ========================
# Load fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_save_path)
model = AutoModelWithLMHead.from_pretrained(model_save_path)
tokenizer.pad_token = tokenizer.eos_token

# Example prompt
prompt_sent = "[negative]: Obwohl ich mir Mühe gegeben habe,"

# Generation settings
generation_kwargs = {
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 56,
    "num_beams": 4,
    "no_repeat_ngram_size": 3,
    "num_return_sequences": 1
}

# Generate response
input_ids = tokenizer.encode(prompt_sent, return_tensors="pt")
output = model.generate(input_ids, **generation_kwargs)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)

