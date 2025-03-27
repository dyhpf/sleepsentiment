import os
import time
import torch
import random
import nltk
import wandb
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datasets import Dataset
from random import choices
from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    pipeline,
    BertTokenizer
)
from trl import (
    PPOTrainer,
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    create_reference_model
)

# Setup
nltk.download('punkt')
nltk.download('stopwords')
spacy.load('de_core_news_md')
tqdm.pandas()
torch.manual_seed(1)
np.random.seed(1)

# Hyperparameters and paths
txt_in_len = 5
txt_out_len = 20
seed = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
local_model_path = "models/german_gpt2_sft"
local_classifier_path = "models/sentiment_discriminator_bert_finetuned"
data_path = "data/train_for_sft.tsv"
output_path = "models/ppo_gpt2_rl"

# Install needed libraries
# Run this once manually or use subprocess in code if automation is necessary
# pip install wandb datasets trl transformers
# python -m spacy download de_core_news_md

# Initialize WandB
wandb.login()

# Load PPO config
config = PPOConfig(
    model_name=local_model_path,
    mini_batch_size=16,
    steps=51200,
    learning_rate=1.41e-5,
    remove_unused_columns=False,
    log_with="wandb"
)

# Load model and tokenizer
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
model_ref = create_reference_model(model)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess data
data = pd.read_csv(data_path, sep='\t')[["preprocessed_text", "sentiment"]]
dataset = Dataset.from_pandas(data)
dataset = dataset.filter(lambda x: len(x["preprocessed_text"]) > 500)
dataset = dataset.map(lambda x: {"preprocessed_text": x["preprocessed_text"][:1000]})
dataset = dataset.map(lambda x: {
    "input_ids": tokenizer.encode(" " + x["preprocessed_text"], return_tensors="pt", truncation=True, max_length=1024)[0, :txt_in_len]
})
dataset = dataset.map(lambda x: {"query": tokenizer.decode(x["input_ids"])})
dataset = dataset.shuffle(seed=42)[:14770]
dataset.set_format("pytorch")

# Define collator
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# Setup PPO trainer
ppo_trainer = PPOTrainer(config, model, model_ref, tokenizer, dataset, data_collator=collator)

# Control tokens
ctrl = ["[negative]", "[positive]"]
ctrl_tokens = {c: tokenizer.encode(c, return_tensors="pt").squeeze().to(device) for c in ctrl}

# Sentiment classifier
classifier = pipeline("sentiment-analysis", model=local_classifier_path, top_k=None, function_to_apply="none")

# Helper functions
def get_logits(texts):
    scores_texts = []
    for text in texts:
        output = classifier(text)[0]
        score_dict = {item['label']: item['score'] for item in output}
        negative_score = score_dict.get('NEGATIVE', 0.0)
        positive_score = score_dict.get('POSITIVE', 0.0)
        scores_texts.append([negative_score, positive_score])
    return scores_texts

def logit_to_reward(logit, task):
    scores = []
    for i in range(len(logit)):
        if task[i] == "[negative]":
            scores.append(logit[i][0])
        elif task[i] == "[positive]":
            scores.append(logit[i][1])
    return [torch.tensor(score, dtype=torch.float32) for score in scores]

# Generation settings
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 30
}

# PPO Training loop
for epoch in range(2):
    for batch in ppo_trainer.dataloader:
        tasks = choices(ctrl, k=config.batch_size)
        query_tensors = [torch.cat((ctrl_tokens[task], input_ids)) for task, input_ids in zip(tasks, batch["input_ids"])]

        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(query, **generation_kwargs, return_prompt=True)
            response_tensors.append(response.squeeze()[-30:])

        response = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        texts = [q + r for q, r in zip(batch["query"], response)]

        logits = get_logits(texts)
        rewards = logit_to_reward(logits, tasks)

        torch.cuda.empty_cache()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

# Save final models
ppo_trainer.save_pretrained(output_path)
model_ref.save_pretrained(output_path + "_ref")
model.save_pretrained(output_path + "_base")

# Inference example
def generate_example(prompt="[negative]"):
    tokenizer_test = AutoTokenizer.from_pretrained(output_path)
    model_test = AutoModelWithLMHead.from_pretrained(output_path)
    tokenizer_test.pad_token = tokenizer_test.eos_token_id

    input_ids = tokenizer_test.encode(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = model_test.generate(input_ids, **generation_kwargs)
    print("Generated Text:\n", tokenizer_test.decode(output[0]))
