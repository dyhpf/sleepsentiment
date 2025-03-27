import os
import re
import math
import torch
import random
import fasttext
import numpy as np
import pandas as pd
import nltk
import treetaggerwrapper
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score
from transformers import AutoTokenizer, pipeline, AutoModelWithLMHead
from huggingface_hub import hf_hub_download
from bertopic import BERTopic
import torch.nn.functional as F
import matplotlib.pyplot as plt

# NLTK setup
nltk.download("stopwords")
nltk.download("punkt")
german_stop_words = set(nltk.corpus.stopwords.words("german"))

# TreeTagger setup
tagger = treetaggerwrapper.TreeTagger(TAGLANG="de", TAGDIR="treetagger/")

# Utility functions
sentiment_map = {
    "[very negative]": 0,
    "[negative]": 1,
    "[neutral]": 2,
    "[positive]": 3,
    "[very positive]": 4,
    "VERY_NEGATIVE": 0,
    "NEGATIVE": 1,
    "NEUTRAL": 2,
    "POSITIVE": 3,
    "VERY_POSITIVE": 4,
}

reverse_sentiment_map = {
    0: "[very negative]",
    1: "[negative]",
    2: "[neutral]",
    3: "[positive]",
    4: "[very positive]"
}

def convert_sentiment(text):
    return sentiment_map.get(text, 2)

def convert_to_sentiment(x):
    return reverse_sentiment_map.get(x, "[neutral]")

def del_stop_words(text):
    words = nltk.word_tokenize(text)
    return ' '.join([word for word in words if word.lower() not in german_stop_words])

def top_k_filtering(logits, top_k=0, filter_value=-float('Inf')):
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    return logits

def noguide(text, tokenizer, model, top_k=0, temperature=1.):
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    logits = model(tokens_tensor).logits[0, -1, :] / temperature
    logits = top_k_filtering(logits, top_k=top_k)
    probs = F.softmax(logits, dim=-1)
    predicted_index = torch.multinomial(probs, 1).item()
    return tokenizer.decode(indexed_tokens + [predicted_index])

def guidance(text, tokenizer, model, guide_word_lemma, fasttext_words, conv_table_gpt_vocab, weight, guide=True, top_k=None, temperature=1., only_max=False):
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    logits = model(tokens_tensor).logits[0, -1, :] / temperature
    logits_pro_guid_word = []
    pred_indexes = []

    for lemma, ft_vec in zip(guide_word_lemma, fasttext_words):
        sim = cosine_similarity(ft_vec.reshape(1, -1), conv_table_gpt_vocab)
        sim_squared = np.clip(np.squeeze(sim), a_min=0, a_max=None) ** 2
        logits_temp = logits + torch.tensor(sim_squared * weight).cuda()
        logits_temp = top_k_filtering(logits_temp, top_k=top_k)
        logits_temp = F.softmax(logits_temp, dim=-1)
        predicted_index = torch.multinomial(logits_temp, 1).item()
        logits_pro_guid_word.append(logits[predicted_index].item())
        pred_indexes.append(predicted_index)

    predicted_index = pred_indexes[np.argmax(logits_pro_guid_word)]
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    pred_word = predicted_text.split()[-1]
    pred_word_lemma = tagger.tag_text(pred_word, tagonly=True)[0].split('\t')[-1]

    if pred_word_lemma in guide_word_lemma:
        idx = guide_word_lemma.index(pred_word_lemma)
        guide_word_lemma.pop(idx)
        fasttext_words.pop(idx)
        guide = len(guide_word_lemma) > 0

    return predicted_text, guide, guide_word_lemma, fasttext_words

# Load models and resources
fasttext_path = hf_hub_download(repo_id="facebook/fasttext-de-vectors", filename="model.bin")
model_fasttext = fasttext.load_model(fasttext_path)

model_gpt = AutoModelWithLMHead.from_pretrained("benjamin/gpt2-wechsel-german").to('cuda')
tokenizer_gpt = AutoTokenizer.from_pretrained("benjamin/gpt2-wechsel-german")

model_perplex = AutoModelWithLMHead.from_pretrained("distilbert-base-german-cased")
tokenizer_perplex = AutoTokenizer.from_pretrained("distilbert-base-german-cased")

# Create GPT vocab -> FastText vector mapping
vocab_size = len(tokenizer_gpt)
conv_table_gpt_vocab = np.zeros((vocab_size, 300))
for i in range(vocab_size):
    try:
        word = tokenizer_gpt.decode([i]).strip().lower()
        conv_table_gpt_vocab[i, :] = model_fasttext.get_word_vector(word)
    except:
        pass

# Placeholder for loading and running guided generation (omitted for brevity)
# You can insert generation loops and evaluation using DataFrames like `df_top_k`

print("✅ Script ready. Insert generation loop and evaluation code below if needed.")
