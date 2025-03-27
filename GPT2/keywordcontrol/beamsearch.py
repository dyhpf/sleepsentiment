import os
import re
import csv
import math
import random
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelWithLMHead, pipeline
import fasttext
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch.nn.functional as F
import treetaggerwrapper
from bertopic import BERTopic

# Setup
nltk.download('punkt')
nltk.download('stopwords')
german_stop_words = set(stopwords.words('german'))

tagger = treetaggerwrapper.TreeTagger(TAGLANG='de', TAGDIR='treetagger/')

# Load FastText
model_fasttext = fasttext.load_model("cc.de.300.bin")  # Make sure file is locally available

# Load model and tokenizer
model_path = "models/german_gpt2_sft_2_epoch_rl_2epochs"
tokenizer_gpt = AutoTokenizer.from_pretrained(model_path)
model_gpt = AutoModelWithLMHead.from_pretrained(model_path)
model_gpt.to('cuda')
tokenizer_gpt.pad_token = tokenizer_gpt.eos_token

# Vocabulary conversion table
vocab_size = len(tokenizer_gpt)
conv_table_gpt_vocab = np.zeros((vocab_size, 300))
for i in range(vocab_size):
    word = tokenizer_gpt.decode([i]).strip().lower()
    try:
        vec = model_fasttext.get_word_vector(word)
        conv_table_gpt_vocab[i, :] = vec
    except:
        pass

# Load perplexity model
tokenizer_perplex = AutoTokenizer.from_pretrained("distilbert-base-german-cased")
model_perplex = AutoModelWithLMHead.from_pretrained("distilbert-base-german-cased").to("cuda")

# Utility Functions
def convert_sentiment(text):
    return 0 if text in ["[negative]", "NEGATIVE"] else 1

def del_stop_words(text):
    words = word_tokenize(text, language='german')
    return ' '.join([w for w in words if w.lower() not in german_stop_words])

def guidance(context, guide_words, guide_vecs, weight, top_k, top_p):
    input_ids = tokenizer_gpt.encode(context, return_tensors="pt").to("cuda")
    logits = model_gpt(input_ids).logits[0, -1, :]  # last token logits
    logits = logits / 1.0  # temperature

    best_idx = -1
    best_score = -1
    for word_vec in guide_vecs:
        sim = cosine_similarity(word_vec.reshape(1, -1), conv_table_gpt_vocab)[0]
        sim = np.clip(sim, 0, None)
        logits_mod = logits + torch.tensor(sim * weight).to("cuda")

        sorted_logits, sorted_indices = torch.sort(logits_mod, descending=True)
        logits_probs = F.softmax(sorted_logits, dim=-1)

        idx = torch.multinomial(logits_probs, 1).item()
        score = logits_mod[sorted_indices[idx]].item()

        if score > best_score:
            best_score = score
            best_idx = sorted_indices[idx].item()

    pred_token = tokenizer_gpt.decode([best_idx])
    return context + pred_token

def compute_perplexity(text):
    tokens = tokenizer_perplex.tokenize(text)
    input_ids = tokenizer_perplex.convert_tokens_to_ids(tokens)
    tensor_input = torch.tensor([input_ids]).to("cuda")
    loss = model_perplex(tensor_input, labels=tensor_input)[0]
    return math.exp(loss.item())

# Load keyword data
df_keywords = pd.read_csv("data/keywords_list.tsv", sep="\t")
df_beam = pd.DataFrame(columns=["sentiment", "text", "perplexity", "keywords", "num_keywords", "num_keywords_not_used"])

# Beam Search Generation
for j in range(50):
    sentiment = df_keywords.loc[j, "sent"]
    context = [sentiment + " Wir waren in diesem Hotel."]
    keywords = eval(df_keywords.loc[j, "keywords"])
    stemmed_keywords = [kw.split('\t')[-1] for kw in tagger.tag_text(keywords, tagonly=True)]
    keyword_vecs = [model_fasttext.get_word_vector(kw) for kw in stemmed_keywords]
    original_keywords = stemmed_keywords.copy()

    for _ in range(30):
        candidates = []
        for ctx in context:
            for _ in range(3):  # beam width
                new_ctx = guidance(ctx, stemmed_keywords, keyword_vecs, weight=30, top_k=0, top_p=0.5)
                candidates.append(new_ctx)
        context = sorted(candidates, key=compute_perplexity)[:3]

        last_text = context[0]
        lemma = [k.split("\t")[-1] for k in tagger.tag_text(word_tokenize(last_text), tagonly=True)]
        stemmed_keywords = [kw for kw in stemmed_keywords if kw not in lemma]
        keyword_vecs = [model_fasttext.get_word_vector(kw) for kw in stemmed_keywords]

        if not stemmed_keywords:
            break

    final_text = re.sub(r"\n", "", context[0])
    perplex = compute_perplexity(final_text)
    unused = len([kw for kw in original_keywords if kw not in final_text])

    df_beam.loc[len(df_beam)] = [sentiment, final_text, perplex, original_keywords, len(original_keywords), unused]

# Save output
df_beam.to_csv("results/beam_keyword_guided_generation.tsv", sep="\t", index=False)
print("Beam search keyword-guided generation completed and saved.")
