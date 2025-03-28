import csv
import math
import random
import torch
import time
import os
import numpy as np
import re
from sklearn.utils import shuffle
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead
import fasttext
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
import treetaggerwrapper
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from bertopic import BERTopic
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

# Setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tagger = treetaggerwrapper.TreeTagger(TAGLANG='de', TAGDIR='treetagger/')
german_stop_words = set(stopwords.words('german'))

# Load models
model_path = hf_hub_download(repo_id="facebook/fasttext-de-vectors", filename="model.bin")
model_fasttext = fasttext.load_model(model_path)
tokenizer_gpt = AutoTokenizer.from_pretrained("benjamin/gpt2-wechsel-german")
model_gpt = AutoModelWithLMHead.from_pretrained("benjamin/gpt2-wechsel-german")
tokenizer_perplex = AutoTokenizer.from_pretrained("distilbert-base-german-cased")
model_perplex = AutoModelWithLMHead.from_pretrained("distilbert-base-german-cased")

# Vocabulary table for guidance
vocab_size = len(tokenizer_gpt)
conv_table_gpt_vocab = np.zeros((vocab_size, 300))
for i in range(vocab_size):
    try:
        word = tokenizer_gpt.decode([i]).strip().lower()
        fast_vec = model_fasttext.get_word_vector(word)
        conv_table_gpt_vocab[i, :] = fast_vec
    except:
        conv_table_gpt_vocab[i, :] = np.zeros((300))

# Helper functions
def del_stop_words(text):
    words = nltk.word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in german_stop_words]
    return ' '.join(filtered_words)

def greedy_filtering(logits, filter_value=-float('Inf')):
    top_k = 1
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value
    return logits

def noguide(text, tokenizer, model, temperature=1.):
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    logits = model(tokens_tensor).logits[0, -1, :] / temperature
    logits = greedy_filtering(logits)
    logits = F.softmax(logits, dim=-1)
    predicted_index = torch.multinomial(logits, 1).item()
    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
    return predicted_text

def guidance(text, tokenizer, model, guide_word_lemma, fasttext_words, conv_table_gpt_vocab, weight, guide=True, temperature=1.):
    indexed_tokens = tokenizer.encode(text)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
    logits = model(tokens_tensor).logits[0, -1, :] / temperature
    logits_pro_guid_word, pred_indexes = [], []

    for k, fasttext_word in zip(guide_word_lemma, fasttext_words):
        sim = cosine_similarity(np.reshape(fasttext_word, (1, -1)), conv_table_gpt_vocab)
        sim_squared = np.clip(np.squeeze(sim), a_min=0, a_max=None) ** 2
        logits_temp = logits + torch.tensor(sim_squared * weight).cuda()
        logits_temp = greedy_filtering(logits_temp)
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
        if not guide_word_lemma:
            guide = False
    return predicted_text, guide, guide_word_lemma, fasttext_words

# Generate guided greedy text
number_of_words_per_sentence = 100
weight = 30
df_keywords = pd.read_csv("/keywords_list.tsv")
df_top_k = pd.DataFrame(columns=["sentiment", "text", "gen_text", "perplexity", "keywords", "num_keywords", "num_keywords_not_used"])

for i in range(0, 50):
    keywords = eval(df_keywords.loc[i, "keywords"])
    guide_word_lemma = [k.split('\t')[-1] for k in tagger.tag_text(keywords, tagonly=True)]
    fasttext_words = [model_fasttext.get_word_vector(k) for k in guide_word_lemma]
    guide_word_lemma_cp, fasttext_words_cp = guide_word_lemma.copy(), fasttext_words.copy()

    context = " Wir haben nicht so gut geschlafen."
    guide_next = True
    for j in range(number_of_words_per_sentence):
        if guide_next:
            context, guide_next, guide_word_lemma, fasttext_words = guidance(
                context, tokenizer_gpt, model_gpt,
                guide_word_lemma, fasttext_words, conv_table_gpt_vocab,
                weight, guide_next)
        else:
            context = noguide(context, tokenizer_gpt, model_gpt)

    # Evaluate
    generated_text = re.sub(r'\n', '', context)
    tokens = tokenizer_perplex.tokenize(generated_text)
    tensor_input = torch.tensor([tokenizer_perplex.convert_tokens_to_ids(tokens)])
    loss, _ = model_perplex(tensor_input, labels=tensor_input)[:2]

    df_top_k.loc[i] = [
        df_keywords.loc[i, "sent"],
        context,
        generated_text,
        math.exp(loss),
        guide_word_lemma_cp,
        len(guide_word_lemma_cp),
        len(guide_word_lemma)
    ]

# Save result
out_path = '/Results/greedy_search_gpt_5stage.tsv'
df_top_k.to_csv(out_path, sep="\t")
