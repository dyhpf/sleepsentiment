import os
import pandas as pd
import numpy as np
import random
import math
import torch
import re
import nltk
import textstat
import matplotlib.pyplot as plt
import spacy
from collections import Counter
from rouge import Rouge
from nltk.corpus import stopwords
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, recall_score, precision_score, f1_score
)
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
from bertopic import BERTopic
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

# Setup
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("de_core_news_sm")
stop_words = set(stopwords.words('german'))

# ====== Configurable Paths ======
model_base_path = "models/"
results_path = "results/"
data_path = "data/test_for_sft.tsv"
sentiment_model_path = os.path.join(model_base_path, "sentiment_discriminator_bert_finetuned")

# ====== Load Classifier ======
classifier = pipeline("sentiment-analysis", model=sentiment_model_path, top_k=None, function_to_apply="none")

# ====== Preprocessing Functions ======
def convert_sentiment(text):
    return 0 if text in ["[negative]", "NEGATIVE"] else 1

def convert_to_sentiment(x):
    return "[negative]" if x == 0 else "[positive]"

def take_10_words(text):
    return " ".join(text.split()[:10])

def del_stop_words(text):
    words = nltk.word_tokenize(text, language="german")
    return ' '.join([w for w in words if w.lower() not in stop_words])

# ====== Scoring Functions ======
def calculate_bleu_score(row):
    return sentence_bleu([row['org_text'].split()], row['text'].split())

def calculate_rouge_score(row):
    rouge = Rouge()
    scores = rouge.get_scores(str(row['text']), str(row['org_text']))
    return scores[0]["rouge-1"]["f"]

def slor_score(text, tokenizer_ref, model_ref):
    text = text.strip()
    slor_scores = []
    for sent in nlp(text).sents:
        sentence = sent.text
        tokenized_input = tokenizer_ref.encode(sentence, return_tensors="pt", truncation=True)
        loss, logits = model_ref(tokenized_input, labels=tokenized_input)[:2]
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]
        sentence_prob = np.prod([p.max().item() for p in probs])
        words = re.findall(r'\b\w+\b', sentence.lower())
        freqs = Counter(words)
        total_words = len(words)
        unigram_prob = np.prod([freq / total_words for freq in freqs.values()])
        if total_words > 0:
            slor = math.log(sentence_prob) / total_words - math.log(unigram_prob) / total_words
            slor_scores.append(slor)
    return np.mean(slor_scores) if slor_scores else 0

def lda_coherence_score(text):
    data_words = [w for w in nltk.word_tokenize(text, language="german") if w.lower() not in stop_words]
    bigrams = gensim.models.Phrases([data_words], min_count=5, threshold=100)
    bigram_mod = gensim.models.phrases.Phraser(bigrams)
    data_bigrams = [bigram_mod[data_words]]
    lemmatized = [[token.lemma_ for token in nlp(word)] for word in data_bigrams[0]]
    id2word = corpora.Dictionary(lemmatized)
    corpus = [id2word.doc2bow(text) for text in lemmatized]
    lda = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=4,
                                  random_state=100, chunksize=100, passes=20)
    coherence_model = CoherenceModel(model=lda, texts=lemmatized, dictionary=id2word, coherence='c_v')
    return coherence_model.get_coherence()

# ====== Generation Function ======
def generation(data, tokenizer, tokenizer_perplex, model, model_perplex, iterations):
    df = pd.DataFrame(columns=["sentiment", "text", "org_text", "perplexity",
                               "sent_predicted_label", "sent_predicted_score",
                               "sentiment_conv", "sent_predicted_label_conv"])

    for i in range(iterations):
        prompt = data.loc[i, "instruction"]
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, do_sample=True, top_p=0.5, pad_token_id=tokenizer.eos_token_id, max_new_tokens=100)
        generated_text = tokenizer.decode(output[0]).replace("\n", "")

        tokenize_input = tokenizer_perplex.tokenize(generated_text)
        tensor_input = torch.tensor([tokenizer_perplex.convert_tokens_to_ids(tokenize_input)])
        loss, _ = model_perplex(tensor_input, labels=tensor_input)[:2]

        pred = classifier(generated_text)
        df.loc[i] = {
            "sentiment": data.loc[i, "sent_str"],
            "text": generated_text,
            "org_text": data.loc[i, "preprocessed_text"],
            "perplexity": math.exp(loss),
            "sent_predicted_label": pred[0][0]["label"],
            "sent_predicted_score": pred[0][0]["score"],
            "sentiment_conv": convert_sentiment(data.loc[i, "sent_str"]),
            "sent_predicted_label_conv": convert_sentiment(pred[0][0]["label"])
        }
        df.loc[i, 'text_del_stop_words'] = del_stop_words(df.loc[i, 'text'])
    return df

# ====== Evaluation Function ======
def evaluate(df, tokenizer_ref, model_ref):
    df["slor"] = df["text"].apply(lambda x: slor_score(x, tokenizer_ref, model_ref))
    df["readability_score"] = df["text"].apply(textstat.flesch_reading_ease)
    df["text_processed"] = df["text"].str.lower().str.replace(r'[,\.!?]', '', regex=True)
    df["lda_score"] = df["text_processed"].apply(lda_coherence_score)
    df["rouge_score_f"] = df.apply(calculate_rouge_score, axis=1)

    y_true = df["sentiment_conv"]
    y_pred = df["sent_predicted_label_conv"]

    print(f"SLOR mean: {df['slor'].mean():.3f}")
    print(f"FRE mean: {df['readability_score'].mean():.3f}")
    print(f"LDA Coherence mean: {df['lda_score'].mean():.3f}")
    print(f"Perplexity mean: {df['perplexity'].mean():.3f}")
    print(f"ROUGE F1 mean: {df['rouge_score_f'].mean():.3f}")

    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred, average=None).mean())
    print("Precision:", precision_score(y_true, y_pred, average=None).mean())
    print("F1 score:", f1_score(y_true, y_pred, average=None).mean())

# ====== BERTopic Modeling ======
def bert_modeling(texts):
    topic_model = BERTopic(language="multilingual")
    topics, _ = topic_model.fit_transform(texts)
    print(topic_model.get_topic_info())
    print(topic_model.get_topics())

# ====== Main Process ======
if __name__ == "__main__":
    data = pd.read_csv(data_path, sep="\t")
    data["sent_str"] = data["sentiment"].apply(convert_to_sentiment)
    data["seq_start"] = data["preprocessed_text"].apply(take_10_words)
    data["instruction"] = data["sent_str"] + " " + data["seq_start"]

    sampled = pd.concat([
        data[data["sentiment"] == 1].sample(n=50, random_state=123),
        data[data["sentiment"] == 0].sample(n=50, random_state=123)
    ]).sample(frac=1, random_state=42).reset_index()

    # Load evaluation/tokenizer models
    tokenizer_perplex = AutoTokenizer.from_pretrained("distilbert-base-german-cased")
    model_perplex = AutoModelWithLMHead.from_pretrained("distilbert-base-german-cased")
    tokenizer_ref = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-cased")
    model_ref = AutoModelWithLMHead.from_pretrained("dbmdz/bert-base-german-cased")

    # ===== Evaluate a fine-tuned model =====
    model_path = os.path.join(model_base_path, "german_gpt2_sft_3_epochs")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelWithLMHead.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    df_generated = generation(sampled, tokenizer, tokenizer_perplex, model, model_perplex, 100)
    evaluate(df_generated, tokenizer_ref, model_ref)

    # Optional: save results
    df_generated.to_csv(os.path.join(results_path, "evaluation_sft3.csv"), sep="\t", index=False)

    # Topic modeling
    bert_modeling(df_generated["text_del_stop_words"])
