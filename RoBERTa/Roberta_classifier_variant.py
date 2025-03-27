from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Example sleep disorder-related German text
text = "Ich schlafe kaum noch durch, wache ständig auf und bin morgens erschöpft."

# Load your 5-class fine-tuned model
model_name = "finetunedbaserobertaxlm/roberta-xlm-5class-model"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize input text
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

# Predict sentiment
with torch.no_grad():
    outputs = model(**inputs)

# Convert logits to probabilities
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# Define 5-stage sentiment labels
sentiment_classes = ['very negative', 'negative', 'neutral', 'positive', 'very positive']

# Output results
predicted_class = predictions.argmax().item()
print(f"Predicted sentiment: {sentiment_classes[predicted_class]}")
print("Class probabilities:")
for label, prob in zip(sentiment_classes, predictions[0]):
    print(f"{label:>15}: {prob.item():.4f}")
