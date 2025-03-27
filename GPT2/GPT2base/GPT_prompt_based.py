
import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load German GPT-2 model (fine-tuned or generic)
model_name = "benjamin/gpt2-wechsel-german"  # or your fine-tuned GPT-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.eval()

# Sentiment labels
labels = ["[very negative]", "[negative]", "[neutral]", "[positive]", "[very positive]"]

# Sample input
text = "Ich habe sehr schlecht geschlafen. Ich bin die ganze Nacht aufgewacht."

# Prompt for generation
prompt = f"Text: {text}\nSentiment:"

# Tokenize prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate output
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=5,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode and extract prediction
generated_text = tokenizer.decode(output_ids[0])
predicted_label = None
for label in labels:
    if label in generated_text:
        predicted_label = label
        break

print("Generated:", generated_text)
print("Predicted sentiment:", predicted_label or "Unknown")
