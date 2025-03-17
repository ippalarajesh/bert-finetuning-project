from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = './outputs/fine_tuned_bert'  # Path to the directory containing the fine-tuned model
tokenizer_path = './outputs/tokenizer'    # Path to the directory containing the tokenizer

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

# Function to make predictions on new text
def predict(texts):
    # Tokenize the input texts
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)

    # Pass the tokenized inputs to the model to get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Convert logits to probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class label
    predicted_class = torch.argmax(probabilities, dim=-1).item()

    # Return the predicted class and probabilities
    return predicted_class, probabilities

# Example usage: Predict for new text
texts = ["This movie is amazing!", "I hated this film."]
for text in texts:
    predicted_class, probabilities = predict(text)
    print(f"Text: {text}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Probabilities: {probabilities}")
