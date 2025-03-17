import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# 1. Load and prepare dataset
dataset = load_dataset("imdb")  # You can replace this with your dataset

# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(dataset)
# Define the path where you want to save the dataset
dataset_path = './data/train.csv'

# Save the DataFrame to a CSV file in the `data` folder
df.to_csv(dataset_path, index=False)

print(f"Dataset saved to {dataset_path}")


# 2. Tokenize the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets['train']
val_dataset = tokenized_datasets['test']

# 3. Load pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 4. Define training arguments
training_args = TrainingArguments(
    output_dir='./results',              # output directory for results the model's training outputs, like intermediate checkpoints and logs, will be saved here
    evaluation_strategy="epoch",         # evaluation strategy to adopt during training , An epoch represents a full pass through the entire dataset during training.
    learning_rate=2e-5,                  # learning rate 2e-5 is scientific notation for 0.00002, which is a small learning rate, meaning the model will make smaller updates to its weights in each step, ensuring stable learning
    per_device_train_batch_size=16,      # batch size for training specifies that 16 samples(records) will be processed together in one forward and backward pass on each GPU device
    per_device_eval_batch_size=64,       # batch size for evaluation
    num_train_epochs=3,                  # number of training epochs
    weight_decay=0.01,                   # weight decay
)

# 5. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

# 6. Train the model
trainer.train()

# 7. Save the model and tokenizer
model.save_pretrained('./outputs/fine_tuned_bert')
tokenizer.save_pretrained('./outputs/tokenizer')

# 8. Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")
