import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Load FAQ data from dataset.txt file
def load_faq_data(file_path):
    data = []
    current_category = None

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('[') and line.endswith(']'):
                current_category = line[1:-1]
            elif '|' in line:
                question, answer = line.split('|', 1)
                data.append({"category": current_category, "question": question.strip(), "answer": answer.strip()})

    return pd.DataFrame(data)

# Load the data
file_path = '/Users/aadithyaram/Desktop/Projekts/SRMChatbot/ChatBot-main-2/static/dataset/dataset.txt'
faq_data = load_faq_data(file_path)

# Assign unique labels to each category
faq_data['label'] = faq_data['category'].astype('category').cat.codes

# Initialize the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize the data
def preprocess_function(examples):
    return tokenizer(examples['question'], truncation=True, padding='max_length', max_length=128)

# Convert the DataFrame to a Hugging Face dataset
dataset = Dataset.from_pandas(faq_data)

# Apply the preprocessing
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into train and test
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

# Initialize the model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=faq_data['label'].nunique())

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model('./results')
tokenizer.save_pretrained('./results')
