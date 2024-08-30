import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from huggingface_hub import HfApi
from huggingface_hub import login, HfApi

# Log in with your Hugging Face token
login(token="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")
# Load and prepare data
human_annotations = pd.read_csv('/home/vtiyyal1/cannabisproject/Delta 8 AEs (Human Annotations).csv')

# Combine title and selftext
human_annotations['text'] = human_annotations['title'].astype(str) + ' ' + human_annotations['selftext'].astype(str)

# Split data
train_val_data, test_data = train_test_split(human_annotations, test_size=0.1, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.05, random_state=42)

# Prepare data for N-gram model
vectorizer = TfidfVectorizer(min_df=3, max_df=0.9, ngram_range=(1, 3))
X_train_ngram = vectorizer.fit_transform(train_data['text'])
X_val_ngram = vectorizer.transform(val_data['text'])
X_test_ngram = vectorizer.transform(test_data['text'])

y_train = train_data['ae'].values
y_val = val_data['ae'].values
y_test = test_data['ae'].values

# Prepare data for BERT and RoBERTa
def tokenize_data(data, tokenizer):
    return tokenizer(data['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
roberta_tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')

train_bert = tokenize_data(train_data, bert_tokenizer)
val_bert = tokenize_data(val_data, bert_tokenizer)
test_bert = tokenize_data(test_data, bert_tokenizer)

train_roberta = tokenize_data(train_data, roberta_tokenizer)
val_roberta = tokenize_data(val_data, roberta_tokenizer)
test_roberta = tokenize_data(test_data, roberta_tokenizer)

# Create DataLoaders
batch_size = 16

train_dataset_bert = TensorDataset(train_bert['input_ids'], train_bert['attention_mask'], torch.tensor(y_train))
val_dataset_bert = TensorDataset(val_bert['input_ids'], val_bert['attention_mask'], torch.tensor(y_val))
test_dataset_bert = TensorDataset(test_bert['input_ids'], test_bert['attention_mask'], torch.tensor(y_test))

train_dataloader_bert = DataLoader(train_dataset_bert, batch_size=batch_size, shuffle=True)
val_dataloader_bert = DataLoader(val_dataset_bert, batch_size=batch_size)
test_dataloader_bert = DataLoader(test_dataset_bert, batch_size=batch_size)

train_dataset_roberta = TensorDataset(train_roberta['input_ids'], train_roberta['attention_mask'], torch.tensor(y_train))
val_dataset_roberta = TensorDataset(val_roberta['input_ids'], val_roberta['attention_mask'], torch.tensor(y_val))
test_dataset_roberta = TensorDataset(test_roberta['input_ids'], test_roberta['attention_mask'], torch.tensor(y_test))

train_dataloader_roberta = DataLoader(train_dataset_roberta, batch_size=batch_size, shuffle=True)
val_dataloader_roberta = DataLoader(val_dataset_roberta, batch_size=batch_size)
test_dataloader_roberta = DataLoader(test_dataset_roberta, batch_size=batch_size)

# Train N-gram model
ngram_model = LogisticRegression(max_iter=1000)
ngram_model.fit(X_train_ngram, y_train)

# Train function for BERT and RoBERTa
def train_model(model, train_dataloader, val_dataloader, epochs, optimizer, scheduler, device, model_name):
    best_val_auc = 0.0
    train_losses = []
    val_losses = []
    epochs_without_improvement = 0
    best_model = None
    patience = 3  # Number of epochs to wait for improvement before stopping

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0

        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_true = []

        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}

            with torch.no_grad():
                outputs = model(**inputs)

            val_loss += outputs.loss.item()
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            val_true.extend(inputs['labels'].cpu().numpy())

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        val_accuracy = accuracy_score(val_true, val_preds)
        val_precision = precision_score(val_true, val_preds)
        val_recall = recall_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds)
        val_auc = roc_auc_score(val_true, val_preds)

        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
        print(f"  Val Precision: {val_precision:.4f}")
        print(f"  Val Recall: {val_recall:.4f}")
        print(f"  Val F1-score: {val_f1:.4f}")
        print(f"  Val AUC-ROC: {val_auc:.4f}")

        wandb.log({
            f"{model_name} Train Loss": avg_train_loss,
            f"{model_name} Val Loss": avg_val_loss,
            f"{model_name} Val Accuracy": val_accuracy,
            f"{model_name} Val Precision": val_precision,
            f"{model_name} Val Recall": val_recall,
            f"{model_name} Val F1-score": val_f1,
            f"{model_name} Val AUC-ROC": val_auc
        })

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model.state_dict()
            epochs_without_improvement = 0
            upload_model_to_huggingface(model, model_name)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Load the best model before returning
    model.load_state_dict(best_model)

    # Plotting the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'{model_name} Training and Validation Loss Curves')
    plt.show()

    return model

# Function to upload the best model to Hugging Face Hub
def upload_model_to_huggingface(model, model_name):
    model.save_pretrained(f"best_{model_name}_model")
    api = HfApi()
    api.upload_folder(
        folder_path=f"best_{model_name}_model",
        repo_id=f"vtiyyal1/AE-classification-{model_name}",
        repo_type="model",
    )

# Train BERT and RoBERTa models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
roberta_model = RobertaForSequenceClassification.from_pretrained('FacebookAI/roberta-base', num_labels=2).to(device)

bert_optimizer = AdamW(bert_model.parameters(), lr=2e-5, eps=1e-8)
roberta_optimizer = AdamW(roberta_model.parameters(), lr=2e-5, eps=1e-8)

bert_scheduler = get_linear_schedule_with_warmup(bert_optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader_bert) * 10)
roberta_scheduler = get_linear_schedule_with_warmup(roberta_optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader_roberta) * 10)

# Set up wandb for tracking and visualization
wandb.init(project='ae-detection-training', entity='vijumuraari')
wandb.watch(bert_model)
wandb.watch(roberta_model)

# Train models
# Train models
#print("Training BERT model...")
#bert_model = train_model(bert_model, train_dataloader_bert, val_dataloader_bert, 50, bert_optimizer, bert_scheduler, device, "BERT")

bert_model = BertForSequenceClassification.from_pretrained('best_BERT_model', num_labels=2).to(device)
print("Training RoBERTa model...")
roberta_model = train_model(roberta_model, train_dataloader_roberta, val_dataloader_roberta, 5, roberta_optimizer, roberta_scheduler, device, "RoBERTa")
# Evaluate models on test set
def evaluate_model(model, test_dataloader, device, model_name):
    model.eval()
    test_preds = []
    test_true = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
            outputs = model(**inputs)
            logits = outputs.logits
            test_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            test_true.extend(batch[2].cpu().numpy())

    accuracy = accuracy_score(test_true, test_preds)
    precision = precision_score(test_true, test_preds)
    recall = recall_score(test_true, test_preds)
    f1 = f1_score(test_true, test_preds)
    auc = roc_auc_score(test_true, test_preds)

    print(f"{model_name} Test Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

    wandb.log({
        f"{model_name} Test Accuracy": accuracy,
        f"{model_name} Test Precision": precision,
        f"{model_name} Test Recall": recall,
        f"{model_name} Test F1-score": f1,
        f"{model_name} Test AUC-ROC": auc
    })

# Evaluate N-gram model
ngram_preds = ngram_model.predict(X_test_ngram)
ngram_accuracy = accuracy_score(y_test, ngram_preds)
ngram_precision = precision_score(y_test, ngram_preds)
ngram_recall = recall_score(y_test, ngram_preds)
ngram_f1 = f1_score(y_test, ngram_preds)
ngram_auc = roc_auc_score(y_test, ngram_preds)

print("N-gram Model Test Results:")
print(f"  Accuracy: {ngram_accuracy:.4f}")
print(f"  Precision: {ngram_precision:.4f}")
print(f"  Recall: {ngram_recall:.4f}")
print(f"  F1-score: {ngram_f1:.4f}")
print(f"  AUC-ROC: {ngram_auc:.4f}")

wandb.log({
    "N-gram Test Accuracy": ngram_accuracy,
    "N-gram Test Precision": ngram_precision,
    "N-gram Test Recall": ngram_recall,
    "N-gram Test F1-score": ngram_f1,
    "N-gram Test AUC-ROC": ngram_auc
})

# Evaluate BERT and RoBERTa models
evaluate_model(bert_model, test_dataloader_bert, device, "BERT")
evaluate_model(roberta_model, test_dataloader_roberta, device, "RoBERTa")

wandb.finish()