import os
import json
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Initialize Weights & Biases for experiment tracking
import wandb
wandb.init(project='ae-detection-inference', entity='vijumuraari')

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = RobertaForSequenceClassification.from_pretrained('vtiyyal1/AE-classification-RoBerta').to(device)
model.eval()

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Function to prepare data loader
def prepare_data_loader(texts, batch_size=16):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,  # Limit the length to 512 tokens
            padding='max_length',  # Pad to max length
            truncation=True,  # Truncate longer sequences
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    dataset = TensorDataset(input_ids, attention_masks)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader

# Function to get predictions
def get_predictions(data_loader):
    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
    return all_preds

# Path to your JSON files
data_dir = '/home/vtiyyal1/data-mdredze1/vtiyyal1/cannabis_project/cannabisallfiltered'
output_dir = '/home/vtiyyal1/data-mdredze1/vtiyyal1/cannabis_project/outputs'
os.makedirs(output_dir, exist_ok=True)

# Initialize statistics tracking
statistics = []

# Process each JSON file
for filename in os.listdir(data_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(data_dir, filename)
        try:
            # Load JSON data
            with open(file_path, 'r') as file:
                data = json.load(file)
            
            # Check if data is empty
            if not data:
                statistics.append({'filename': filename, 'num_instances': 0, 'num_adverse_events': 0})
                continue
            
            # Extract texts and combine title and selftext
            texts = [item['title'] + " " + item['selftext'] for item in data]

            # Prepare data loader
            data_loader = prepare_data_loader(texts)

            # Get predictions
            predictions = get_predictions(data_loader)

            # Add predictions to data
            for i, item in enumerate(data):
                item['roberta_ae_prediction'] = int(predictions[i])

            # Convert to DataFrame and save to CSV
            df = pd.DataFrame(data)
            output_file_path = os.path.join(output_dir, f'annotated_{filename.replace(".json", ".csv")}')
            df.to_csv(output_file_path, index=False)

            # Gather statistics
            num_instances = len(data)
            num_adverse_events = sum(predictions)
            statistics.append({'filename': filename, 'num_instances': num_instances, 'num_adverse_events': num_adverse_events})

            # Log to Weights & Biases
            wandb.log({filename: {"num_instances": num_instances, "num_adverse_events": num_adverse_events}})

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Save statistics to a CSV file
statistics_df = pd.DataFrame(statistics)
statistics_df.to_csv(os.path.join(output_dir, 'annotation_statistics.csv'), index=False)

# Sample and save some examples
sampled_df = pd.DataFrame()
for filename in statistics_df['filename']:
    df = pd.read_csv(os.path.join(output_dir, f'annotated_{filename.replace(".json", ".csv")}'))
    sampled_df = sampled_df.append(df.sample(min(10, len(df))))

sampled_df.to_csv(os.path.join(output_dir, 'sampled_annotations.csv'), index=False)
