import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import os

# Load the data
news_df = pd.read_csv('News_Train.csv')
price_df = pd.read_csv('Nifty_50_data.csv')
with open('combined_strengths.json', 'r') as f:
    causal_strengths = json.load(f)

# Preprocess the news data
news_df['combined_text'] = news_df['headline'] + ' ' + news_df['description'] + ' ' + news_df['articleBody']
news_df['date'] = pd.to_datetime(news_df['datePublished']).dt.tz_localize(None)
news_df = news_df.sort_values('date')

# Preprocess the price data
price_df['Date'] = pd.to_datetime(price_df['Date'])
price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
price_df[price_columns] = price_df[price_columns].apply(pd.to_numeric, errors='coerce')
price_df = price_df.sort_values('Date')

# Merge news and price data
df = pd.merge_asof(news_df, price_df, left_on='date', right_on='Date', direction='forward')

# Create labels (1 if price goes up, 0 if it goes down or stays the same)
df['label'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop rows with NaN values
df = df.dropna()

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# Normalize price data
scaler = StandardScaler()
df[price_columns] = scaler.fit_transform(df[price_columns])

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NewsStockDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['combined_text']
        label = self.data.iloc[idx]['label_encoded']
        price_data = self.data.iloc[idx][self.price_columns].astype(float).values

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'price_data': torch.tensor(price_data, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long)
        }

class HAN(nn.Module):
    def __init__(self, bert_model, num_classes, hidden_size=256):
        super(HAN, self).__init__()
        self.bert = bert_model
        self.word_gru = nn.GRU(768, hidden_size, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(hidden_size * 2, 1)
        self.news_gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.news_attention = nn.Linear(hidden_size * 2, 1)
        self.price_fc = nn.Linear(6, hidden_size)  # 6 is the number of price features
        self.fc = nn.Linear(hidden_size * 2 + hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, price_data):
        # Word-level encoding
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        word_hidden, _ = self.word_gru(bert_output)
        word_attention = self.word_attention(word_hidden)
        word_attention = torch.softmax(word_attention, dim=1)
        word_context = torch.sum(word_hidden * word_attention, dim=1)

        # News-level encoding
        news_hidden, _ = self.news_gru(word_context.unsqueeze(1))
        news_attention = self.news_attention(news_hidden)
        news_attention = torch.softmax(news_attention, dim=1)
        news_context = torch.sum(news_hidden * news_attention, dim=1)

        # Price data encoding
        price_features = self.price_fc(price_data)

        # Combine news and price features
        combined_features = torch.cat((news_context, price_features), dim=1)

        # Final prediction
        output = self.fc(combined_features)
        return output, word_attention, news_attention

def compute_auxiliary_loss(word_attention, causal_strengths, words):
    aux_loss = 0
    for i, word in enumerate(words):
        if word in causal_strengths:
            normalized_strength = torch.softmax(torch.tensor(list(causal_strengths[word].values())), dim=0)
            aux_loss += torch.sum((word_attention[i] - normalized_strength) ** 2)
    return aux_loss

def compute_total_loss(outputs, labels, word_attention, causal_strengths, words, lambda_aux=0.1):
    criterion = nn.CrossEntropyLoss()
    cross_entropy_loss = criterion(outputs, labels)
    aux_loss = compute_auxiliary_loss(word_attention, causal_strengths, words)
    total_loss = cross_entropy_loss + lambda_aux * aux_loss
    return total_loss

def train_model(model, train_loader, val_loader, causal_strengths, num_epochs=10, learning_rate=2e-5, checkpoint_dir='checkpoints'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            price_data = batch['price_data'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs, word_attention, _ = model(input_ids, attention_mask, price_data)

            words = tokenizer.convert_ids_to_tokens(input_ids[0])
            loss = compute_total_loss(outputs, labels, word_attention, causal_strengths, words)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                price_data = batch['price_data'].to(device)
                labels = batch['label'].to(device)

                outputs, _, _ = model(input_ids, attention_mask, price_data)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)

        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, checkpoint_path)
            print(f'Checkpoint saved to {checkpoint_path}')

        print('-' * 40)

    return model

# Create data loaders
train_dataset = NewsStockDataset(train_df, tokenizer)
test_dataset = NewsStockDataset(test_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = HAN(bert_model, num_classes=2)

# Train the model
trained_model = train_model(model, train_loader, test_loader, causal_strengths)

# Save the final model
torch.save(trained_model.state_dict(), 'final_han_model.pth')

# Evaluate the model on the test set
model.eval()
test_loss = 0
correct = 0
total = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        price_data = batch['price_data'].to(device)
        labels = batch['label'].to(device)

        outputs, _, _ = model(input_ids, attention_mask, price_data)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Loss: {test_loss/len(test_loader):.4f}')
print(f'Test Accuracy: {100 * correct / total:.2f}%')