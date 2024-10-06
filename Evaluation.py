import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import json
import numpy as np

class HAN(nn.Module):
    def __init__(self, bert_model, num_classes, hidden_size=256, dropout_rate=0.3):
        super(HAN, self).__init__()
        self.bert = bert_model
        self.word_gru = nn.GRU(768, hidden_size, bidirectional=True, batch_first=True)
        self.word_attention = nn.Linear(hidden_size * 2, 1)
        self.news_gru = nn.GRU(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True)
        self.news_attention = nn.Linear(hidden_size * 2, 1)
        self.price_fc = nn.Linear(6, hidden_size)  # 6 is the number of price features
        self.fc = nn.Linear(hidden_size * 2 + hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids, attention_mask, price_data):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        word_hidden, _ = self.word_gru(bert_output)
        word_hidden = self.dropout(word_hidden)
        word_attention = self.word_attention(word_hidden)
        word_attention = torch.softmax(word_attention, dim=1)
        word_context = torch.sum(word_hidden * word_attention, dim=1)

        news_hidden, _ = self.news_gru(word_context.unsqueeze(1))
        news_hidden = self.dropout(news_hidden)
        news_attention = self.news_attention(news_hidden)
        news_attention = torch.softmax(news_attention, dim=1)
        news_context = torch.sum(news_hidden * news_attention, dim=1)

        price_features = self.price_fc(price_data)
        price_features = self.dropout(price_features)

        combined_features = torch.cat((news_context, price_features), dim=1)
        output = self.fc(combined_features)
        return output, word_attention, news_attention


# Load the data
news_test_df = pd.read_csv('News_Test.csv')
price_test_df = pd.read_csv('Nifty_50_data_test.csv')

# Preprocess the news data
news_test_df['combined_text'] = news_test_df['headline'] + ' ' + news_test_df['description'] + ' ' + news_test_df[
    'articleBody']
news_test_df['date'] = pd.to_datetime(news_test_df['datePublished']).dt.tz_localize(None)
news_test_df = news_test_df.sort_values('date')

# Preprocess the price data
price_test_df['Date'] = pd.to_datetime(price_test_df['Date'])
price_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
for col in price_columns:
    price_test_df[col] = pd.to_numeric(price_test_df[col], errors='coerce')
price_test_df = price_test_df.sort_values('Date')

# Merge news and price data
df_test = pd.merge_asof(news_test_df, price_test_df, left_on='date', right_on='Date', direction='forward')

# Create labels (0 for downward, 1 for neutral, 2 for upward)
df_test['price_change'] = df_test['Close'].shift(-1) - df_test['Close']
df_test['label'] = np.select(
    [df_test['price_change'] < -0.001, df_test['price_change'] > 0.001],
    [0, 2],
    default=1
)

# Drop rows with NaN values
df_test = df_test.dropna()

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model = BertModel.from_pretrained('bert-base-uncased')
loaded_model = HAN(bert_model, num_classes=3)

# Load the checkpoint
checkpoint = torch.load('checkpoints/best_model_epoch_5.pth', map_location=device)

# Load only the model state dict
loaded_model.load_state_dict(checkpoint['model_state_dict'])
loaded_model.to(device)
loaded_model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import numpy as np


def predict(text, price_data):
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Convert price_data to a pandas Series and fill NaN values
    price_data_series = pd.Series(price_data)  # Convert to pandas Series
    price_data_numeric = pd.to_numeric(price_data_series, errors='coerce').fillna(0).values
    price_tensor = torch.tensor(price_data_numeric, dtype=torch.float).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _, _ = loaded_model(input_ids, attention_mask, price_tensor)
        _, predicted = torch.max(outputs, 1)

    return predicted.item()


# Make predictions and calculate accuracy
correct_predictions = 0
total_predictions = 0

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
    text = row['combined_text']
    price_data = row[price_columns]
    true_label = row['label']

    try:
        predicted_label = predict(text, price_data)

        if predicted_label == true_label:
            correct_predictions += 1
        total_predictions += 1
    except Exception as e:
        print(f"Error processing row: {e}")
        continue

accuracy = (correct_predictions / total_predictions) * 100

print(f"Total predictions: {total_predictions}")
print(f"Correct predictions: {correct_predictions}")
print(f"Accuracy: {accuracy:.2f}%")

# Print confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true = df_test['label']
y_pred = [predict(row['combined_text'], row[price_columns].values) for _, row in df_test.iterrows()]

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the confusion matrix as an image file (PNG format)
plt.savefig('confusion_matrix.png', dpi=300)  # You can adjust the dpi for higher resolution
plt.show()