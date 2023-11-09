import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset, DataLoader
from yet_another_retnet.retnet import retnet_1_3b

df = pd.read_csv("cleaned_wikipedia_data.csv")

# Assuming 'df["words"]' contains your text data
vectorizer = CountVectorizer()

# Fit the vectorizer and then transform
vectorizer.fit(df["words"])
vocab_size = len(vectorizer.vocabulary_)

# Load the model
retnet = retnet_1_3b(num_tokens=vocab_size, device="cuda")
model = retnet.half()

# Convert words to unique integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(df["words"])

data = vectorizer.transform(df["words"])

# Convert data to tensor
converted_data = torch.from_numpy(data.toarray()).int()

# Convert labels to tensor
converted_labels = torch.from_numpy(encoded_labels).long()


# Define Dataset
class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx], self.labels[idx]
        return sample


# Create dataset and dataloader
dataset = MyDataset(data=converted_data, labels=converted_labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = inputs.to("cuda")  # Ensure inputs are half-precision
        targets = targets.to("cuda")
        optimizer.zero_grad()
        outputs = model(inputs, targets)  # If the model requires labels
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Save the model
torch.save(model.state_dict(), "retnet_state_dict.pth")
