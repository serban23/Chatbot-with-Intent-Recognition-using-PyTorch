import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import random

# Download the punkt tokenizer for word_tokenize
nltk.download('punkt')

# Tokenization (splitting a sentence into words)
# and stemming (reducing a word to its base/root form)
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    stemmer = PorterStemmer()
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    # Create a bag of words vector with the same length as all_words, filled with 0s
    bow = np.zeros(len(all_words), dtype=np.float32)
    # Apply stemming to each word in the sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    # Set 1.0 for every word in the vocabulary that appears in the tokenized sentence
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bow[idx] = 1.0
    return bow

# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents_data = json.load(f)

# Prepare training data
all_words = []  # Vocabulary
tags = []       # Labels/intents
xy = []         # Tuple of (tokenized_pattern, tag)
ignore_words = ['?', '!', '.', ',']

# Extract tokens and tags from the intents JSON
for intent in intents_data['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        tokens = tokenize(pattern)
        all_words.extend(tokens)
        xy.append((tokens, tag))

# Apply stemming and remove duplicates
all_words = sorted(set([stem(w) for w in all_words if w not in ignore_words]))
tags = sorted(set(tags))

print(tags)
print(all_words)

# Create training data: X = input features, Y = output labels
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bow = bag_of_words(pattern_sentence, all_words)
    X_train.append(bow)
    label = tags.index(tag)
    y_train.append(label)

# Convert to NumPy arrays for PyTorch compatibility
X_train = np.array(X_train)
y_train = np.array(y_train)

# Dataset class to interface with PyTorch DataLoader
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train.astype(np.int64)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index].squeeze()
    
    def __len__(self):
        return self.n_samples

# Define the neural network for intent classification
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        self.layer_3 = nn.Linear(hidden_size, hidden_size)
        self.layer_4 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    # Define the forward pass
    def forward(self, x):
        out = self.relu(self.layer_1(x))
        out = self.relu(self.layer_2(out))
        out = self.relu(self.layer_3(out))
        out = self.layer_4(out)  # No activation at the last layer
        return out

# Training parameters
batch_size = 8
hidden_size = 16
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = .001 
num_epochs = 1000

# Create dataset and dataloader
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# Choose device: GPU if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss={loss.item():.4f}')

# Save the model
data = {
    'model_state': model.state_dict(),
    'input_size': input_size,
    'output_size': output_size,
    'hidden_size': hidden_size,
    'all_words': all_words,
    'tags': tags,
}

FILE = 'data.pth'
torch.save(data, FILE)
print(f'We have {len(tags)} topics!')
print(f'Training complete. File saved to {FILE}')

# Rebuild the model for inference
model_state = model.state_dict()
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Alex'
print("Let's chat! Type 'quit' to exit.")

# Chat loop
while True:
    sentence = input('You: ')
    if sentence.lower() == 'quit':
        break

    tokens = tokenize(sentence)
    bow = bag_of_words(tokens, all_words)
    bow = bow.reshape(1, bow.shape[0])
    bow_tensor = torch.from_numpy(bow).to(device)

    output = model(bow_tensor)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents_data['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                print(f"{bot_name}: {response}")
    else:
        print("I do not understand...")
