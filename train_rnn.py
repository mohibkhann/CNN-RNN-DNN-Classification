import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.legacy import data, datasets

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define fields for text and labels
TEXT = data.Field(tokenize="spacy", lower=True, include_lengths=True)
LABEL = data.LabelField(dtype=torch.long)

# Load dataset
train_data, test_data = datasets.AG_NEWS.splits(TEXT, LABEL)

# Build vocabulary
TEXT.build_vocab(train_data, max_size=25000)
LABEL.build_vocab(train_data)

# Create data loaders
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=64,
    sort_within_batch=True,
    device=device
)

# Define RNN Model
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        packed_embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_embedded)
        return self.fc(hidden.squeeze(0))

# Model, Loss, Optimizer
INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = len(LABEL.vocab)

model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train_model(model, iterator, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in iterator:
            text, text_lengths = batch.text
            labels = batch.label
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(text, text_lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(iterator):.4f}")

# Train the model
if __name__ == "__main__":
    train_model(model, train_iterator)
