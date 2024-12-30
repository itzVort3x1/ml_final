import torch
import torch.nn as nn
import torch.nn.functional as F
from midiutil import MIDIFile
import math

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 1000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# Load melody dataset
with open('inputMelodiesAugmented.txt', 'r') as f:
    text = f.read()

# Tokenization: Map unique tokens to indices and vice versa
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}  # String-to-index mapping
itos = {i: ch for i, ch in enumerate(chars)}  # Index-to-string mapping

# Encoding and decoding functions
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert data to tensor
data = torch.tensor(encode(text), dtype=torch.long)
train_data = data[:int(0.8 * len(data))]
val_data = data[int(0.8 * len(data)):]

# Function to generate training/validation batches
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Model Definition
class GPTLanguageModel(nn.Module):
    def _init_(self):
        super()._init_()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(idx.shape[1], device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        targets = targets.view(B * T)
        loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Block(nn.Module):
    def _init_(self, n_embd, n_head):
        super()._init_()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MultiHeadAttention(nn.Module):
    def _init_(self, num_heads, head_size):
        super()._init_()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class Head(nn.Module):
    def _init_(self, head_size):
        super()._init_()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class FeedForward(nn.Module):
    def _init_(self, n_embd):
        super()._init_()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Training Loop
model = GPTLanguageModel().to(device)

# Load model state_dict from the saved checkpoint
checkpoint_path = "melody_gpt.pth"  # Path to the saved model file
model.load_state_dict(torch.load(checkpoint_path))

# Recreate the optimizer to continue training from where we left off
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Optional: Load optimizer state if available (to resume optimizer state)
optimizer_path = "optimizer_state.pth"  # Path to the saved optimizer state
try:
    optimizer.load_state_dict(torch.load(optimizer_path))
    print("Optimizer state loaded successfully.")
except FileNotFoundError:
    print("No optimizer state found, starting fresh.")

# Continue training from the last checkpoint
for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        model.eval()
        train_loss, val_loss = 0, 0
        with torch.no_grad():
            for _ in range(eval_iters):
                xb, yb = get_batch('train')
                _, loss = model(xb, yb)
                train_loss += loss.item()
                xb, yb = get_batch('val')
                _, loss = model(xb, yb)
                val_loss += loss.item()
        print(f"Step {iter}: Train Loss {train_loss/eval_iters:.4f}, Val Loss {val_loss/eval_iters:.4f}")
        model.train()

    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)  # Move to correct device
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Save the model and optimizer state
torch.save(model.state_dict(), checkpoint_path)
torch.save(optimizer.state_dict(), "optimizer_state.pth")

# Evaluation: Perplexity and Accuracy
def calculate_perplexity(model, data):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch(data)
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)

def calculate_accuracy(model, data):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch(data)
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == yb).sum().item()
            total += yb.numel()
    return correct / total

train_perplexity = calculate_perplexity(model, 'train')
val_perplexity = calculate_perplexity(model, 'val')
train_accuracy = calculate_accuracy(model, 'train')
val_accuracy = calculate_accuracy(model, 'val')

print(f"Train Perplexity: {train_perplexity:.4f}, Validation Perplexity: {val_perplexity:.4f}")
print(f"Train Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Subjective Evaluation: MIDI Conversion
def melody_to_midi(melody, output_file="generated_melody.mid", tempo=120):
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)
    pitch_map = {
        'C': 60,
        'C#': 61, 
        'D': 62, 
        'D#': 63, 
        'E': 64,
        'F': 65, 
        'F#': 66, 
        'G': 67, 
        'G#': 68, 
        'A': 69,
        'A#': 70, 
        'B': 71, 
        'R': None
    }
    time = 0
    duration = 1
    for token in melody:
        pitch = pitch_map.get(token)
        if pitch is not None:
            midi.addNote(0, 0, pitch, time, duration, 100)
        time += duration
    with open(output_file, "wb") as f:
        midi.writeFile(f)
    print(f"MIDI file saved: {output_file}")

def display_and_save_melody(start_token='R', max_new_tokens=100, output_file="generated_melody.mid", tempo=120):
    idx = torch.tensor([[stoi[start_token]]], dtype=torch.long, device=device)
    generated_indices = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()
    melody = decode(generated_indices)
    print("\nGenerated Melody:")
    print(melody)
    melody_to_midi(melody, output_file, tempo)

display_and_save_melody(max_new_tokens=100, output_file="generated_melody.mid")