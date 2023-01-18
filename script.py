import torch

from language_model import BigramLangModel, loss_estimation, get_batch
from params import device, learning_rate, max_iters, eval_inter


# Set seed to 1337 cuz leet br0000
torch.manual_seed(1337)

# Read text data
with open('all_of_shakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create a sorted list of every unique character in the text
chars = sorted(list(set(text)))
n_chars = len(chars)

# Encoder and decoder for tokenization
normal = { char: idx for idx, char in enumerate(chars) }
inverse = { idx: char for idx, char in enumerate(chars) }
encoder = lambda x: [normal[char] for char in x]
decoder = lambda x: ''.join([inverse[char] for char in x])

# Tokenize the data 
data = torch.tensor(encoder(text), dtype=torch.long)

# Split training (90%) and testing (10%) data 
n = int(0.9 * len(data))
training_data = data[:n]
testing_data = data[n:]


# Define model instance 
model = BigramLangModel(n_chars)
_model = model.to(device)

print(sum(p.numel() for p in _model.parameters()) / 1e6, 'Million parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_inter == 0 or iter == max_iters - 1:
        losses = loss_estimation(model, training_data, testing_data)
        print(f"iteration {iter}: training loss {losses['train']:.4f}, testing loss {losses['test']:.4f}")

    # Sample batch
    xb, yb = get_batch('train')

    # Evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(_model.generate(ctx, max_new_tokens=500)[0].tolist()))






