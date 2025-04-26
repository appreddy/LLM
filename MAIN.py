# Import required libraries
from transformers import GPT2Config, AutoTokenizer, GPT2LMHeadModel, AdamW, pipeline
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
import torch

# Define tokenizer and model configuration
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Define DistilGPT2 configuration and initialize the model
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_layer=6,  # DistilGPT2 has 6 layers
    n_head=12,
)
model = GPT2LMHeadModel(config)

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  # Replace with your dataset
data = dataset["train"].train_test_split(shuffle=True, seed=42, test_size=0.2)
train = data["train"]
val = data["test"]

# Tokenize data
def tokenization(data):
    tokens = tokenizer(data["text"], padding="max_length", truncation=True, max_length=512)
    return tokens

train_token = train.map(tokenization, batched=True, remove_columns=["text"])
val_token = val.map(tokenization, batched=True, remove_columns=["text"])

# Add labels for language modeling
def create_labels(data):
    data["labels"] = data["input_ids"].copy()
    return data

lm_train = train_token.map(create_labels, batched=True)
lm_val = val_token.map(create_labels, batched=True)

# Prepare DataLoader
from torch.utils.data import DataLoader

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

train_dataloader = DataLoader(
    lm_train,
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

val_dataloader = DataLoader(
    lm_val,
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)

# Set up optimizer and learning rate scheduler
from transformers import get_scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=len(train_dataloader) * 3
)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch + 1}/{epochs} completed.")

# Save the trained model
model.save_pretrained("./distilgpt2-from-scratch")
tokenizer.save_pretrained("./distilgpt2-from-scratch")

# Text generation pipeline
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=50,
)

# Generate text
test_sentence = "Once upon a time"
generated_text = text_generator(test_sentence)
print(generated_text)