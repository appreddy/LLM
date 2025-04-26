Train a lightweight GPT-2 (DistilGPT2) model from scratch using the WikiText-2 dataset, with Hugging Face Transformers and Datasets libraries.

📦 Setup
Install the required libraries:

pip install torch transformers datasets

🛠️ How It Works

Tokenizer: Load GPT-2 tokenizer and set pad_token to eos_token.

Model: Define a smaller GPT-2 model (6 layers) using GPT2Config.

Dataset: Load WikiText-2-raw-v1, split into train/validation, and tokenize it.

DataLoader: Create data loaders with DataCollatorForLanguageModeling.

Training: Train the model for 3 epochs with AdamW optimizer and linear scheduler.

Saving: Save the trained model and tokenizer locally.

Text Generation: Generate text using the trained model via Hugging Face's pipeline.

🧩 Folder Structure

├── distilgpt2-from-scratch/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── merges.txt

🚀 Quick Start
Train and save the model:

python train.py

Generate text:

from transformers import pipeline

generator = pipeline("text-generation", model="./distilgpt2-from-scratch")
print(generator("Once upon a time", max_new_tokens=50))
