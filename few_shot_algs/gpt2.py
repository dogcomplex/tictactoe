import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from few_shot_algs.few_shot_alg import Algorithm
import random
from torch.utils.data import DataLoader, Dataset
import warnings

# Disable the specific warning
warnings.filterwarnings("ignore", message="This implementation of AdamW is deprecated")

# Check for accelerate library
try:
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate library not found. GPT2Algorithm will use a simpler prediction method without training.")

class StateLabelDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_text = example['input']
        label_text = example['label']

        input_ids = self.tokenizer.encode(input_text, truncation=True, max_length=50)
        label_ids = self.tokenizer.encode(label_text, truncation=True, max_length=5)

        # Create a single sequence with the label as the next token to predict
        input_ids += label_ids
        labels = [-100] * len(input_ids)  # -100 tokens will be ignored in the loss calculation
        labels[-len(label_ids):] = label_ids

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

# Example generation function to create the training data
def create_training_data(history, num_examples=10):
    examples = []
    for state, guess, label in history[-num_examples:]:
        example_text = f"State: {state} -> Label: {label}\n"
        examples.append({'input': example_text, 'label': f"{label}"})
    return examples

# Create a few-shot prompt for GPT-2 with known examples and a new query
def create_few_shot_prompt(known_examples, query_state):
    prompt = ""
    for example in known_examples:
        prompt += f"State: {example['input']} -> Label: {example['label']}\n"
    prompt += f"State: {query_state} -> Label: "
    return prompt

class SimpleDataset(Dataset):
    def __init__(self, history, tokenizer):
        self.history = history
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.history)

    def __getitem__(self, idx):
        observation, _, label = self.history[idx]
        text = f"[STATE]{observation}[LABEL]{label}"
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=20)
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

class GPT2Algorithm(Algorithm):
    def __init__(self, model_name='gpt2', few_shot_examples=5):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.few_shot_examples = few_shot_examples
        self.is_trained = False

        # Set pad_token to eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Add special tokens for the input and output format
        special_tokens = {'additional_special_tokens': ['[STATE]', '[LABEL]']}
        self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def predict(self, observation: str) -> int:
        if len(self.history) < self.few_shot_examples:
            return random.randint(0, 4)

        known_examples = random.sample(self.history, self.few_shot_examples)
        prompt = self.create_few_shot_prompt(known_examples, observation)

        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        with torch.no_grad():
            output = self.model.generate(
                input_ids, 
                max_length=input_ids.shape[1] + 2,  # Only generate one more token
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_k=5  # Limit to the 5 most likely tokens
            )
        
        prediction = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        predicted_label = prediction.split("[LABEL]")[-1].strip()
        try:
            return int(predicted_label)
        except ValueError:
            return random.randint(0, 4)

    def update_history(self, observation: str, guess: int, correct_label: int):
        super().update_history(observation, guess, correct_label)
        self.train_model()

    def train_model(self):
        if len(self.history) < 5:
            return

        try:
            self.is_trained = True
            dataset = SimpleDataset(self.history, self.tokenizer)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

            optimizer = AdamW(self.model.parameters(), lr=1e-5)
            self.model.train()

            for _ in range(3):  # Perform 3 epochs of training
                for batch in dataloader:
                    optimizer.zero_grad()
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

            self.model.eval()
        except Exception as e:
            print(f"Warning: Training failed due to an error: {str(e)}")
            print("Continuing with the pre-trained model without fine-tuning.")

    @staticmethod
    def create_few_shot_prompt(known_examples, query_state):
        prompt = ""
        for example in known_examples:
            prompt += f"[STATE]{example[0]}[LABEL]{example[2]}\n"
        prompt += f"[STATE]{query_state}[LABEL]"
        return prompt

    @staticmethod
    def observation_to_features(observation: str) -> str:
        return observation
