import re
from transformers import AutoTokenizer
import torch


class DataPreparation:
    def __init__(self, df, tokenizer_name="microsoft/deberta-v3-small", max_len=512):
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=False)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            item = self.df.iloc[idx]
            prompt = safe_encode(item['prompt'])
            response_a = safe_encode(item['response_a'])
            response_b = safe_encode(item['response_b'])

            # kalau kosong total, skip
            if not (prompt.strip() or response_a.strip() or response_b.strip()):
                raise ValueError("Empty text fields")

            # cari index kelas (0=A, 1=B, 2=Tie)
            if item['winner_model_a'] == 1:
                label = 0
            elif item['winner_model_b'] == 1:
                label = 1
            elif item['winner_tie']:
                label = 2

            text = f"PROMPT: {str(prompt)} [SEP] RESPONSE A: {str(response_a)} [SEP] RESPONSE B: {str(response_b)}"

            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )

            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "ground_truth": torch.tensor(label, dtype=torch.long)
            }
        except Exception as e:
            # log dulu biar tahu index mana
            print(f"⚠️ Error at index {idx}: {e}")
            item = self.df.iloc[idx]
            prompt = item['prompt']
            response_a = item['response_a']
            response_b = item['response_b']
            print(f"   prompt: {prompt}")
            print(f"   response_a: {response_a}")
            print(f"   response_b: {response_b}")
            # buat dummy data biar worker gak error
            dummy_ids = torch.zeros(self.max_len, dtype=torch.long)
            return {
                "input_ids": dummy_ids,
                "attention_mask": torch.zeros_like(dummy_ids),
                "ground_truth": torch.tensor([0, 0, 1], dtype=torch.float)
            }


def safe_encode(text):
    try:
        # Hilangkan karakter non-UTF8
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        # Hilangkan surrogate pairs
        text = re.sub(r'[\ud800-\udfff]', '', text)
        return text
    except Exception as e:
        print(f"Encoding error: {e}")
        return ""
