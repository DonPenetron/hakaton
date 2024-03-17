import math
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TranslationModel:

    def __init__(self, model_name: str, device="cuda"):

        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def translate(self, texts: str, batch_size=16):
        n_batches = math.ceil(len(texts)/batch_size)
        texts_enc = self.tokenizer(texts, max_length=512, truncation=True, padding="max_length", return_tensors="pt")
        translations = list()
        for i in tqdm(range(n_batches)):
            input_ids = texts_enc["input_ids"][i*batch_size : (i+1)*batch_size].to(self.device)
            attention_mask = texts_enc["attention_mask"][i*batch_size : (i+1)*batch_size].to(self.device)
            encoded_input = {"input_ids": input_ids, "attention_mask": attention_mask}
            with torch.no_grad():
                model_output = self.model.generate(**encoded_input)
                y = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)
            translations.extend(y)
        return translations