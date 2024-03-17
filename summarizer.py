import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import copy
import math
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from translator import TranslationModel



# def mean_pooling(model_output, attention_mask):
#     token_embeddings = model_output[0] #First element of model_output contains all token embeddings
#     input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#     sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#     sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#     return sum_embeddings / sum_mask

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def extract_sentences(result: list):
    sentences = list()
    for item in result:
        sentences.append(item['text'])
    return sentences

def encode_sentences(sentences: list, tokenizer, model, batch_size):
    n_batches = math.ceil(len(sentences)/batch_size)
    sentences_tokenized = tokenizer(sentences, padding=True, truncation=True, max_length=256, return_tensors='pt')
    sentence_embeddings = list()

    for i in tqdm(range(n_batches)):
        input_ids = sentences_tokenized["input_ids"][i*batch_size : (i+1)*batch_size].to(model.device)
        attention_mask = sentences_tokenized["attention_mask"][i*batch_size : (i+1)*batch_size].to(model.device)
        encoded_input = {"input_ids": input_ids, "attention_mask": attention_mask}
        with torch.no_grad():
            model_output = model(**encoded_input)
        y = mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings.append(y)
    sentence_embeddings = torch.concat(sentence_embeddings, dim=0)
    return sentence_embeddings

def construct_blocks(sentence_embeddings, block_size):
    block_size = 3
    n_blocks = math.ceil(len(sentence_embeddings) / block_size)
    block_embeddings = list()
    for i in range(n_blocks):
        block_embeddings.append(sentence_embeddings[i*block_size : (i+1)*block_size].mean(dim=0).reshape((1, -1)))
    block_embeddings = torch.concat(block_embeddings, dim=0)
    return block_embeddings


def compute_sim_scores(block_embeddings: np.ndarray):
    return np.diag(cosine_similarity(block_embeddings.numpy()[:-1], block_embeddings.numpy()[1:]))

def compute_threshold(cos_scores: np.ndarray):
    return cos_scores.mean() - cos_scores.std()

def find_break_ids(cos_scores: np.ndarray, threshold: float):
    return np.arange(len(cos_scores))[cos_scores < threshold]+1

def divide_sentences(sentences: list, break_ids: np.ndarray, block_size: int):
    blocks = list()
    for i, _ in enumerate(break_ids):
        if i == 0:
            start = 0
            end = break_ids[i]*block_size
        else:
            start = break_ids[i-1]*block_size
            end = break_ids[i]*block_size
        print(start, end, len(sentences))
        blocks.append(sentences[start : end])
    return blocks


def truncate_texts(texts, tokenizer, max_length=1024):
    texts_enc = tokenizer(texts, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    y = tokenizer.batch_decode(texts_enc["input_ids"], skip_special_tokens=True)
    return y


class SummaryModel:

    def summarize_utterances(
            self, utterances_en: list, speakers: list, speaker_selected = None, batch_size=16, block_size=3
    ):

        device = torch.device("cuda")

        # load models
        model_name = "sentence-transformers/bert-base-nli-mean-tokens"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        ru_en_tr = TranslationModel("Helsinki-NLP/opus-mt-ru-en")

        tokenizer_sum = AutoTokenizer.from_pretrained("knkarthick/MEETING_SUMMARY")
        summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY", device=device)

        if speaker_selected is None:
            utterances = [f"{speaker}: {utterance}" for utterance, speaker in zip(utterances_en, speakers)]
            utterances_raw = copy.copy(utterances_en)
        else:
            utterances = list()
            utterances_raw = list()
            for utterance, speaker in zip(utterances_en, speakers):
                if speaker == speaker_selected:
                    utterances.append(f"{speaker}: {utterance}")
                    utterances_raw.append(utterance)

        # summrization logic
        sentence_embeddings = encode_sentences(utterances_raw, tokenizer, model, batch_size=batch_size)
        block_embeddings = construct_blocks(sentence_embeddings, block_size=block_size)
        cos_scores = compute_sim_scores(block_embeddings)
        threshold = compute_threshold(cos_scores)
        break_ids = find_break_ids(cos_scores, threshold)
        blocks = divide_sentences(utterances, break_ids, block_size=block_size)
        blocks_str = ["\n".join(x) for x in blocks]
        blocks_str_trunc = truncate_texts(blocks_str, tokenizer_sum)

        summaries = summarizer(blocks_str_trunc)
        summaries = [x["summary_text"] for x in summaries]

        return summaries