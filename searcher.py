import torch
from sentence_transformers import SentenceTransformer, util


class SearchModel:
    def __init__(self, embedder):
        self.embedder = embedder

    def get_top_similar(self, query_en, context, context_en):
        context_en_embeddings = self.embedder.encode(context_en, convert_to_tensor=True)
        query_en_embeddings = self.embedder.encode(query_en, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_en_embeddings, context_en_embeddings)[0]
        top_results = torch.topk(cos_scores, k=5)
        result = list()
        for idx in top_results[1]:
            result.append(context[idx])
        return result