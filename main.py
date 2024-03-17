from fastapi import FastAPI, Request, HTTPException
from contextlib import asynccontextmanager
from transcriber import ASRModel
from summarizer import SummaryModel
from translator import TranslationModel
from tasks_parser import TasksModel
from sentence_transformers import SentenceTransformer, util
from searcher import SearchModel
from utils import correct_translation

import re
import os
import json
import requests
import numpy as np
import yargy

from utils import extract_speakers


ml_models = {}
tasks_data = {}
results_root = "/home/greenatom-admin/TEST/results"
tasks = {
    "semantic_search": [
        "Найди в документе"
    ],
    "tasks": [
        "Сформируй поручения"
    ],
    "summurize_speaker": [
        "Что сказал", "О чем говорил"
    ],
    "participants": [
        "Список участников"
    ]
}

def prepare_tasks(embedder, tokenizer):
    i = 0
    index_to_task = dict()
    tasks_flatten = list()
    tasks_dicts = dict()
    for key, values in tasks.items():
        tasks_flatten.extend(values)
        tasks_dicts[key] = list()
        for _ in values:
            index_to_task[i] = key
            i += 1
            tasks_dicts[key].extend(key.split())
        tasks_dicts[key] = set([x.normalized for x in tokenizer(" ".join(tasks_dicts[key]))])
    tasks_embeddings = embedder.encode(tasks_flatten, convert_to_tensor=True)
    
    return index_to_task, tasks_embeddings, tasks_dicts

def find_similar_task(query_embedding, tasks_embeddings):
    cos_scores = util.cos_sim(query_embedding, tasks_embeddings)[0]
    task_idx = np.argmax(cos_scores.cpu().numpy())
    score = cos_scores[task_idx]
    if score > 0.7:
        return tasks_data["index_to_task"][task_idx]
    else:
        return "non_match"
    
def find_query_type(query: str):
    query_embedding = ml_models["embedder"].encode(query, convert_to_tensor=True)
    task = find_similar_task(query_embedding, tasks_data["embeddings"])
    return task

def prepare_query(query: str, task: str):
    if task in ["semantic_search"]:
        query_prep = list()
        for token in ml_models["tokenizer"](query):
            if token.normalized not in tasks_data["tasks_dicts"]:
                query_prep.append(token.value)
        query_prep = " ".join(query_prep)
    elif task == "summurize_speaker":
        query_prep = re.findall("\d{1,2}", query)
        if len(query_prep) == 0:
            query_prep = ""
        else:
            query_prep = query_prep[0]
    else:
        query_prep = ""
    return query_prep


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["asr_model"] = ASRModel()
    ml_models["sum_model"] = SummaryModel()
    ml_models["task_model"] = TasksModel()
    ml_models["embedder"] = SentenceTransformer("all-MiniLM-L6-v2")
    ml_models["tokenizer"] = yargy.tokenizer.MorphTokenizer()
    ml_models["search_model"] = SearchModel(ml_models["embedder"])
    index_to_task, tasks_embeddings, tasks_dicts = prepare_tasks(ml_models["embedder"], ml_models["tokenizer"])
    tasks_data["embeddings"] = tasks_embeddings
    tasks_data["index_to_task"] = index_to_task
    tasks_data["tasks_dicts"] = tasks_dicts
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/transcribe/")
async def api_data(filepath: str, id: int):

    result = ml_models["asr_model"].process(filepath)

    if "error" in result:
        raise HTTPException(status_code=404, detail="Item not found")

    utterances = list()
    speakers = list()
    for item in result["segments"]:
        utterances.append(item['text'])
        speakers.append(item["speaker"])

    if os.path.exists(filepath):
        # with open("/home/greenatom-admin/TEST/results.json", "r", encoding="utf-8") as f:
        #     d = json.load(f)
        #     data = list()
        #     speakers = extract_speakers(d)
        #     for item in d:
        #         item_clean = {**item}
        #         item_clean.pop("words")
        #         data.append(item_clean)
        #     with open("./sum.txt", "r") as f:
        #         short = f.read()

        print("--- translating ---")
        ru_en_tr = TranslationModel("Helsinki-NLP/opus-mt-ru-en")
        en_ru_tr = TranslationModel("Helsinki-NLP/opus-mt-en-ru")

        print("--- summarizing ---")
        utterances_en = ru_en_tr.translate(utterances)
        summaries = ml_models["sum_model"].summarize_utterances(utterances_en, speakers)

        print("--- translating ---")
        summaries_ru = en_ru_tr.translate(summaries)
        summaries_ru = [correct_translation(x) for x in summaries_ru]

        full = result["segments"]
        # short = list()
        short = "\n".join(summaries_ru)
        speakers_set = result["speakers"]

        print("--- parsing facts ---")
        tasks_full = ml_models["task_model"].parse(utterances)
        tasks_short = ml_models["task_model"].parse(summaries_ru)
        tasks = list()
        for item in tasks_full + tasks_short:
            tasks.append(" ".join([value for value in item.values()]))

        filename = os.path.splitext(os.path.basename(filepath))[0]
        file_dir = os.path.join(results_root, filename)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        # with open(os.path.join(file_dir, "full.json"), "w") as f:
        #     json.dump(full, f, ensure_ascii=False, indent=4)
        with open(os.path.join(file_dir, "utterances_en.json"), "w") as f:
            json.dump(utterances_en, f, ensure_ascii=False, indent=4)
        with open(os.path.join(file_dir, "utterances.json"), "w") as f:
            json.dump(utterances, f, ensure_ascii=False, indent=4)
        with open(os.path.join(file_dir, "speakers.json"), "w") as f:
            json.dump(speakers, f, ensure_ascii=False, indent=4)

        print("--- finish ---")

        r = requests.post("http://158.160.7.241:8004/api/v1/ml-results/", json={"id":id, "full": full, "short": short, "task": tasks, "speakers": speakers_set})
        print(r.status_code)
        print(r.content)
        if r:
        # return {"full": json.dumps(data, ensure_ascii=False), "short": short, "task": "", "speakers": speakers}
            return {"status": "OK", "id": id}
    else:
        return {"filepath": filepath, "status": "ne OK"}
    

@app.get("/chat/")
def chat_interaction(query: str, filename: str):
    print(filename)
    path_to_utterances_en = os.path.join(results_root, os.path.splitext(filename)[0], "utterances_en.json")
    path_to_utterances = os.path.join(results_root, os.path.splitext(filename)[0], "utterances.json")
    path_to_speakers = os.path.join(results_root, os.path.splitext(filename)[0], "speakers.json")

    if not (os.path.exists(path_to_utterances_en) 
            and os.path.exists(path_to_utterances) 
            and os.path.exists(path_to_speakers)):
        return {"status": "Transcription results doesn't exists for file {filename}"}
    task = find_query_type(query)
    with open(path_to_utterances_en, "r") as f:
        utterances_en = json.load(f)
    with open(path_to_utterances, "r") as f:
        utterances = json.load(f)
    with open(path_to_speakers, "r") as f:
        speakers = json.load(f)

    query_prep = prepare_query(query, task)
    print(query_prep)
    if task in "summurize_speaker":
        speaker = None
        for s in speakers:
            print(s)
            if query_prep in s:
                speaker = s
                break
            return {"status": "Can't find speaker with name/id"}
        summary = ml_models["sum_model"].summarize_utterances(utterances_en, speakers, speaker)
        en_ru_tr = TranslationModel("Helsinki-NLP/opus-mt-en-ru")
        summary_ru = en_ru_tr.translate(summary)
        summary_ru = [correct_translation(x) for x in summary_ru]
        return {"status": "OK", "result": summary_ru}
    elif task == "semantic_search":
        ru_en_tr = TranslationModel("Helsinki-NLP/opus-mt-ru-en")
        query_en = ru_en_tr.translate([query_prep])[0]
        result = ml_models["search_model"].get_top_similar(query_en, utterances, utterances_en)
        return {"status": "OK", "result": result}
    elif task == "participants":
        return {"status": "OK", "result": speakers}
    else:
        return {"status": "Can't find appropriate task"}