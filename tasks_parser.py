import re
import yargy
from navec import Navec
from slovnet import Syntax, Morph
from razdel import sentenize
from tqdm import tqdm


def validate_features(features: dict, target_features: dict):
    return all([features.get(key, None) == value for key, value in target_features.items()])

def find_anchor(sentences: list, markup_morph: list, patterns: list, target_features: dict):
    anchors = list()
    for i, (sentence, morph_tags) in enumerate(zip(sentences, markup_morph)):
        anchors_cur = list()
        for j, (token, morph_tag) in enumerate(zip(sentence, morph_tags.tokens)):  
            # print(morph_tag)         
            if (token in patterns 
                and validate_features(morph_tag.feats, target_features)):
                anchors_cur.append(j)
        # print(anchors_cur)
        anchors.append(anchors_cur)
    return anchors

def check_task_past(anchors: list, markup_synt: list, markup_morph: list):
    tasks = list()
    for anchors_cur, synt_tags, morph_tags in zip(anchors, markup_synt, markup_morph):
        for anchor_idx in anchors_cur:
            anchor_id = synt_tags.tokens[anchor_idx].id
            childs = {
                "subject": None, "object": None, "xcomp": None, "iobject": None
            }
            for node_synt, node_morph in zip(synt_tags.tokens, morph_tags.tokens):
                # print(anchor_id, node_synt.head_id)
                if node_synt.head_id == anchor_id:
                    print(node_synt)
                    if "subj" in node_synt.rel:
                        childs["subject"] = node_synt
                    elif ("obj" in node_synt.rel 
                          and validate_features(node_morph.feats, {"Animacy": "Anim"})):
                        childs["object"] = node_synt
                    elif "xcomp" in node_synt.rel:
                        childs["xcomp"] = node_synt
            if childs["xcomp"] is not None:
                for node_synt in synt_tags.tokens:
                    if (node_synt.head_id == childs["xcomp"].id
                        and node_synt.rel == "obj"):
                        childs["iobject"] = node_synt
            
            # childs_new = {
            #     "subject": childs["subject"].text,
            #     "object": childs["object"].text,
            #     "xcomp": childs["xcomp"].text,
            #     "iobject": childs["iobject"].text,
            #     "action": synt_tags.tokens[anchor_idx].text
            # }
            childs_new = dict()
            for key, value in childs.items():
                if value is not None:
                    childs_new[key] = value.text
            childs_new["action"] = synt_tags.tokens[anchor_idx].text

            # if all(childs.values()):
            #     tasks.append(childs)
            if any(childs.values()):
                tasks.append(childs_new)
    return tasks


def check_task_imp(anchors: list, markup_synt: list, markup_morph: list):
    tasks = list()
    for anchors_cur, synt_tags, morph_tags in zip(anchors, markup_synt, markup_morph):
        for anchor_idx in anchors_cur:
            anchor_id = synt_tags.tokens[anchor_idx].id
            childs = {
                "subject": None, "object": None, "xcomp": None, "iobject": None
            }
            for node_synt, node_morph in zip(synt_tags.tokens, morph_tags.tokens):
                if node_synt.head_id == anchor_id:
                    # print(node_synt, synt_tags.tokens[anchor_idx])
                    if "obj" in node_synt.rel:
                        childs["object"] = node_synt
                    elif "subj" in node_synt.rel:
                        childs["subject"] = node_synt
                # elif synt_tags.tokens[anchor_idx].head_id == node_synt.id:
                #     if "subj" in synt_tags.tokens[anchor_idx].rel:
                #         childs["subject"] = node_synt
            childs_new = dict()
            for key, value in childs.items():
                if value is not None:
                    childs_new[key] = value.text
            childs_new["action"] = synt_tags.tokens[anchor_idx].text
            if any(childs.values()):
                tasks.append(childs_new)
    return tasks


class TasksModel:
    def __init__(self):
        self.tokenizer = yargy.tokenizer.MorphTokenizer()
        self.patterns_past = [
            "попросил", "сказал"
        ]
        self.past_features = {
            "Tense": "Past"
        }
        self.patterns_past = [x.normalized for x in self.tokenizer(" ".join(self.patterns_past))]
        self.patterns_imp = [
            "сделай", "выполни"
        ]
        self.imp_features = {
            'Aspect': "Perf",
            'Mood': 'Imp'
        }
        self.patterns_imp = [x.normalized for x in self.tokenizer(" ".join(self.patterns_imp))]

        self.navec = Navec.load('navec_news_v1_1B_250K_300d_100q.tar')
        self.morph = Morph.load('slovnet_morph_news_v1.tar', batch_size=4)
        self.morph.navec(self.navec)
        self.syntax = Syntax.load('slovnet_syntax_news_v1.tar')
        self.syntax.navec(self.navec);

    def parse(self, utterances: list):
        
        tasks = list()
        for utterance in tqdm(utterances):
            sentences, sentences_norm = list(), list()
            for sent in sentenize(utterance):
                tokens, tokens_norm = list(), list()
                for _ in self.tokenizer(re.sub("\W+|\d+|_+|пожалуйста", " ", sent.text)):
                    tokens.append(_.value)
                    tokens_norm.append(_.normalized)
                sentences.append(tokens)
                sentences_norm.append(tokens_norm)

            markup_synt = list(self.syntax.map(sentences))
            markup_morph = list(self.morph.map(sentences))

            anchors = find_anchor(sentences_norm, markup_morph, self.patterns_past, self.past_features)
            tasks.extend(check_task_past(anchors, markup_synt, markup_morph))

            anchors = find_anchor(sentences_norm, markup_morph, self.patterns_imp, self.imp_features)
            tasks.extend(check_task_imp(anchors, markup_synt, markup_morph))
        return tasks