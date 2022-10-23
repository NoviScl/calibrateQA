import json
from tkinter import FALSE
from tkinter.tix import Tree
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 
import torch
from sklearn.linear_model import LogisticRegression
import pickle
from collections import Counter

import re 
import string
np.random.seed(2022)

'''
dict_keys(['predicted_start_index', 'predicted_end_index', 'answer_text', 'answer_ids', 
'passage_index', 'passage_text', 'passage_ids', 'passage_logit_raw', 'passage_logit_log_softmax', 
'start_logit_raw', 'start_logit_log_softmax', 'end_logit_raw', 'end_logit_log_softmax', 
'log_softmax', 'gold_answer', 'score', 'question'])

format: a list of len N, where N is the number of questions. Each element is a list of length 100, 
where each element is an answer prediction dict.
'''

## util functions
def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def softmax(x):
    ## minus max for more numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
    # return np.exp(x) / np.sum(np.exp(x), axis=0)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def num_correct(psg_lst):
    '''
    Given a list of predictions, return the number of correct ones.
    '''
    score = 0
    for dp in psg_lst:
        score += int(dp['score'])
    return score

def get_bucket_scores_and_confidence(all_scores, all_probs, buckets=10):
    '''
    all_scores: EM scores of all predictions.
    all_probs: confidence scores for all predictions.
    buckets: number of buckets.
    '''
    bucket_probs = [[] for _ in range(buckets)]
    bucket_scores = [[] for _ in range(buckets)]
    for i, prob in enumerate(all_probs):
        for j in range(buckets):
            if prob < float((j+1) / buckets):
                break
        bucket_probs[j].append(prob)
        bucket_scores[j].append(all_scores[i])
    
    per_bucket_confidence = [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_probs
    ]

    per_bucket_score = [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_scores
    ]

    bucket_sizes = [
        len(bucket) 
        for bucket in bucket_scores
    ]
    
    return per_bucket_score, per_bucket_confidence, bucket_sizes

def get_bucket_scores_and_confidence_by_density(all_scores, all_probs, buckets=10):
    '''
    all_scores: EM scores of all predictions.
    all_probs: confidence scores for all predictions.
    buckets: number of buckets.
    Group the predictions into K equally-sized buckets, and then take average of the ECE for each bucket.
    '''
    bucket_probs = [[] for _ in range(buckets)]
    bucket_scores = [[] for _ in range(buckets)]
    sorted_probs_idx = np.argsort(all_probs)
    all_probs = [all_probs[i] for i in sorted_probs_idx]
    all_scores = [all_scores[i] for i in sorted_probs_idx]
    each_bucket_size = len(all_probs) // buckets
    for i in range(buckets):
        for j in range(i*each_bucket_size, (i+1)*each_bucket_size):
            bucket_probs[i].append(all_probs[j])
            bucket_scores[i].append(all_scores[j])
        if i == (buckets - 1):
            ## add all remaining ones to the last bucket
            bucket_probs[i].extend(all_probs[(i+1)*each_bucket_size : ])
            bucket_scores[i].extend(all_scores[(i+1)*each_bucket_size : ])
    
    per_bucket_confidence = [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_probs
    ]

    per_bucket_score = [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_scores
    ]

    bucket_sizes = [
        len(bucket) 
        for bucket in bucket_scores
    ]
    
    return per_bucket_score, per_bucket_confidence, bucket_sizes

def ECE(per_bucket_score, per_bucket_confidence, bucket_sizes):
    n_samples = sum(bucket_sizes)
    ece = 0.
    for i in range(len(bucket_sizes)):
        if bucket_sizes[i] > 0:
            delta = abs(per_bucket_score[i] - per_bucket_confidence[i])
            ece += (bucket_sizes[i] / n_samples) * delta
    return ece * 100

def instance_ECE(all_scores, all_probs):
    # compute instance level ECE
    total_error = 0
    for i in range(len(all_probs)):
        total_error += abs(all_scores[i] - all_probs[i])
    if len(all_scores) == 0:
        return 0
    return total_error / len(all_probs) * 100

def category_ECE(all_scores, all_probs):
    # compute the average of positive-example-ECE and negative-example-ECE
    pos_probs = []
    pos_scores = []
    neg_probs = []
    neg_scores = []
    for i in range(len(all_scores)):
        if all_scores[i] == 1:
            pos_probs.append(all_probs[i])
            pos_scores.append(all_scores[i])
        else:
            neg_probs.append(all_probs[i])
            neg_scores.append(all_scores[i])
    pos_ece = instance_ECE(pos_scores, pos_probs)
    neg_ece = instance_ECE(neg_scores, neg_probs)
    return pos_ece, neg_ece, (pos_ece + neg_ece) / 2

def cross_entropy(output, target):
    '''
    output: softmax distribution
    target: one hot labels
    '''
    return F.kl_div(output, target, reduction='sum').item()


class Calibrator:
    def __init__(self, score_func=2, answer_agg=False, prob="softmax", pipeline=False, printing=True, reranking=False, match_ans=False, bart=False, \
            ece_type="interval", task="qa"):
        self.dev_set = None 
        self.test_set = None 
        self.score_func = score_func # {1, 2, 3}
        self.answer_agg = answer_agg # {True, False}
        self.prob = prob # {"softmax", "sigmoid"}
        # self.calibrator = calibrator # {"NO", "TS", "BC"}
        self.temperature = 1.0
        self.printing = printing # whether to print results or not
        self.pipeline = pipeline
        self.reranking = reranking # whether to calibrate reranking or not
        self.match_ans = match_ans # whether to re-do answer matching for each span prediction
        self.bart = bart
        self.ece_type = ece_type # options: {interval, density, instance, category}
        self.group_threshold = 0 # used to split predictions into groups
        self.temp_a = 1.0 # temperature for the first group
        self.temp_b = 1.0 # temperature for the second group
        self.task = task

        ## logits: the raw score of each span
        ## scores: whether each span is correct or not (0/1)
        ## spans: the answer strings
        self.dev_logits = []
        self.dev_scores = []
        self.dev_spans = []

        self.test_logits = []
        self.test_scores = []
        self.test_spans = []

        ## extracted top prediction for each question
        self.dev_top_scores = []
        self.dev_top_probs = []
        self.dev_top_idxes = []

        self.test_top_scores = []
        self.test_top_probs = []
        self.test_top_idxes = []
    
    def load_dev(self, data_dir="all_predictions_dev.json"):
        with open(data_dir, "r") as f:
            self.dev_set = json.load(f)
        if self.printing:
            print ("dev set loaded: ", data_dir)
    
    def load_test(self, data_dir="all_predictions_test.json"):
        with open(data_dir, "r") as f:
            self.test_set = json.load(f)
        if self.printing:
            print ("test set loaded: ", data_dir)
    
    def score_spans(self, data):
        all_logits = []
        all_scores = []
        all_spans = []
        for i in range(len(data)):
            all_logits.append([])
            all_scores.append([])
            all_spans.append([])

            if self.score_func in [4, 6]:
                start_logits = softmax([dp["start_logit_raw"] / self.temperature for dp in data[i]])
                end_logits = softmax([dp["end_logit_raw"] / self.temperature for dp in data[i]])
            elif self.score_func in [3, 5]:
                start_logits = softmax([dp["start_logit_raw"] for dp in data[i]])
                end_logits = softmax([dp["end_logit_raw"] for dp in data[i]])
            
            if self.score_func == 4:
                psg_logits = {}
                for dp in data[i]:
                    psg_logits[dp["passage_index"]] = dp["passage_logit_raw"] / self.temperature
                psg_logits_softmax = softmax(list(psg_logits.values()))
                p = 0
                for k,v in psg_logits.items():
                    psg_logits[k] = psg_logits_softmax[p]
                    p += 1
            elif self.score_func == 3:
                psg_logits = {}
                for dp in data[i]:
                    psg_logits[dp["passage_index"]] = dp["passage_logit_raw"] 
                psg_logits_softmax = softmax(list(psg_logits.values()))
                p = 0
                for k,v in psg_logits.items():
                    psg_logits[k] = psg_logits_softmax[p]
                    p += 1

            for j in range(len(data[i])):
                dp = data[i][j]
                if self.score_func == 1:
                    span_score = \
                        dp["start_logit_raw"] + dp["end_logit_raw"]
                elif self.score_func == 2:
                    span_score = \
                        dp["passage_logit_raw"] + dp["start_logit_raw"] + dp["end_logit_raw"]
                elif self.score_func == 3:
                    span_score = \
                        start_logits[j] * end_logits[j] * psg_logits[dp["passage_index"]]
                        # np.exp(dp["start_logit_log_softmax"] + dp["end_logit_log_softmax"] + dp["passage_logit_log_softmax"])
                elif self.score_func == 4:
                    span_score = \
                        start_logits[j] * end_logits[j] * psg_logits[dp["passage_index"]]
                elif self.score_func == 5:
                    span_score = \
                        start_logits[j] * end_logits[j]
                        # np.exp(dp["start_logit_log_softmax"] + dp["end_logit_log_softmax"])
                elif self.score_func == 6:
                    span_score = \
                        start_logits[j] * end_logits[j]
                else:
                    raise NotImplementedError("Unknown scoring function")
                
                ans_score = int(dp["score"])
                if self.match_ans:
                    ans_score = 0
                    for ans in list(set(dp["gold_answer"])):
                        if normalize_answer(dp["answer_text"]) == normalize_answer(ans):
                            ans_score = 1
                            break
                all_logits[-1].append(span_score)
                all_scores[-1].append(ans_score)
                all_spans[-1].append(dp["answer_text"])
            
        return all_logits, all_scores, all_spans

    def score_spans_pipeline(self, data):
        ## for each question, only save the spans in the top passage
        all_logits = []
        all_scores = []
        all_spans = []
        for i in range(len(data)):
            all_logits.append([])
            all_scores.append([])
            all_spans.append([])

            ## first find best passage 
            passage_scores = [dp["passage_logit_raw"] for dp in data[i]]
            best_psg_idx = np.argmax(passage_scores)
            best_psg = data[i][best_psg_idx]["passage_index"]

            if self.score_func in [4, 6]:
                start_logits = softmax([dp["start_logit_raw"] / self.temperature for dp in data[i]])
                end_logits = softmax([dp["end_logit_raw"] / self.temperature for dp in data[i]])
            elif self.score_func in [3, 5]:
                start_logits = softmax([dp["start_logit_raw"] for dp in data[i]])
                end_logits = softmax([dp["end_logit_raw"] for dp in data[i]])
            
            if self.score_func == 4:
                psg_logits = {}
                for dp in data[i]:
                    psg_logits[dp["passage_index"]] = dp["passage_logit_raw"] / self.temperature
                psg_logits_softmax = softmax(list(psg_logits.values()))
                p = 0
                for k,v in psg_logits.items():
                    psg_logits[k] = psg_logits_softmax[p]
                    p += 1
            elif self.score_func == 3:
                psg_logits = {}
                for dp in data[i]:
                    psg_logits[dp["passage_index"]] = dp["passage_logit_raw"] 
                psg_logits_softmax = softmax(list(psg_logits.values()))
                p = 0
                for k,v in psg_logits.items():
                    psg_logits[k] = psg_logits_softmax[p]
                    p += 1

            for j in range(len(data[i])):
                dp = data[i][j]
                if dp["passage_index"] != best_psg:
                    continue
                if self.score_func == 1:
                    span_score = \
                        dp["start_logit_raw"] + dp["end_logit_raw"]
                elif self.score_func == 2:
                    span_score = \
                        dp["passage_logit_raw"] + dp["start_logit_raw"] + dp["end_logit_raw"]
                elif self.score_func == 3:
                    span_score = \
                        start_logits[j] * end_logits[j] * psg_logits[dp["passage_index"]]
                        # np.exp(dp["start_logit_log_softmax"] + dp["end_logit_log_softmax"] + dp["passage_logit_log_softmax"])
                elif self.score_func == 4:
                    span_score = \
                        start_logits[j] * end_logits[j] * psg_logits[dp["passage_index"]]
                elif self.score_func == 5:
                    span_score = \
                        start_logits[j] * end_logits[j]
                        # np.exp(dp["start_logit_log_softmax"] + dp["end_logit_log_softmax"])
                elif self.score_func == 6:
                    span_score = \
                        start_logits[j] * end_logits[j]
                else:
                    raise NotImplementedError("Unknown scoring function")
                
                ans_score = int(dp["score"])
                if self.match_ans:
                    ans_score = 0
                    for ans in list(set(dp["gold_answer"])):
                        if normalize_answer(dp["answer_text"]) == normalize_answer(ans):
                            ans_score = 1
                            break
                all_logits[-1].append(span_score)
                all_scores[-1].append(ans_score)
                all_spans[-1].append(dp["answer_text"])
            
        return all_logits, all_scores, all_spans

    def score_passages(self, data):
            ## for each question, extract the k(=10) passages' scores and confidences
            all_logits = []
            all_scores = []
            retrieved = 0
            for q in range(len(data)):
                psg_logits = [0] * 10
                psg_scores = [0] * 10
                for psg_idx in range(10):
                    found = False
                    score = 0
                    for dp in data[q]:
                        if dp["passage_index"] == psg_idx:
                            found = True
                            psg_logits[psg_idx] = dp["passage_logit_raw"]
                            ## score of psg
                            for ans in dp["gold_answer"]:
                                if normalize_answer(ans) in normalize_answer(dp["passage_text"]):
                                    score = 1
                                    break
                            break
                    assert found == True, "passage missing"
                    psg_scores[psg_idx] = score

                all_logits.append(psg_logits)
                all_scores.append(psg_scores) 

                # print ("question: ", q)
                # print ("psg logits: ", psg_logits)
                # print ("psg scores: ", psg_scores)
                # print ()
                if sum(psg_scores) >= 1:
                    retrieved += 1
            if self.printing:
                print ("retriever top-10: {}/{}={}%".format(retrieved, len(data), retrieved/len(data) * 100))
            return all_logits, all_scores

    def score_spans_bart(self, data):
        all_logits = []
        all_scores = []
        all_spans = []
        for dq in data:
            if self.score_func != "softmax":
                all_scores.append(dq["score"])
                all_spans.append(dq["decoded string"])

            ## compute score
            if self.score_func == "average":
                score = 0
                counter = 0
                for i in range(len(dq["ids"])):
                    if dq["ids"][i] > 3:
                        score += dq["logits"][i]
                        counter += 1
                all_logits.append(score / counter)
            elif self.score_func == "product":
                score = 1.0
                for i in range(len(dq["ids"])):
                    if dq["ids"][i] > 3:
                        score *= dq["logits"][i]
                all_logits.append(score)
            elif self.score_func == "norm_product":
                score = 1.0
                counter = 0
                for i in range(len(dq["ids"])):
                    if dq["ids"][i] > 3:
                        score *= dq["logits"][i]
                        counter += 1
                all_logits.append(score ** (1 / counter))
            elif self.score_func == "softmax":
                raw_logits = [d["logits"] for d in dq]
                raw_scores = [d["score"] for d in dq]
                raw_preds = [d["decoded string"] for d in dq]
                softmax_logits = softmax([logit / self.temperature for logit in raw_logits])
                idx = np.argmax(softmax_logits)
                all_logits.append(softmax_logits[idx])
                all_scores.append(raw_scores[idx])
                all_spans.append(raw_preds[idx])

        return all_logits, all_scores, all_spans

    def score_spans_data(self, data):
        assert data != None, "no data!"

        if self.reranking:
            return self.score_passages(self.dev_set)
        elif self.bart:
            return self.score_spans_bart(self.dev_set)
        elif self.pipeline:
            return self.score_spans_pipeline(self.dev_set)
        else:
            return self.score_spans(self.dev_set)

    def score_spans_dev(self):
        assert self.dev_set != None, "dev set is not loaded yet!"
        
        if self.reranking:
            self.dev_logits, self.dev_scores = self.score_passages(self.dev_set)
        elif self.bart:
            self.dev_logits, self.dev_scores, self.dev_spans = self.score_spans_bart(self.dev_set)
        elif self.pipeline:
            self.dev_logits, self.dev_scores, self.dev_spans = self.score_spans_pipeline(self.dev_set)
        else:
            self.dev_logits, self.dev_scores, self.dev_spans = self.score_spans(self.dev_set)
        if self.printing:
            print ("dev spans/passages scored")
    
    def score_spans_test(self):
        assert self.test_set != None, "test set is not loaded yet!"

        if self.reranking:
            self.test_logits, self.test_scores = self.score_passages(self.test_set)
        elif self.bart:
            self.test_logits, self.test_scores, self.test_spans = self.score_spans_bart(self.test_set)
        elif self.pipeline:
            self.test_logits, self.test_scores, self.test_spans = self.score_spans_pipeline(self.test_set)
        else:
            self.test_logits, self.test_scores, self.test_spans = self.score_spans(self.test_set)

        if self.printing:
            print ("test spans/passages scored")
    
    def answer_aggregation(self, data_logits, data_scores, data_spans):
        all_logits = []
        all_scores = []
        all_spans = []
        
        for i in range(len(data_logits)):
            span_logits = {}
            span_scores = {}
            for j in range(len(data_logits[i])):
                span = normalize_answer(data_spans[i][j])
                score = data_scores[i][j]
                logit = data_logits[i][j]

                if span not in span_logits:
                    span_logits[span] = logit
                    span_scores[span] = score 
                else:
                    span_logits[span] += logit 
                    span_scores[span] = max(score, span_scores[span])
            
            all_logits.append([])
            all_scores.append([])
            all_spans.append([])

            for k,v in span_logits.items():
                all_logits[-1].append(v)
                all_scores[-1].append(span_scores[k])
                all_spans[-1].append(k)

        return all_logits, all_scores, all_spans
    
    def answer_agg_dev(self):
        assert len(self.dev_logits) > 0, "dev set is not loaded yet!"
        self.dev_logits, self.dev_scores, self.dev_spans = self.answer_aggregation(self.dev_logits, self.dev_scores, self.dev_spans)
        if self.printing:
            print ("answer aggregation on dev set done")
    
    def answer_agg_test(self):
        assert len(self.test_logits) > 0, "test set is not loaded yet!"
        self.test_logits, self.test_scores, self.test_spans = self.answer_aggregation(self.test_logits, self.test_scores, self.test_spans)
        if self.printing:
            print ("answer aggregation on test set done")

    def extract_top_span(self, data_logits, data_scores, temperature=1.0):
        top_scores = []
        top_probs = []
        top_idxes = []

        for i in range(len(data_logits)):
            raw_logits = data_logits[i]
            if self.prob == "sigmoid":
                logits = [sigmoid(logit / temperature) for logit in raw_logits]
            elif self.prob == "softmax":
                logits = [logit / temperature for logit in raw_logits]
                logits = softmax(logits)
            elif self.prob == "signorm":
                mean = np.mean(raw_logits)
                print (raw_logits)
                print (mean)
                logits = [sigmoid((logit - mean) / temperature) for logit in raw_logits]
                print (logits)
                print ()
            elif self.prob == "none":
                logits = raw_logits
            else:
                raise NotImplementedError("unknown prob function")
            
            top_idx = np.argmax(logits)
            top_idxes.append(top_idx)
            top_probs.append(logits[top_idx])
            top_scores.append(data_scores[i][top_idx])
        
        return top_probs, top_scores, top_idxes

    def top_spans_data(self, data_logits, data_scores, temperature):
        top_probs, top_scores, _ = self.extract_top_span(data_logits, data_scores, temperature)
        return top_probs, top_scores

    def top_span_dev(self):
        if self.task == "qa":
            self.score_spans_dev()
        if self.answer_agg:
            self.answer_agg_dev()
        if self.bart:
            self.dev_top_probs = self.dev_logits 
            self.dev_top_scores = self.dev_scores
        else:
            self.dev_top_probs, self.dev_top_scores, self.dev_top_idxes = self.extract_top_span(self.dev_logits, self.dev_scores, temperature=self.temperature)
    
    def top_span_test(self):
        if self.task == "qa":
            self.score_spans_test()
        if self.answer_agg:
            self.answer_agg_test()
        if self.bart:
            self.test_top_probs = self.test_logits 
            self.test_top_scores = self.test_scores
        else:
            self.test_top_probs, self.test_top_scores, self.test_top_idxes = self.extract_top_span(self.test_logits, self.test_scores, temperature=self.temperature)

        # ## avg calibration
        # self.test_top_probs = [np.mean(self.test_top_scores)] * len(self.test_top_probs)
        # print ("test top probs: ", self.test_top_probs[:20])

        # ## 0/1 calibration
        # acc = np.mean(self.test_top_scores)
        # sorted_probs = sorted(self.test_top_probs)[::-1]
        # threshold = sorted_probs[int(len(self.test_top_probs) * acc)]
        # new_test_top_probs = []
        # for p in self.test_top_probs:
        #     if p >= threshold:
        #         new_test_top_probs.append(1)
        #     else:
        #         new_test_top_probs.append(0)
        # self.test_top_probs = new_test_top_probs
        # print ("threshold: ", threshold)
        # print ("test top probs: ", self.test_top_probs[:20])

    def dev_ece(self):
        if self.ece_type == "category":
            pos_ece, neg_ece, ece = category_ECE(self.dev_top_scores, self.dev_top_probs)
            if self.printing:
                print ("dev set")
                print ("EM: ", np.mean(self.dev_top_scores) * 100)
                print ("pos ECE: ", pos_ece)
                print ("neg ECE: ", neg_ece)
                print ("average ECE: ", ece)
                print ()
        else:
            if self.ece_type == "interval":
                per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(self.dev_top_scores, self.dev_top_probs)
                ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
            elif self.ece_type == "density":
                per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence_by_density(self.dev_top_scores, self.dev_top_probs)
                ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
            elif self.ece_type == "instance":
                ece = instance_ECE(self.dev_top_scores, self.dev_top_probs)
            if self.printing:
                print ("dev set")
                print ("EM: ", np.mean(self.dev_top_scores) * 100)
                if self.ece_type in ["interval", "density"]:
                    print ("Acc: ", [round(num, 2) for num in per_bucket_score])
                    print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
                    print ("Sizes: ", bucket_sizes)
                print ("dev ECE: ", ece)
                print ()
        return ece

    def test_ece(self):
    
        if "interval" in self.ece_type:
            per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(self.test_top_scores, self.test_top_probs)
            ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
        
            if self.printing:
                print ("test set interval")
                print ("EM: ", np.mean(self.test_top_scores) * 100)
                if "interval" in self.ece_type or "density" in self.ece_type:
                    print ("Acc: ", [round(num, 2) for num in per_bucket_score])
                    print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
                    print ("Sizes: ", bucket_sizes)
                print ("test ECE: ", ece)
                print ()
        
        if "density" in self.ece_type:
            per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence_by_density(self.test_top_scores, self.test_top_probs)
            ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
        
            if self.printing:
                print ("test set density")
                print ("EM: ", np.mean(self.test_top_scores) * 100)
                if "interval" in self.ece_type or "density" in self.ece_type:
                    print ("Acc: ", [round(num, 2) for num in per_bucket_score])
                    print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
                    print ("Sizes: ", bucket_sizes)
                print ("test ECE: ", ece)
                print ()
        
        
        if "instance" in self.ece_type:
            ece = instance_ECE(self.test_top_scores, self.test_top_probs)
        
            if self.printing:
                print ("test set instance")
                print ("test ECE: ", ece)
                print ()
        
        if "category" in self.ece_type:
            pos_ece, neg_ece, ece = category_ECE(self.test_top_scores, self.test_top_probs)
            if self.printing:
                print ("test set category")
                print ("EM: ", np.mean(self.test_top_scores) * 100)
                print ("pos ECE: ", pos_ece)
                print ("neg ECE: ", neg_ece)
                print ("average ECE: ", ece)
                print ()
        
        
        
        return ece

    def rank(self, scores, probs):
        ## rank predictions by probabilities and print the coverage at each accuracy
        sorted_idx = np.argsort(probs)[::-1]
        correct = 0
        total = 0
        pct_90 = False
        pct_80 = False
        pct_70 = False
        pct_60 = False
        pct_50 = False
        pct_40 = False
        pct_30 = False
        pct_20 = False
        
        counter = 0
        for idx in sorted_idx:
            counter += 1
            total += 1
            correct += scores[idx]
            pct = correct / total 
            # print (probs[idx], scores[idx])
            if counter > 100:
                ## give some buffer (20) in case the first examples are wrong
                if pct < 0.9 and not pct_90:
                    pct_90 = True 
                    print ("Cov@90%: {}/{}={}%".format(total, len(scores), total / len(scores) * 100))
                if pct < 0.8 and not pct_80:
                    pct_80 = True 
                    print ("Cov@80%: {}/{}={}%".format(total, len(scores), total / len(scores) * 100))
                if pct < 0.7 and not pct_70:
                    pct_70 = True 
                    print ("Cov@70%: {}/{}={}%".format(total, len(scores), total / len(scores) * 100))
                if pct < 0.6 and not pct_60:
                    pct_60 = True 
                    print ("Cov@60%: {}/{}={}%".format(total, len(scores), total / len(scores) * 100))
                if pct < 0.5 and not pct_50:
                    pct_50 = True 
                    print ("Cov@50%: {}/{}={}%".format(total, len(scores), total / len(scores) * 100))
                if pct < 0.4 and not pct_40:
                    pct_40 = True 
                    print ("Cov@40%: {}/{}={}%".format(total, len(scores), total / len(scores) * 100))
                if pct < 0.3 and not pct_30:
                    pct_30 = True 
                    print ("Cov@30%: {}/{}={}%".format(total, len(scores), total / len(scores) * 100))
                if pct < 0.2 and not pct_20:
                    pct_20 = True 
                    print ("Cov@20%: {}/{}={}%".format(total, len(scores), total / len(scores) * 100))
                    break
                    
    def test_ece_zero_correct(self):
        test_scores = []
        test_probs = []
        for i in range(len(self.test_set)):
            psg_lst = self.test_set[i]
            correct = num_correct(psg_lst)
            if correct == 0:
                test_scores.append(self.test_top_scores[i])
                test_probs.append(self.test_top_probs[i])
        print ("#questions with ZERO correct spans: ", len(test_scores))
        per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(test_scores, test_probs)
        ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
        print ("Acc: ", [round(num, 2) for num in per_bucket_score])
        print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
        print ("Sizes: ", bucket_sizes)
        print ("test ZERO correct subset ECE: ", ece)
    
    def test_ece_one_correct(self):
        test_scores = []
        test_probs = []
        for i in range(len(self.test_set)):
            psg_lst = self.test_set[i]
            correct = num_correct(psg_lst)
            if correct == 1:
                test_scores.append(self.test_top_scores[i])
                test_probs.append(self.test_top_probs[i])
        print ("#questions with ONE correct spans: ", len(test_scores))
        per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(test_scores, test_probs)
        ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
        print ("Acc: ", [round(num, 2) for num in per_bucket_score])
        print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
        print ("Sizes: ", bucket_sizes)
        print ("test ONE correct subset ECE: ", ece)

    def test_ece_more_correct(self):
        test_scores = []
        test_probs = []
        for i in range(len(self.test_set)):
            psg_lst = self.test_set[i]
            correct = num_correct(psg_lst)
            if correct > 1:
                test_scores.append(self.test_top_scores[i])
                test_probs.append(self.test_top_probs[i])
        print ("#questions with > 1 correct spans: ", len(test_scores))
        per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(test_scores, test_probs)
        ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
        print ("Acc: ", [round(num, 2) for num in per_bucket_score])
        print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
        print ("Sizes: ", bucket_sizes)
        print ("test > 1 correct subset ECE: ", ece)

    def search_temperature(self):
        best_ece = float('inf')
        best_temp = -1
        temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))
        for temp in tqdm(temp_values):
            # print ("temperature: ", temp)
            self.temperature = temp
            self.top_span_dev()
            ece = self.dev_ece()
            if ece < best_ece:
                best_ece = ece 
                best_temp = temp
        
        print ("best temp: ", best_temp)
        self.temperature = best_temp

        # self.printing = True 
        # self.top_span_dev()
        # self.dev_ece()
        # self.top_span_test()
        # self.test_ece()
    
        return best_temp
    
    def search_temperature_nll(self):
        ## search temperature by NLL / CE
        ## this is not suitable for QA because there can be 0/1/>1 correct answers
        best_nll = float('inf')
        best_temp = -1
        temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))
        for temp in temp_values:
            self.temperature = temp
            self.score_spans_dev()
            nll = 0
            for i in range(len(self.dev_logits)):
                kl = cross_entropy(F.log_softmax(torch.tensor(self.dev_logits[i]).float()  / temp, dim=0), torch.tensor(self.dev_scores[i]).float())
                nll += kl
            
                cross_entropy(F.log_softmax(torch.tensor([0.2, 0.3, 0.5]).float(), dim=0), torch.tensor([0,0,0]).float())
            if nll < best_nll:
                best_nll = nll 
                best_temp = temp
        
        print ("best temp: ", best_temp)
        self.temperature = best_temp

        self.printing = True 
        self.top_span_dev()
        self.dev_ece()
        self.top_span_test()
        self.test_ece()
    
        return best_temp

    def find_group_threshold(self):
        ## find group threshold on dev data
        dev_em = np.mean(self.dev_top_scores)
        dev_conf_sorted = sorted(self.dev_top_probs)
        self.group_threshold = dev_conf_sorted[int((1 - dev_em) * len(self.dev_top_probs))]
        print ("group confidence threshold: ", self.group_threshold)
        return self.group_threshold

    def search_temperature_by_instance(self, data_logits, data_scores):
        # search for best temperature on a group 
        # ece is computed as instance-level
        best_ece = float('inf')
        best_temp = -1
        temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))

        for temp in temp_values:
            top_probs, top_scores = self.top_spans_data(data_logits, data_scores, temp)
            _, _, ece = category_ECE(top_scores, top_probs)
            if ece < best_ece:
                best_ece = ece 
                best_temp = temp
        
        return best_temp
    
    def search_temperature_by_group(self):
        ## search for best temperature on different groups in the dev set
        self.find_group_threshold()
        group_a_idx = []
        group_b_idx = []
        # for i in range(len(self.dev_top_probs)):
        #     if self.dev_top_probs[i] >= self.group_threshold:
        #         group_a_idx.append(i)
        #     else:
        #         group_b_idx.append(i)

        ## oracle grouping
        for i in range(len(self.dev_top_probs)):
            if self.dev_top_scores[i] == 1:
                group_a_idx.append(i)
            else:
                group_b_idx.append(i)
        group_a_logits = [self.dev_logits[i] for i in group_a_idx]
        group_a_scores = [self.dev_scores[i] for i in group_a_idx]
        group_b_logits = [self.dev_logits[i] for i in group_b_idx]
        group_b_scores = [self.dev_scores[i] for i in group_b_idx]

        self.temp_a = self.search_temperature_by_instance(group_a_logits, group_a_scores)
        print ("best group_a temp: ", self.temp_a)
        self.temp_b = self.search_temperature_by_instance(group_b_logits, group_b_scores)
        print ("best group_b temp: ", self.temp_b)

    def temperature_scale_by_group(self, data="dev"):
        if data == "dev":
            data_logits = self.dev_logits
            data_scores = self.dev_scores 
            top_probs = self.dev_top_probs
            top_scores = self.dev_top_scores
        elif data == "test":
            data_logits = self.test_logits
            data_scores = self.test_scores 
            top_probs = self.test_top_probs
            top_scores = self.test_top_scores
        
        group_a_idx = []
        group_b_idx = []
        # for i in range(len(data_logits)):
        #     if top_probs[i] >= self.group_threshold:
        #         group_a_idx.append(i)
        #     else:
        #         group_b_idx.append(i)

        ## oracle grouping
        for i in range(len(data_logits)):
            if top_scores[i] == 1:
                group_a_idx.append(i)
            else:
                group_b_idx.append(i)

        group_a_logits = [data_logits[i] for i in group_a_idx]
        group_a_scores = [data_scores[i] for i in group_a_idx]
        group_b_logits = [data_logits[i] for i in group_b_idx]
        group_b_scores = [data_scores[i] for i in group_b_idx]

        group_a_top_probs, group_a_top_scores, _ = self.extract_top_span(group_a_logits, group_a_scores, self.temp_a)
        group_b_top_probs, group_b_top_scores, _ = self.extract_top_span(group_b_logits, group_b_scores, self.temp_b)
        pos, neg, avg = category_ECE(group_a_top_scores+group_b_top_scores, group_a_top_probs+group_b_top_probs)
        print ("{} set".format(data))
        print ("size of group a: ", len(group_a_top_probs))
        print ("EM group a: ", np.mean(group_a_top_scores) * 100)
        print ("avg conf group a: ", np.mean(group_a_top_probs) * 100)
        print ("size of group b: ", len(group_b_top_probs))
        print ("EM group b: ", np.mean(group_b_top_scores) * 100)
        print ("avg conf group b: ", np.mean(group_b_top_probs) * 100)
        print ("  ----- category level ECE -----  ")
        print ("EM: ", np.mean(group_a_top_scores+group_b_top_scores) * 100)
        print ("pos ECE: ", pos)
        print ("neg ECE: ", neg)
        print ("average ECE: ", avg)
        print ()

        

def eval(calibrator):
    calibrator.load_dev(data_dir="/data3/private/clsi/OpenQA/AmbigQA/out/bart_nq/all_predictions_dev_beam5.json")
    calibrator.load_test(data_dir="/data3/private/clsi/OpenQA/AmbigQA/out/bart_nq/all_predictions_test_beam5.json")
    calibrator.top_span_dev()
    calibrator.top_span_test()    
    calibrator.test_ece()
    # calibrator.test_ece_zero_correct()
    # calibrator.test_ece_one_correct()
    # calibrator.test_ece_more_correct()

def search(calibrator):
    # calibrator.load_dev(data_dir="/data3/private/clsi/OpenQA/AmbigQA/out/bart_nq/all_predictions_dev_beam5.json")
    # calibrator.load_test(data_dir="/data3/private/clsi/OpenQA/AmbigQA/out/bart_nq/all_predictions_test_beam5.json")
    calibrator.load_dev(data_dir="/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_predictions_dev_alpha09.json")
    calibrator.load_test(data_dir="/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_predictions_test_alpha09.json")
    calibrator.printing = False
    calibrator.search_temperature()

    # calibrator.test_ece_zero_correct()
    # calibrator.test_ece_one_correct()
    # calibrator.test_ece_more_correct()

def most_frequent(List):
    occurence_count = Counter(List)
    return occurence_count.most_common(1)[0][0]

if __name__ == "__main__":
    # all_dirs = ["/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_test_ckpt102000.json", \
    #     "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_8ckpts/all_predictions_test_ckpt96k.json", \
    #     "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_seed120/all_predictions_test_ckpt96000.json", \
    #     "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_seed2022/all_predictions_test_ckpt96000.json", \
    #     "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_seed996/all_predictions_test_ckpt96000.json"]

    # all_dirs = ["/data3/private/clsi/OpenQA/AmbigQA/out/reader_hotpotqa/all_predictions_17ckpts/all_predictions_test_ckpt102000.json", \
    #     "/data3/private/clsi/OpenQA/AmbigQA/out/reader_hotpotqa/all_predictions_8ckpts/all_predictions_test_ckpt96000.json", \
    #     "/data3/private/clsi/OpenQA/AmbigQA/out/reader_hotpotqa/all_predictions_seed120/all_predictions_test_ckpt96000.json", \
    #     "/data3/private/clsi/OpenQA/AmbigQA/out/reader_hotpotqa/all_predictions_seed2022/all_predictions_test_ckpt96000.json", \
    #     "/data3/private/clsi/OpenQA/AmbigQA/out/reader_hotpotqa/all_predictions_seed996/all_predictions_test_ckpt96000.json"]

    all_dirs = ["/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_test_ckpt102000.json", \
        "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_test_ckpt96000.json", \
        "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_test_ckpt72000.json", \
        "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_test_ckpt48000.json", \
        "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_test_ckpt24000.json"]


    all_top_spans = []
    all_top_scores = []
    all_top_probs = []
    for data_dir in tqdm(all_dirs):
        calibrator = Calibrator(score_func=2, answer_agg=False, pipeline=False, prob="softmax", printing=True, ece_type=["interval", "density", "instance", "category"], task="qa")
        calibrator.load_dev(data_dir=data_dir)
        calibrator.top_span_dev()
        all_spans = calibrator.dev_spans
        top_idxes = calibrator.dev_top_idxes
        top_spans = []
        for ii,idx in enumerate(top_idxes):
            top_spans.append(normalize_answer(all_spans[ii][idx]))
        
        all_top_spans.append(top_spans)
        all_top_scores.append(calibrator.dev_top_scores)
        all_top_probs.append(calibrator.dev_top_probs)

    new_probs = []
    new_scores = []
    for i in range(len(all_top_spans[0])):
        candidates = [spans[i] for spans in all_top_spans]
        scores = [scores[i] for scores in all_top_scores]
        probs = [probs[i] for probs in all_top_probs]

        ## take majority vote
        ans = most_frequent(candidates)
        score = 0
        confs = 0
        counts = 0
        for j in range(len(candidates)):
            if candidates[j] == ans:
                counts += 1
                score = scores[j]
                confs += probs[j]
        
        # confs = confs / counts ## average confs
        # new_probs.append(confs)

        if counts >= 3:
            confs = 1
        else:
            confs = 0

        # confs = counts / 5

        new_scores.append(score)
        new_probs.append(confs)
        
    

    print ("After Consistency Calibration: ")
    calibrator.test_top_scores = new_scores
    calibrator.test_top_probs = new_probs
    calibrator.test_ece()


    

  


