import json
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 
import torch
from sklearn.linear_model import LogisticRegression
import pickle

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

def ECE(per_bucket_score, per_bucket_confidence, bucket_sizes):
    n_samples = sum(bucket_sizes)
    ece = 0.
    for i in range(len(bucket_sizes)):
        if bucket_sizes[i] > 0:
            delta = abs(per_bucket_score[i] - per_bucket_confidence[i])
            ece += (bucket_sizes[i] / n_samples) * delta
    return ece * 100


class Calibrator:
    def __init__(self, score_func, answer_agg, prob, calibrator, pipeline=False, printing=True):
        self.dev_set = None 
        self.test_set = None 
        self.score_func = score_func # {1, 2, 3}
        self.answer_agg = answer_agg # {True, False}
        self.prob = prob # {"softmax", "sigmoid"}
        self.calibrator = calibrator # {"NO", "TS", "BC"}
        self.temperature = 1.0
        self.printing = printing # whether to print results or not
        self.pipeline = pipeline

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

        self.test_top_scores = []
        self.test_top_probs = []
    
    def load_dev(self, data_dir="all_predictions_dev.json"):
        with open(data_dir, "r") as f:
            self.dev_set = json.load(f)
        if self.printing:
            print ("dev set loaded")
    
    def load_test(self, data_dir="all_predictions_test.json"):
        with open(data_dir, "r") as f:
            self.test_set = json.load(f)
        if self.printing:
            print ("test set loaded")
    
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
                
                all_logits[-1].append(span_score)
                all_scores[-1].append(int(dp["score"]))
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
                
                all_logits[-1].append(span_score)
                all_scores[-1].append(int(dp["score"]))
                all_spans[-1].append(dp["answer_text"])
            
        return all_logits, all_scores, all_spans

    def score_spans_dev(self):
        assert self.dev_set != None, "dev set is not loaded yet!"
        if self.pipeline:
            self.dev_logits, self.dev_scores, self.dev_spans = self.score_spans_pipeline(self.dev_set)
        else:
            self.dev_logits, self.dev_scores, self.dev_spans = self.score_spans(self.dev_set)
        if self.printing:
            print ("dev spans scored")
    
    def score_spans_test(self):
        assert self.test_set != None, "test set is not loaded yet!"
        if self.pipeline:
            self.test_logits, self.test_scores, self.test_spans = self.score_spans_pipeline(self.test_set)
        else:
            self.test_logits, self.test_scores, self.test_spans = self.score_spans(self.test_set)
        if self.printing:
            print ("test spans scored")
    
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
            top_probs.append(logits[top_idx])
            top_scores.append(data_scores[i][top_idx])
        
        return top_probs, top_scores 
    
    def top_span_dev(self):
        self.score_spans_dev()
        if self.answer_agg:
            self.answer_agg_dev()
        self.dev_top_probs, self.dev_top_scores = self.extract_top_span(self.dev_logits, self.dev_scores, temperature=self.temperature)
    
    def top_span_test(self):
        self.score_spans_test()
        if self.answer_agg:
            self.answer_agg_test()
        self.test_top_probs, self.test_top_scores = self.extract_top_span(self.test_logits, self.test_scores, temperature=self.temperature)

    def dev_ece(self):
        per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(self.dev_top_scores, self.dev_top_probs)
        ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
        if self.printing:
            print ("dev set")
            print ("EM: ", np.mean(self.dev_top_scores))
            print ("Acc: ", [round(num, 2) for num in per_bucket_score])
            print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
            print ("Sizes: ", bucket_sizes)
            print ("dev ECE: ", ece)
        return ece

    def test_ece(self):
        per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(self.test_top_scores, self.test_top_probs)
        ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
        if self.printing:
            print ("test set")
            print ("EM: ", np.mean(self.test_top_scores))
            print ("Acc: ", [round(num, 2) for num in per_bucket_score])
            print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
            print ("Sizes: ", bucket_sizes)
            print ("test ECE: ", ece)
        return ece

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
        # temp_values = map(lambda x: round(x / 10 + 0.1, 1), range(100))
        for temp in temp_values:
            self.temperature = temp
            self.top_span_dev()
            ece = self.dev_ece()
            if ece < best_ece:
                best_ece = ece 
                best_temp = temp
        
        print ("best temp: ", best_temp)
        self.temperature = best_temp

        self.printing = True 
        self.top_span_dev()
        self.dev_ece()
        self.top_span_test()
        self.test_ece()
    
        return best_temp



def eval(calibrator):
    calibrator.load_dev()
    calibrator.load_test()
    calibrator.top_span_dev()
    calibrator.top_span_test()
    calibrator.dev_ece()
    calibrator.test_ece()
    calibrator.test_ece_zero_correct()
    calibrator.test_ece_one_correct()
    calibrator.test_ece_more_correct()

def search(calibrator):
    # calibrator.load_dev()
    # calibrator.load_test()
    calibrator.search_temperature()

    calibrator.test_ece_zero_correct()
    calibrator.test_ece_one_correct()
    calibrator.test_ece_more_correct()

if __name__ == "__main__":
    for score_func in [2]:
        for prob in ["softmax"]:
            for answer_agg in [False, True]:
                print ("Hyper-params: score_func: {}, answer_agg: {}, prob: {}, calibrator: NO".format(str(score_func), str(answer_agg), prob))
                calibrator = Calibrator(score_func = score_func, answer_agg = answer_agg, prob = prob, calibrator = "NO", pipeline=True, printing=True)
                eval(calibrator)
                calibrator.printing = False
                print ("\n")
                print ("Hyper-params: score_func: {}, answer_agg: {}, prob: {}, calibrator: TS".format(str(score_func), str(answer_agg), prob))
                search(calibrator)
                print ("\n")

