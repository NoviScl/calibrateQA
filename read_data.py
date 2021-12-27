import json
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 
import torch
from utils import normalize_answer, softmax


'''
dict_keys(['predicted_start_index', 'predicted_end_index', 'answer_text', 'answer_ids', 
'passage_index', 'passage_text', 'passage_ids', 'passage_logit_raw', 'passage_logit_log_softmax', 
'start_logit_raw', 'start_logit_log_softmax', 'end_logit_raw', 'end_logit_log_softmax', 
'log_softmax', 'gold_answer', 'score', 'question'])

format: a list of len N, where N is the number of questions. Each element is a list of length 100, 
where each element is an answer prediction dict.
'''

def extract_correct(psg_lst):
    '''
    Given a list of predictions, return the number of correct ones.
    '''
    score = 0
    for dp in psg_lst:
        score += int(dp['score'])
    return score

def get_stats(data):
    correct_dict = {}
    for pl in data:
        score = extract_correct(pl)
        if score not in correct_dict:
            correct_dict[score] = 1
        else:
            correct_dict[score] += 1

    for k,v in correct_dict.items():
        print ("scores: {}, counts: {}".format(k, v))
    

def answer_aggregation(psg_lst, add_sel_logit=True):
    '''
    Given a list of 100 answer predictions, aggregate scores for equivalent spans.
    Output: list of raw logits and sscores (0/1) for the distinct spans.
    By default we take psg_logit + start_logit + end_logit as the overall score. 
    '''
    span_logits = {}
    span_scores = {}
    for dp in psg_lst:
        span = normalize_answer(dp["answer_text"])
        score = int(dp["score"])
        logit = dp['start_logit_log_softmax'] + dp['end_logit_log_softmax']
        if add_sel_logit:
            logit += dp['passage_logit_log_softmax']
        logit = np.exp(logit)
        if span not in span_logits:
            span_logits[span] = logit
            span_scores[span] = score 
        else:
            span_logits[span] += logit
            span_scores[span] = max(score, span_scores[span])
    all_logits = []
    all_scores = []
    all_spans = []
    for k,v in span_logits.items():
        all_logits.append(v)
        all_scores.append(span_scores[k])
        all_spans.append(k)
    return all_logits, all_scores, all_spans


def passage_aggregation(psg_lst):
    '''
    Given a list of 100 answer predictions, extract the top 10 passages. 
    Output: list of raw logits and sscores (0/1) for the distinct spans.
    By default we take psg_logit + start_logit + end_logit as the overall score. 
    '''
    passage_scores = [0] * 10
    passage_logits = [0] * 10
    
    for dp in psg_lst:
        pid = int(dp["passage_index"])
        passage_scores[pid] = max(passage_scores[pid], int(dp["score"]))
        passage_logits[pid] = dp["passage_logit_raw"]
    
    return passage_logits, passage_scores




def get_top_span_by_span_logits(psg_lst, add_sel_logit=True, temp=1.0):
    '''
    Softmax over (start_logit + end_logit) as the span prob. 
    Return the best span's EM (0/1) and the confidence score.
    '''
    all_logits = []
    for dp in psg_lst:
        logit = dp['start_logit_raw'] + dp['end_logit_raw']
        if add_sel_logit:
            logit += dp['passage_logit_raw']
        all_logits.append(logit / temp)
    all_logits = softmax(all_logits)
    best_index = np.argmax(all_logits)
    best_logit = np.max(all_logits)
    score = psg_lst[best_index]['score']
    return score, best_logit


def get_top_span_in_lst(logits_lst, scores_lst, temp=1.0):
    '''
    Given logits list and scores list, 
    First perform softmax over the logits list,
    Then return the best span's (softmax) logit and socre
    '''
    logits_lst = [logit / temp for logit in logits_lst]
    logits = softmax(logits_lst)
    best_index = np.argmax(logits)
    best_logit = np.max(logits)
    score = scores_lst[best_index]
    return score, best_logit, best_index



def get_score_distribution(psg_lst, add_sel_logit=True):
    '''
    get all scores and confidences (raw logits)
    '''
    all_logits = []
    all_scores = []
    for dp in psg_lst:
        logit = dp['start_logit_raw'] + dp['end_logit_raw']
        if add_sel_logit:
            logit += dp['passage_logit_raw']
        all_logits.append(logit)
        all_scores.append(dp['score'])
    return all_scores, all_logits


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


def cross_entropy(output, target):
    '''
    output: softmax distribution
    target: one hot labels
    '''
    return F.kl_div(output, target, reduction='sum').item()



if __name__ == "__main__":
    ## find best temp on dev
    with open("all_predictions_dev.json", "r") as f:
        data = json.load(f)
    
    # all_scores = []
    # all_logits = []
    # for pl in data:
    #     # score = extract_correct(pl)
    #     # if score != 1:
    #     #     continue
    #     # agg_logits, agg_scores = answer_aggregation(pl)
    #     agg_logits, agg_scores = passage_aggregation(pl)
    #     # scores, logits = get_score_distribution(pl, add_sel_logit=True)
    #     all_scores.append(agg_scores)
    #     all_logits.append(agg_logits)
    
    # ## temperature tuning
    # best_nll = float('inf')
    # best_temp = -1

    # temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))
    # for temp in tqdm(temp_values):
    #     nll = 0
    #     for i in range(len(all_scores)):
    #         nll += cross_entropy(F.log_softmax(torch.tensor(all_logits[i]).float()  / temp, dim=0), torch.tensor(all_scores[i]).float())
    #     if nll < best_nll:
    #         best_nll = nll
    #         best_temp = temp
            
    #         # print ("best temp: ", best_temp)
    #         # print ("best nll: ", best_nll / len(all_scores))

    
    # print ("best temp: ", best_temp)
    # print ("best nll: ", best_nll / len(all_scores)) 



    # evaluate with best temp

    # best_temp = 2.14
    # add_sel_logit = True
    # with open("all_predictions_test.json", "r") as f:
    #     data = json.load(f)

    # # with open("nq-annotations.jsonl", "r") as f:
    # #     ref = [json.loads(l) for l in f]
    
    # all_scores = []
    # all_probs = []
    # zero = 0
    # one = 0
    # more = 0
    # lens = 0
    # for i in range(len(data)):
    #     pl = data[i]
    #     # if "no_answer_overlap" not in ref[i]["labels"]:
    #     #     continue

    # #     agg_logits, agg_scores = answer_aggregation(pl)
    # #     lens += len(agg_logits)
    # #     s = sum(agg_scores)
    # #     if s > 1:
    # #         print (s)
    # #     elif s == 0:
    # #         zero += 1
    # #     elif s == 1:
    # #         one += 1
    # # print ("zero: {}, one: {}".format(zero, one))
    # # print ("avg len: ", lens / len(data))
    #     # score = extract_correct(pl)
    #     # if score > 0:
    #     #     continue

    #     # agg_logits, agg_scores, agg_spans = answer_aggregation(pl)
    #     # em, confidence, best_index = get_top_span_in_lst(agg_logits, agg_scores, temp=best_temp)

    #     agg_logits, agg_scores = passage_aggregation(pl)
    #     em, confidence, best_index = get_top_span_in_lst(agg_logits, agg_scores, temp=best_temp)

    #     scores = sum(agg_scores)
    #     # if scores == 0:
    #     #     zero += 1
    #     # elif scores == 1:
    #     #     one += 1
    #     # else:
    #     #     more += 1
    #     if scores <= 1:
    #         continue

    #     # if confidence > 0.95 and em == 0:
    #     #     print ("question: ", pl[0]["question"])
    #     #     print ("gold answer: ", pl[0]["gold_answer"])
    #     #     print ("prediction: ", agg_spans[best_index])
    #     #     print ("confidence: ", confidence)
    #     #     print ()

    #     # if confidence < 0.05 and em == 1:
    #     #     print ("question: ", pl[0]["question"])
    #     #     print ("gold answer: ", pl[0]["gold_answer"])
    #     #     print ("prediction: ", agg_spans[best_index])
    #     #     print ("confidence: ", confidence)
    #     #     print ()
            

    #     all_scores.append(em)
    #     all_probs.append(confidence)
    
    # # print ("number of questions: ", len(data))
    # # print ("zero correct passage: ", zero)
    # # print ("one correct passage: ", one)
    # # print ("more than one correct passage: ", more)
    
    # print ("number of questions: ", len(all_scores))
    # print ("EM: ", round(np.mean(all_scores) * 100, 2))
    
    # per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(all_scores, all_probs, buckets=10)
    # print ("Acc: ", [round(p, 2) for p in per_bucket_score])
    # print ("Conf: ", [round(p, 2) for p in per_bucket_confidence])
    # print ("Sizes: ", bucket_sizes)
    # print ("ECE: ", round(ECE(per_bucket_score, per_bucket_confidence, bucket_sizes), 2))
    # print ()


    # all_scores = []
    # all_probs = []
    # zero = 0
    # one = 0
    # lens = 0
    # for pl in data:
    # #     agg_logits, agg_scores = answer_aggregation(pl)
    # #     lens += len(agg_logits)
    # #     s = sum(agg_scores)
    # #     if s > 1:
    # #         print (s)
    # #     elif s == 0:
    # #         zero += 1
    # #     elif s == 1:
    # #         one += 1
    # # print ("zero: {}, one: {}".format(zero, one))
    # # print ("avg len: ", lens / len(data))
    #     score = extract_correct(pl)
    #     if score != 1:
    #         continue
    #     agg_logits, agg_scores = answer_aggregation(pl)
    #     em, confidence = get_top_span_in_lst(agg_logits, agg_scores, temp=best_temp)

    #     all_scores.append(em)
    #     all_probs.append(confidence)
    
    # print ("number of questions: ", len(all_scores))
    # print ("EM: ", np.mean(all_scores) * 100)
    
    # per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence(all_scores, all_probs, buckets=10)
    # print (per_bucket_score)
    # print (per_bucket_confidence)
    # print (bucket_sizes)
    # print ("ECE: ", ECE(per_bucket_score, per_bucket_confidence, bucket_sizes))
    # print ()





