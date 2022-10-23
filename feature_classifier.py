import json
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 
import torch
from sklearn.linear_model import LogisticRegression
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import pickle
from calibrator import *


import time
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import shuffle

np.random.seed(2022)

def load_train_data(data_file="/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_dev_ckpt102000.json"):
    with open(data_file, "r") as f:
        data = json.load(f)

    correct = []
    wrong = []

    print ("#total questions: ", len(data))
    for qn in data:
        all_answers = [normalize_answer(dp["answer_text"]) for dp in qn]
        all_starts = softmax([dp["start_logit_raw"] for dp in qn])
        all_ends = softmax([dp["end_logit_raw"] for dp in qn])
        all_spans = softmax([dp["start_logit_raw"] + dp["end_logit_raw"] for dp in qn])

        for p, dp in enumerate(qn):
            features = []
            ## P, Q, A lengths
            features.append(float(len(dp["answer_text"].split())))
            features.append(float(len(dp["passage_text"].split())))
            features.append(float(len(dp["question"].split())))

            ## raw logits and softmax probs
            ## passage logit is softmaxed over top 10 retrievals
            features.append(float(dp["passage_logit_raw"]))
            # features.append(float(dp["passage_logit_log_softmax"]))
            features.append(np.exp(float(dp["passage_logit_log_softmax"])))

            ## span logits
            features.append(float(dp["start_logit_raw"]))
            # features.append(float(dp["start_logit_log_softmax"]))
            features.append(np.exp(float(dp["start_logit_log_softmax"])))
            features.append(float(dp["end_logit_raw"]))
            # features.append(float(dp["end_logit_log_softmax"]))
            features.append(np.exp(float(dp["end_logit_log_softmax"])))

            ## softmaxed scores 
            features.append(all_starts[p])
            features.append(all_ends[p])
            features.append(all_spans[p])

            ## scores of other spans
            features.extend([all_starts[i] for i in range(len(all_starts)) if i!=p])
            features.extend([all_ends[i] for i in range(len(all_ends)) if i!=p])
            features.extend([all_spans[i] for i in range(len(all_spans)) if i!=p])

            ## counts
            features.append(normalize_answer(dp["passage_text"]).count(normalize_answer(dp["answer_text"])))
            features.append(normalize_answer(dp["question"]).count(normalize_answer(dp["answer_text"])))
            features.append(all_answers.count(normalize_answer(dp["answer_text"])))

            if dp["score"] == 0:
                wrong.append(features)
            elif dp["score"] == 1:
                correct.append(features)

    np.random.shuffle(correct)
    np.random.shuffle(wrong)

    wrong = wrong[ : len(correct)]
    X = np.concatenate((correct, wrong), axis=0)
    Y = np.array([1] * len(correct) + [0] * len(wrong))
    idx = np.array(list(range(len(Y))))
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    print ("#correct: ", len(correct))
    print ("#wrong: ", len(wrong))

    print ("training data size: ", X.shape)

    return X, Y

def load_test_data(data_file="/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_test_ckpt102000.json"):
    with open(data_file, "r") as f:
        data = json.load(f)

    test_X = []
    test_Y = []
    print ("#total questions: ", len(data))
    ## for HotpotQA, a few questions got <100 predictions, ignore them during eval
    for qn in data:
        if len(qn) != 100:
            continue

        all_answers = [normalize_answer(dp["answer_text"]) for dp in qn]
        all_starts = softmax([dp["start_logit_raw"] for dp in qn])
        all_ends = softmax([dp["end_logit_raw"] for dp in qn])
        all_spans = softmax([dp["start_logit_raw"] + dp["end_logit_raw"] for dp in qn])

        for p, dp in enumerate(qn):
            features = []
            ## P, Q, A lengths
            features.append(float(len(dp["answer_text"].split())))
            features.append(float(len(dp["passage_text"].split())))
            features.append(float(len(dp["question"].split())))

            ## raw logits and softmax probs
            ## passage logit is softmaxed over top 10 retrievals
            features.append(float(dp["passage_logit_raw"]))
            # features.append(float(dp["passage_logit_log_softmax"]))
            features.append(np.exp(float(dp["passage_logit_log_softmax"])))

            ## span logits
            features.append(float(dp["start_logit_raw"]))
            # features.append(float(dp["start_logit_log_softmax"]))
            features.append(np.exp(float(dp["start_logit_log_softmax"])))
            features.append(float(dp["end_logit_raw"]))
            # features.append(float(dp["end_logit_log_softmax"]))
            features.append(np.exp(float(dp["end_logit_log_softmax"])))

            ## softmaxed scores 
            features.append(all_starts[p])
            features.append(all_ends[p])
            features.append(all_spans[p])

            ## scores of other spans
            features.extend([all_starts[i] for i in range(len(all_starts)) if i!=p])
            features.extend([all_ends[i] for i in range(len(all_ends)) if i!=p])
            features.extend([all_spans[i] for i in range(len(all_spans)) if i!=p])

            ## counts
            features.append(normalize_answer(dp["passage_text"]).count(normalize_answer(dp["answer_text"])))
            features.append(normalize_answer(dp["question"]).count(normalize_answer(dp["answer_text"])))
            features.append(all_answers.count(normalize_answer(dp["answer_text"])))

            test_X.append(features)
            test_Y.append(int(dp["score"]))
    
    X = np.array(test_X)
    Y = np.array(test_Y)

    print ("test data size: ", X.shape)

    return X, Y

def fit_and_save(X, Y, tol):
    # classifier = linear_model.SGDClassifier(loss='log', penalty='l2', max_iter=300, tol=tol, verbose=1,
    #     early_stopping=True, validation_fraction=0.1, n_iter_no_change=5)
    
    # classifier = linear_model.LogisticRegression(C=1.0, max_iter=300, verbose=2, tol=tol)
    # classifier = KNeighborsClassifier(n_neighbors=5)
    # classifier = SVC(kernel="linear", C=0.025, verbose=True, probability=True)
    # classifier = SVC(gamma=2, C=1, verbose=True, probability=True)
    # classifier = DecisionTreeClassifier()
    classifier = RandomForestClassifier()
    # classifier =  AdaBoostClassifier()
    # classifier = MLPClassifier(verbose=True, early_stopping=True, validation_fraction=0.1, n_iter_no_change=2, tol=1e-4)
    # classifier = LinearRegression()

    classifier.fit(X, Y)
    train_score = classifier.score(X, Y)

    print ("Acc on training set: {:.3f}".format(train_score))

    with open("classifiers_k8/LR_calibrator.pkl", "wb") as f:
        pickle.dump(classifier, f)

def load_and_predict(X):
    with open("classifiers_k8/LR_calibrator.pkl", "rb") as f:
        classifier = pickle.load(f)
    
    # print ("coefs: ", classifier.coef_)
    
    return classifier.predict_proba(X)
    # return classifier.predict(X)

def intervalECE(all_scores, all_probs, buckets=10):
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

    print ("Acc: ", [round(num, 2) for num in per_bucket_score])
    print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
    print ("Sizes: ", bucket_sizes)

    n_samples = sum(bucket_sizes)
    ece = 0.
    for i in range(len(bucket_sizes)):
        if bucket_sizes[i] > 0:
            delta = abs(per_bucket_score[i] - per_bucket_confidence[i])
            ece += (bucket_sizes[i] / n_samples) * delta
    return ece * 100



def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":    
    # train_X, train_Y = load_train_data()
    # np.save("classifiers_k8/train_X.npy", train_X)
    # np.save("classifiers_k8/train_Y.npy", train_Y)
    # test_X, test_Y = load_test_data()
    # np.save("classifiers_k8/test_X.npy", test_X)
    # np.save("classifiers_k8/test_Y.npy", test_Y)

    # train_X = np.load("classifiers_k8/train_X.npy")
    # train_Y = np.load("classifiers_k8/train_Y.npy")
    # test_X = np.load("classifiers_k8/test_X.npy")
    # test_Y = np.load("classifiers_k8/test_Y.npy")

    ## Load OOD test set
    test_X, test_Y = load_test_data(data_file="/data3/private/clsi/OpenQA/AmbigQA/out/reader_hotpotqa/all_predictions_17ckpts/all_predictions_test_ckpt102000.json")


    ## can do some hyper-param tuning here
    # for tol in [0]:
    #     # for n_iter in [1]:
    #     fit_and_save(train_X, train_Y, tol)
    #             # # print ("tol: {}, n_iter: {}".format(str(tol), str(n_iter)))
    #             # fit_and_save(train_X, train_Y, tol, n_iter)
    
    
    probs_ = load_and_predict(test_X)  
    probs = probs_[:,1]

    topk = 100 # number of predictions per question
    confs = []
    scores = []
    for i in range(int(len(probs) / topk)):
        candidates = probs[i*topk : (i+1)*topk]
        idx = np.argmax(candidates)
        confs.append(candidates[idx])
        scores.append(test_Y[i*topk + idx])

    print ("EM: ", np.mean(scores))
    
    ece = intervalECE(scores, confs)
    print ("Interval-based ECE: {:.3f}".format(ece))

    print ("Instance ECE: {:.3f}".format(instance_ECE(scores, confs)))

    per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence_by_density(scores, confs)
    ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
    print ("density ECE: {:.3f}".format(ece))

    pos_ece, neg_ece, ece = category_ECE(scores, confs)
    print ("pos ECE: ", pos_ece)
    print ("neg ECE: ", neg_ece)
    print ("category-average ECE: ", ece)
    
   
    
    ## check LR weights
    # with open("LR_calibrator.pkl", "rb") as f:
    #     classifier = pickle.load(f)
    
    # probs = load_and_predict(test_X)
    # print (probs[:100])

    # # print (classifier.n_iter_)
    # # print (classifier.coef_)
    # coefs = classifier.coef_[0]
    # abs_coefs = [abs(c) for c in coefs]
    # idx = np.argsort(abs_coefs)[::-1]
    # for id in idx[:20]:
    #     print (id, coefs[id])
