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

def load_train_data(data_file="nq_calibrate/nq_model_all_predictions_nq_dev.json"):
    all_tops = []
    final_scores = []
    final_probs = []
    for i in tqdm([4,8,12,16,17]):
        data_dir = "/data3/private/clsi/OpenQA/AmbigQA/out/reader_nq_new_em/all_checkpoints_17ckpts/all_predictions_dev_ckpt" + str(i * 6000) + ".json"
        calibrator = Calibrator(score_func=2, answer_agg=False, pipeline=True, prob="softmax", printing=True, ece_type=["interval", "density", "instance", "category"], task="qa")
        calibrator.load_dev(data_dir=data_dir)
        calibrator.top_span_dev()
        all_spans = calibrator.dev_spans
        top_idxes = calibrator.dev_top_idxes
        top_spans = []
        for ii,idx in enumerate(top_idxes):
            top_spans.append(normalize_answer(all_spans[ii][idx]))
        all_tops.append(top_spans)

        if i == 17:
            final_scores = calibrator.dev_top_scores
            final_probs = calibrator.dev_top_probs
    
    print ("#total: ", len(final_scores))
    final_preds = all_tops[-1]
    correct = []
    wrong = []
    for i,pred in enumerate(final_preds):
        features = []
        for preds in all_tops[ : -1]:
            if pred == preds[i]:
                features.append(1.0)
            else:
                features.append(0.0)
        if final_scores[i] == 1:
            correct.append(features)
        else:
            wrong.append(features)

    print ("#correct: ", len(correct))
    print ("#wrong: ", len(wrong))
    # print (correct[:3])
    # print (wrong[:3])

    np.random.shuffle(correct)
    np.random.shuffle(wrong)

    wrong = wrong[ : len(correct)]
    X = np.concatenate((correct, wrong), axis=0)
    Y = np.array([1] * len(correct) + [0] * len(wrong))
    idx = np.array(list(range(len(Y))))
    np.random.shuffle(idx)
    X = X[idx]
    Y = Y[idx]

    print ("training data size: ", X.shape)

    return X, Y

def load_test_data(data_file="nq_calibrate/nq_model_all_predictions_nq_test.json"):
    all_tops = []
    final_scores = []
    final_probs = []
    for i in tqdm([4,8,12,16,17]):
        data_dir = "/data3/private/clsi/OpenQA/AmbigQA/out/reader_hotpotqa/all_predictions_17ckpts/all_predictions_test_ckpt" + str(i * 6000) + ".json"
        calibrator = Calibrator(score_func=2, answer_agg=False, pipeline=True, prob="softmax", printing=True, ece_type=["interval", "density", "instance", "category"], task="qa")
        calibrator.load_dev(data_dir=data_dir)
        calibrator.top_span_dev()
        all_spans = calibrator.dev_spans
        top_idxes = calibrator.dev_top_idxes
        top_spans = []
        for ii,idx in enumerate(top_idxes):
            top_spans.append(normalize_answer(all_spans[ii][idx]))
        all_tops.append(top_spans)

        if i == 17:
            final_scores = calibrator.dev_top_scores
            final_probs = calibrator.dev_top_probs
    
    print ("#total: ", len(final_scores))
    final_preds = all_tops[-1]
    test_X = []
    test_Y = []
    for i,pred in enumerate(final_preds):
        features = []
        for preds in all_tops[ : -1]:
            if pred == preds[i]:
                features.append(1.0)
            else:
                features.append(0.0)
        test_X.append(features)
        test_Y.append(final_scores[i])

    X = np.array(test_X)
    Y = np.array(test_Y)

    print ("test data size: ", X.shape)

    return X, Y

def fit_and_save(X, Y, tol):
    classifier = linear_model.SGDClassifier(loss='log', penalty='l2', max_iter=300, tol=tol, verbose=1,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=5)
    # classifier = linear_model.LogisticRegression(C=1.0, max_iter=300, verbose=2, tol=tol)
    # classifier = KNeighborsClassifier(n_neighbors=5)
    # classifier = SVC(kernel="linear", C=0.025, verbose=True, probability=True)
    # classifier = SVC(gamma=2, C=1, verbose=True, probability=True)
    # classifier = DecisionTreeClassifier()
    # classifier = RandomForestClassifier()
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
    
    print ("coefs: ", classifier.coef_)
    
    # return classifier.predict_proba(X)
    return classifier.predict(X)

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
    test_X, test_Y = load_test_data()
    # np.save("classifiers_k8/test_X.npy", test_X)
    # np.save("classifiers_k8/test_Y.npy", test_Y)

    # train_X = np.load("classifiers_k8/train_X.npy")
    # train_Y = np.load("classifiers_k8/train_Y.npy")
    # test_X = np.load("classifiers_k8/test_X.npy")
    # test_Y = np.load("classifiers_k8/test_Y.npy")


    # with open("train_data/test_top10/test.tsv", "r") as f:
    #     data = f.readlines()[1:]
    # # print (len(data))
    # data = [int(line[0]) for line in data]

    # with open("/home/sichenglei/LM-BFF/result/sst2_rbt_base_preds.pkl", "rb") as f:
    #     preds = pickle.load(f)

    # with open("/home/sichenglei/LM-BFF/data/all_data_final/SST-2/full/test.tsv", "r") as f:
    #     data = f.readlines()
    # labels = [int(line[0]) for line in data]
    # print (len(labels))

    # with open("/data3/private/clsi/reconsider_bert_base_augment_train10/epoch5/predictions_testM5.json", "r") as f:
    #     data = list(f)
    # scores = []
    # confs = []
    # indices = []
    # for p in data:
    #     line = json.loads(p)
    #     indices.append(int(line["prediction"]))
    #     scores.append(line["score"])
    #     # confs.append(line["confidence"])
    #     # confs.append(sigmoid(line["logit"]))
    #     confs.append(line["logit"])

    # best_temp = 1.0
    # # best_cat_ece = 1000
    # # best_temp = 0.02
    # # # temp_values = map(lambda x: round(x / 100 + 0.01, 2), range(1000))
    # # # for temp in tqdm(temp_values):
    # # #     new_confs = [sigmoid(c / temp) for c in confs]
    # # #     _, _, ece = category_ECE(scores, new_confs)
    # # #     if ece < best_cat_ece:
    # # #         best_cat_ece = ece 
    # # #         best_temp = temp 
    # # # print ("best temp: ", best_temp)
    # # # print ("best cat ece: ", best_cat_ece)

    # confs = [sigmoid(c / best_temp) for c in confs]
    # print ("#test questions: ", len(confs))
    # print ("EM: {:.3f}".format(np.mean(scores) * 100))
    # ece = ECE(scores, confs)
    # print ("Interval-based ECE: {:.3f}".format(ece))

    # print ("Instance ECE: {:.3f}".format(instance_ECE(scores, confs)))
    # pos_ece, neg_ece, ece = category_ECE(scores, confs)
    # print ("pos ECE: ", pos_ece)
    # print ("neg ECE: ", neg_ece)
    # print ("category-average ECE: ", ece)

    
    # with open("/data3/private/clsi/calibration/nq_calibrate/nq_model_all_predictions_nq_test.json", "r") as f:
    #     test = json.load(f)

    # for i in range(len(data)):
    #     idx = indices[i]
    #     conf = confs[i]
    #     score = scores[i]
    #     dp = test[i][idx]
    #     if score == 0 and conf >= 0.95:
    #         print (score, conf)
    #         print ("question: ", dp["question"]) 
    #         print ("prediction: ", dp["answer_text"])
    #         print ("passage: ", dp["passage_text"])
    #         print ("ground truth: ", dp["gold_answer"])
    #         print ()


    '''
    Use the corresponding test files used for the reranker to get 
    scores and confidences.
    '''
    
    # print (list(preds[0]))
    # print (np.array(preds).shape)

    # for tol in [0]:
    #     # for n_iter in [1]:
    #     fit_and_save(train_X, train_Y, tol)
    #             # # print ("tol: {}, n_iter: {}".format(str(tol), str(n_iter)))
    #             # fit_and_save(train_X, train_Y, tol, n_iter)
    
    
    probs_ = load_and_predict(test_X)  
    
    # probs_ = probs_[:,1]
    # print (probs_[:10])

    ece = intervalECE(test_Y, probs_)
    print ("Interval-based ECE: {:.3f}".format(ece))

    print ("Instance ECE: {:.3f}".format(instance_ECE(test_Y, probs_)))

    per_bucket_score, per_bucket_confidence, bucket_sizes = get_bucket_scores_and_confidence_by_density(test_Y, probs_)
    ece = ECE(per_bucket_score, per_bucket_confidence, bucket_sizes)
    print ("density ECE: {:.3f}".format(ece))

    pos_ece, neg_ece, ece = category_ECE(test_Y, probs_)
    print ("pos ECE: ", pos_ece)
    print ("neg ECE: ", neg_ece)
    print ("category-average ECE: ", ece)
    
    # print ("probs")
    # print (sum(probs_))

    #     # probs = []
    #     # for p in probs_:
    #     #     if p < 0:
    #     #         probs.append(0)
    #     #     elif p > 1:
    #     #         probs.append(1)
    #     #     else:
    #     #         probs.append(p)
    #     # probs = probs[:, 1] # prob of prediction being correct
        
    #     probs = []
    #     scores = []
    #     confs = []
    #     # for p in preds:
    #     #     # p = list(p)
    #     #     # print (list(p))
    #     #     # print (softmax(list(p)))
    #     #     probs.append(softmax(list(p))[1])
    #     #     # probs.append(sigmoid(p[1]))
    #     for i in range(len(preds)):
    #         p = list(preds[i])
    #         confs.append(np.max(softmax(p)))
    #         scores.append(int(np.argmax(p) == labels[i]))
    #         # print (softmax(p))
    #         # print (int(np.argmax(p) == labels[i]))
    #         # print ()
        
    #     # # print (probs)
    #     # ## get top predictions per questions
    #     # topk = 10
    #     # confs = []
    #     # scores = []
    #     # for i in range(int(len(data) / topk)):
    #     #     candidates = probs[i*topk : (i+1)*topk]
    #     #     idx = np.argmax(candidates)
    #     #     # idx = np.random.choice(range(100), size=1)[0]
    #     #     confs.append(candidates[idx])
    #     #     scores.append(data[i*topk + idx])

        # print ("#test questions: ", len(confs))
        # print ("EM: {:.3f}".format(np.mean(scores) * 100))
        # ece = ECE(scores, confs)
        # print ("Interval-based ECE: {:.3f}".format(ece))

        # print ("Instance ECE: {:.3f}".format(instance_ECE(scores, confs)))
        # pos_ece, neg_ece, ece = category_ECE(scores, confs)
        # print ("pos ECE: ", pos_ece)
        # print ("neg ECE: ", neg_ece)
        # print ("category-average ECE: ", ece)

    # probs = load_and_predict(test_X)   
    # print (probs[:200])

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
