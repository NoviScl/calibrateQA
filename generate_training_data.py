import json
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm 
import torch
np.random.seed(2022)
import json

'''
Generate training data for the calibration classifier.
'''

def load_train_data(data_file="nq_calibrate/nq_model_all_predictions_nq_dev.json"):
    with open(data_file, "r") as f:
        data = json.load(f)
    train_features = []
    dev_features = []
    train_len = int(len(data) * 0.9)
    for question in data[ : train_len]:
        q_correct = []
        q_wrong = []
        prob_wrong = []
        prob_correct = []
        all_features = []
        for p in range(len(question)):
            dp = question[p]
            prob = dp["passage_logit_raw"] + dp["start_logit_raw"] + dp["end_logit_raw"]
            score = int(dp["score"])
            features = []
            features.append(str(score))
            features.append(dp["answer_text"])
            features.append(dp["question"])
            features.append(dp["passage_text"])

            all_features.append(features)

            if score == 0:
                q_wrong.append(p)
                prob_wrong.append(prob)
            else:
                q_correct.append(p)
                prob_correct.append(prob)
        
        if len(q_correct) == 0:
            continue
        
        ## sort negative examples by prob
        prob_wrong = np.array(prob_wrong)
        idx = np.argsort(prob_wrong)[::-1]
        q_wrong = np.array(q_wrong)[idx]
        ## we sample from the harder neg subset
        q_wrong = q_wrong[ : 45]

        correct_selected = np.random.choice(q_correct, size=1, replace=False)
        for c in correct_selected:
            train_features.append(all_features[c])
        
        wrong_selected = np.random.choice(q_wrong, size=1, replace=False)
        for c in wrong_selected:
            train_features.append(all_features[c])

    
    for question in data[train_len : ]:
        q_correct = []
        q_wrong = []
        prob_wrong = []
        prob_correct = []
        all_features = []
        for p in range(len(question)):
            dp = question[p]
            prob = dp["passage_logit_raw"] + dp["start_logit_raw"] + dp["end_logit_raw"]
            score = int(dp["score"])
            features = []
            features.append(str(score))
            features.append(dp["answer_text"])
            features.append(dp["question"])
            features.append(dp["passage_text"])

            all_features.append(features)

            if score == 0:
                q_wrong.append(p)
                prob_wrong.append(prob)
            else:
                q_correct.append(p)
                prob_correct.append(prob)
        
        if len(q_correct) == 0:
            continue
        
        ## sort negative examples by prob
        prob_wrong = np.array(prob_wrong)
        idx = np.argsort(prob_wrong)[::-1]
        q_wrong = np.array(q_wrong)[idx]
        ## we sample from the harder neg subset
        q_wrong = q_wrong[ : 45]

        correct_selected = np.random.choice(q_correct, size=1, replace=False)
        for c in correct_selected:
            dev_features.append(all_features[c])
        
        wrong_selected = np.random.choice(q_wrong, size=1, replace=False)
        for c in wrong_selected:
            dev_features.append(all_features[c])

    print ("train len: ", len(train_features))
    print ("dev len: ", len(dev_features))
    return train_features, dev_features


def reconsider_data(data_file="nq_calibrate/nq_model_all_predictions_nq_dev.json"):
    with open(data_file, "r") as f:
        data = json.load(f)
    train_features = []
    dev_features = []
    train_len = int(len(data) * 0.9)
    for question in data[ : train_len]:
        q_correct = []
        q_wrong = []
        prob_wrong = []
        prob_correct = []
        all_features = []
        for p in range(len(question)):
            dp = question[p]
            prob = dp["passage_logit_raw"] + dp["start_logit_raw"] + dp["end_logit_raw"]
            score = int(dp["score"])
            features = []
            features.append(str(score))
            features.append(dp["answer_text"])
            features.append(dp["question"])
            features.append(dp["passage_text"])

            all_features.append(features)

            if score == 0:
                q_wrong.append(p)
                prob_wrong.append(prob)
            else:
                q_correct.append(p)
                prob_correct.append(prob)
        
        if len(q_correct) == 0:
            continue
        
        ## sort negative examples by prob
        prob_wrong = np.array(prob_wrong)
        idx = np.argsort(prob_wrong)[::-1]
        q_wrong = np.array(q_wrong)[idx]
        ## we sample from the harder neg subset
        q_wrong = q_wrong[ : 45]

        correct_selected = np.random.choice(q_correct, size=1, replace=False)
        for c in correct_selected:
            train_features.append(all_features[c])
        
        wrong_selected = np.random.choice(q_wrong, size=1, replace=False)
        for c in wrong_selected:
            train_features.append(all_features[c])

    
    for question in data[train_len : ]:
        q_correct = []
        q_wrong = []
        prob_wrong = []
        prob_correct = []
        all_features = []
        for p in range(len(question)):
            dp = question[p]
            prob = dp["passage_logit_raw"] + dp["start_logit_raw"] + dp["end_logit_raw"]
            score = int(dp["score"])
            features = []
            features.append(str(score))
            features.append(dp["answer_text"])
            features.append(dp["question"])
            features.append(dp["passage_text"])

            all_features.append(features)

            if score == 0:
                q_wrong.append(p)
                prob_wrong.append(prob)
            else:
                q_correct.append(p)
                prob_correct.append(prob)
        
        if len(q_correct) == 0:
            continue
        
        ## sort negative examples by prob
        prob_wrong = np.array(prob_wrong)
        idx = np.argsort(prob_wrong)[::-1]
        q_wrong = np.array(q_wrong)[idx]
        ## we sample from the harder neg subset
        q_wrong = q_wrong[ : 45]

        correct_selected = np.random.choice(q_correct, size=1, replace=False)
        for c in correct_selected:
            dev_features.append(all_features[c])
        
        wrong_selected = np.random.choice(q_wrong, size=1, replace=False)
        for c in wrong_selected:
            dev_features.append(all_features[c])

    print ("train len: ", len(train_features))
    print ("dev len: ", len(dev_features))
    return train_features, dev_features

# def load_test100_data(data_file="nq_calibrate/nq_model_all_predictions_nq_test.json"):
#     with open(data_file, "r") as f:
#         data = json.load(f)
#     test_features = []
#     for question in data:
#         all_prob = []
#         all_features = []
#         for p in range(len(question)):
#             dp = question[p]
#             prob = dp["passage_logit_raw"] + dp["start_logit_raw"] + dp["end_logit_raw"]
#             score = int(dp["score"])
#             features = []
#             features.append(str(score))
#             features.append(dp["answer_text"])
#             features.append(dp["question"])
#             features.append(dp["passage_text"])

#             all_features.append(features)
#             all_prob.append(prob)
        
#         for f in all_features:
#             test_features.append(f)
    
#     return test_features



def load_test10_data(data_file="nq_calibrate/nq_model_all_predictions_nq_test.json"):
    with open(data_file, "r") as f:
        data = json.load(f)
    test_features = []
    for question in data:
        ## first find best passage 
        passage_scores = [dp["passage_logit_raw"] for dp in question]
        best_psg_idx = np.argmax(passage_scores)
        best_psg = question[best_psg_idx]["passage_index"]
        # print (passage_scores)
        # print (best_psg_idx)

        all_prob = []
        all_features = []
        for p in range(len(question)):
            dp = question[p]
            if dp["passage_index"] != best_psg:
                continue 

            prob = dp["passage_logit_raw"] + dp["start_logit_raw"] + dp["end_logit_raw"]
            score = int(dp["score"])
            features = []
            features.append(str(score))
            features.append(dp["answer_text"])
            features.append(dp["question"])
            features.append(dp["passage_text"])

            all_features.append(features)
            all_prob.append(prob)
        
        for f in all_features:
            test_features.append(f)
    
    return test_features


# train, dev = load_train_data()
# test = load_test100_data()
test = load_test10_data()

# with open("train_data/train.tsv", "w") as f:
#     f.write("\t".join(["score", "prediction", "question", "passage"])+"\n")
#     for line in train:
#         f.write("\t".join(line)+"\n")

# with open("train_data/dev.tsv", "w") as f:
#     f.write("\t".join(["score", "prediction", "question", "passage"])+"\n")
#     for line in dev:
#         f.write("\t".join(line)+"\n")

# print (len(test))
# with open("train_data/test_top10/test.tsv", "w") as f:
#     f.write("\t".join(["score", "prediction", "question", "passage"])+"\n")
#     for line in test:
#         f.write("\t".join(line)+"\n")


# correct = 0
# with open("train_data/dev.tsv", "r") as f:
#     data = f.readlines()

# print (len(data))

# for line in data:
#     if line[0] == "1":
#         correct += 1
# print (correct, len(data))

## correct: 2555