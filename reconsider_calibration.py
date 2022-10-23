import json
import numpy as np
from tqdm import tqdm 
from calibrator import * 
from consistency import *
from classifier import *

# preds = "/data3/private/clsi/reconsider_bert_large/predictions_new_testM3.json"
preds = "/data3/private/clsi/reconsider_bert_large/hotpotpredictions_new_testM3.json"

scores = []
confs = []

with open(preds) as f:
    for line in f:
        line = json.loads(line)
        scores.append(line["score"])
        confs.append(line["confidence"]) # softmax conf
        # confs.append(sigmoid(line["logit"])) # sigmoid conf


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

