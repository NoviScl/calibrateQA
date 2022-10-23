# Re-Examining Calibration: The Case of Question Answering

This codebase supports the calibration experiments in our EMNLP 2022 Findings paper "Re-Examining Calibration: The Case of Question Answering", authored by Chenglei Si, Chen Zhao, Sewon Min, and Jordan Boyd-Graber. 


## Data

We provide a sample prediction file on NQ test set which can be downloaded from [this link](https://drive.google.com/file/d/18VQ2RSYH9kF92CmjTakCxP9e1551EMTw/view?usp=sharing). The codebase is configured to directly work with prediction files formatted in the same way, where for each question, we record the list of top 100 predictions from DPR-BERT along with the passage retriever score, span start score, and span end score.

Specifically, each prediction is a dictionary formatted as:

```
dict_keys(['predicted_start_index', 'predicted_end_index', 'answer_text', 'answer_ids', 
'passage_index', 'passage_text', 'passage_ids', 'passage_logit_raw', 'passage_logit_log_softmax', 
'start_logit_raw', 'start_logit_log_softmax', 'end_logit_raw', 'end_logit_log_softmax', 
'log_softmax', 'gold_answer', 'score', 'question'])
```

And the whole prediction file is a list of length ``N``, where ``N`` is the number of questions. Each element in this list is a list of length 100 where each element is an answer prediction dictionary defined above.

To obtain DPR-BERT predictions on other QA datasets, you can use the original DPR [codebase](https://github.com/facebookresearch/DPR) or Sewon's AmbigQA [repo](https://github.com/shmsw25/AmbigQA) to do the training and inference. If you are using other QA codebases, you just need to convert your prediction files to have the same format as ours so that this codebase can directly be used on top of them. 


## Calibration Code


We provided implementation of all calibration metrics used in the paper in ``calibrator.py``. We provide an example usage of the ``Calibrator`` class below:

```python
from calibrator import Calibrator

test_dir = "all_predictions_test_ckpt96000.json"
calibrator = Calibrator(score_func=2, answer_agg=False, pipeline=False, prob="softmax", printing=True, ece_type=["interval", "density", "instance", "category"], task="qa", buckets=10)
calibrator.load_test(data_dir=test_dir)
calibrator.top_span_test()
calibrator.test_ece()
```

* ``test_dir`` should be the path of your QA prediction file
* ``pipeline`` refers to whether we are using joint or pipeline calibration
* ``ece_type`` refers to what calibration metrics we want to compute 
* ``buckets`` refer to how many buckets (default is 10 in our paper) is used when computing ECE

Given the QA prediction file, the above code will automatically score each prediction, compute the confidence score of the top prediction for each question, and compute the calibration errors with the specified metrics.

You can also directly run ``calibrator.py`` to run the above example. We also support temperature scaling, where you can simply add ``calibrator.search_temperature()`` to the above code snippet. 


## Citation 

If you find our work useful, please consider citing it:
```bibtex
@inproceedings{Si:Zhao:Min:Boyd-Graber-2022,
	Title = {Re-Examining Calibration: The Case of Question Answering},
	Author = {Chenglei Si and Chen Zhao and Sewon Min and Jordan Boyd-Graber},
	Booktitle = {Findings of Empirical Methods in Natural Language Processing},
	Year = {2022},
	Location = {Abu Dhabi},
}
```


