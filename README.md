## calibrateQA

`100_test_predictions.json` contains the predictions for the first 100 questions in the NQ test set. The file is a list with 100 element lists, each corresponding to a question. Each questions contain 100 prediction dictionaries, with the following format: 

`
dict_keys(['predicted_start_index', 'predicted_end_index', 'answer_text', 'answer_ids', 
'passage_index', 'passage_text', 'passage_ids', 'passage_logit_raw', 'passage_logit_log_softmax', 
'start_logit_raw', 'start_logit_log_softmax', 'end_logit_raw', 'end_logit_log_softmax', 
'log_softmax', 'gold_answer', 'score', 'question'])
`

`
format: a list of len N, where N is the number of questions. Each element is a list of length 100, 
where each element is an answer prediction dict.
`
