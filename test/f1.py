from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import jsonlines,json
from tqdm import tqdm
from sklearn.metrics import f1_score
labeler = {'normal oral epithelium':0,  'oral squamous cell carcinoma':1} 


label_path = 'path/oscc_label.json'

with open(label_path, 'r') as f:
    label_dict = json.load(f)

read_path = 'result_path/oscc.jsonl'
with jsonlines.open(read_path) as reader:
    data =list(reader)

prediction = []
true_lable = []
for row in tqdm(data):
    qid = row['question_id']
    gts = label_dict[qid]
    true_lable.append(gts)

    pred = labeler[row['answer']]
    prediction.append(pred)

measure_result = classification_report(true_lable, prediction,digits=4)

micro_f1 = f1_score(true_lable, prediction,average="micro")
macro_f1 = f1_score(true_lable, prediction,average="macro")
print('measure_result = \n', measure_result)
print('micro-f1 = ', micro_f1)
print('macro-f1 = ', macro_f1)