import pandas as pd
import json

# create tokenizer
def create_tokenizer(path_documents="events.json"):
    with open(path_documents) as f:
        data = json.load(f)
    tokenizer_codes = dict()
    list_codes = list()
    for timesteps in data.values():
        for tuple in timesteps.values():
            for code, _ in tuple:
                code = int(code)
                list_codes.append(code)

    counts = pd.DataFrame(list_codes)[0].value_counts()
    kept_codes = counts[counts > 100].index
    list_codes.sort()
    tokenizer_codes[0] = 0 # padding token
    tokenizer_codes[1] = 1 # cls token
    for code in set(list_codes):
        if code in kept_codes:
            tokenizer_codes[code] = len(tokenizer_codes)

    r = json.dumps(tokenizer_codes, sort_keys=True)
    with open('tokenizer.json', "w") as f:
        f.write(r)

from sklearn.metrics import average_precision_score as AUPRC
from sklearn.metrics import roc_auc_score as AUROC
from sklearn.metrics import accuracy_score as accuracy
from sklearn.metrics import log_loss
def compute_metrics(y_true, y_prob):
    acc = 100*accuracy(y_true=y_true, y_pred=list(map(round, y_prob)))
    auprc = AUPRC(y_true=y_true, y_score=y_prob)
    auroc = AUROC(y_true=y_true, y_score=y_prob)
    bce = log_loss(y_true=y_true, y_pred=y_prob)

    return [acc, auprc, auroc, bce]