"""Small, reproducible educational experiment. See README for limitations."""
import json
from pathlib import Path
import numpy as np

def save(result, output):
    Path(output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
DOCS={
"split":"Split data into training validation and test sets. Keep test labels out of model selection.",
"scaling":"Standard scaling subtracts the training mean and divides by the training standard deviation.",
"precision":"Precision is true positives divided by all predicted positives. It measures false alarm burden.",
"recall":"Recall is true positives divided by all actual positives. It measures missed positive cases.",
"drift":"Data drift is a change in input feature distributions. Drift alone does not prove model accuracy fell.",
"leakage":"Target leakage happens when information unavailable at prediction time enters training features.",
"retrieval":"Retrieval ranks relevant documents for a question. Retrieved evidence is not a generated answer.",
"calibration":"Probability calibration compares predicted confidence to observed outcome frequency.",
"forecast":"Time series forecasting evaluates future periods using earlier observations without shuffling time.",
"privacy":"Remove personal identifiers and check consent before sharing private data.",
"baseline":"A baseline is a simple reference such as the majority class or previous seasonal observation.",
"reproducibility":"Record random seeds dependency versions data provenance and commands for reproducibility."}
QUERIES=[("keep test labels away from model selection","split"),("subtract mean divide standard deviation","scaling"),("false alarm precision predicted positives","precision"),("missing actual positive cases recall","recall"),("input distributions have changed","drift"),("information unavailable at prediction time","leakage"),("rank documents for a question","retrieval"),("confidence against outcome frequency","calibration"),("evaluate future time periods","forecast"),("sharing personal identifiers and consent","privacy"),("simple majority class reference","baseline"),("record seeds and dependency versions","reproducibility")]
UNKNOWN=["how to bake a chocolate cake","what is the capital of Peru"]
def rank(query,vectorizer,matrix):
    scores=cosine_similarity(vectorizer.transform([query]),matrix)[0]
    order=np.argsort(-scores,kind="stable")
    return [(list(DOCS)[i],float(scores[i])) for i in order]
def run(output="results.json"):
    vec=TfidfVectorizer(stop_words="english"); matrix=vec.fit_transform(DOCS.values())
    rows=[]
    for q,gold in QUERIES:
        hits=rank(q,vec,matrix); position=next(i+1 for i,(k,s) in enumerate(hits) if k==gold)
        rows.append({"query":q,"expected":gold,"top_document":hits[0][0],"rank":position,"score":hits[0][1]})
    threshold=.1
    unknown=[{"query":q,"best_score":rank(q,vec,matrix)[0][1],"abstained":rank(q,vec,matrix)[0][1]<threshold} for q in UNKNOWN]
    return save({"documents":len(DOCS),"queries":len(rows),"recall_at_1":float(np.mean([r['rank']<=1 for r in rows])),"recall_at_3":float(np.mean([r['rank']<=3 for r in rows])),"mrr":float(np.mean([1/r['rank'] for r in rows])),"abstention_threshold":threshold,"unknown_results":unknown,"rankings":rows},output)

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
