import nltk
import pickle
import sklearn
from pathlib import Path

nltk.download('punkt')

def load_model(name):
    if name == "count":
        return pickle.load(open(Path("pretrained_models/count_vect_model.pkl"), "rb"))
    elif name == "one_hot":
        return pickle.load(open(Path("pretrained_models/onehot_vect_model.pkl"), "rb"))
    else:
        return pickle.load(open(Path("pretrained_models/tf_idf_vect_model.pkl"), "rb"))
