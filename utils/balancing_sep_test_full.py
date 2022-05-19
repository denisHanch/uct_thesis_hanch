import pandas as pd
import numpy as np
import imblearn

import time
import logging
import datetime
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


logging.basicConfig(filename='balancing_full.log', level=logging.INFO)
logging.info(f"{datetime.datetime.now()}: Start")

random_state = 43

for bit in [1024, 2048]:
    feats = [str(x) for x in range(bit)]
    for rad in [2, 4]:
        base_name = "morgan_fullData"
        name_df = base_name + f"_{rad}_{bit}"
        df = pd.read_csv("fullData/" + name_df + ".csv")

        df = df[df.steps <= 10]

        X, y, smiles = np.array(df[[*feats]]), df.steps, df.smiles

        # Stratified train/test split
        X, X_test, y, y_test, smiles, smiles_test = train_test_split(X, y, smiles, random_state=random_state, stratify=y, test_size=0.1)

        y = LabelEncoder().fit_transform(y)

        counter = Counter(y)
        logging.info(f"{datetime.datetime.now()}: Class breakdown for {name_df} BEFORE SMOTE")
        for k, v in counter.items():
            per = v / len(y) * 100
            logging.info(f"{datetime.datetime.now()}: Class={k}, n={v} ({per:.3f}%)")
        logging.info(f"")
            
        oversample = SMOTE(k_neighbors=np.unique(y, return_counts=True)[1].min()-1, random_state=random_state)
        X, y = oversample.fit_resample(X, y)

        counter = Counter(y)
        logging.info(f"{datetime.datetime.now()}: Class breakdown for {name_df} AFTER SMOTE")
        for k, v in counter.items():
            per = v / len(y) * 100
            logging.info(f"{datetime.datetime.now()}: Class={k}, n={v} ({per:.3f}%)")
            
        df_bal = pd.DataFrame(X, columns=feats)
        df_bal["steps"] = np.array(y)
        df_bal.to_csv(f"output/{name_df}_balanced.csv")

        df_test = pd.DataFrame(X_test, columns=feats)
        df_test["steps"] = np.array(y_test)
        df_test["smiles"] = list(smiles_test)
        df_test.to_csv(f"output/{name_df}_test.csv")
        logging.info("=====================================================================\n\n")