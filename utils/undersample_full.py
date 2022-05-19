import pandas as pd
import numpy as np
import imblearn

import rdkit
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprint, GetMorganFingerprintAsBitVect
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

import time
import logging
import datetime
from collections import Counter

from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logging.basicConfig(filename='undersample_full.log', level=logging.INFO)
logging.info(f"{datetime.datetime.now()}: Start")

seed = 66

for bit in [1024, 2048]:
    feats = [str(x) for x in range(bit)]
    for rad in [2, 4]:
        base_name = "morgan_fullData"
        name_df = base_name + f"_{rad}_{bit}"
        df = pd.read_csv("fullData/" + name_df + ".csv")

        steps_to_undersample = 1
        steps_for_count = 2

        restrict_steps = 10

        num_of_samples = df[df.steps == steps_for_count].shape[0]

        fps = [GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), rad, bit) for x in df[df.steps == steps_to_undersample].smiles]

        picker = MaxMinPicker()
        p = picker.LazyBitVectorPick(fps, len(fps), num_of_samples, seed=seed)

        picked_samples = df[df.steps == steps_to_undersample].iloc[p]
        new_df = pd.concat([picked_samples, df[df.steps != steps_to_undersample]], ignore_index=True)
        new_df = new_df[new_df.steps <= restrict_steps].reset_index()
        new_df.to_csv(f"output/{name_df}_undersampled.csv")

        X, y = np.array(df[[*feats]]), df.steps
        y = LabelEncoder().fit_transform(y)

        logging.info(f"{datetime.datetime.now()}: Class breakdown for {base_name} BEFORE UNDERSAMPLE")
        for k, v in dict(df.steps.value_counts()).items():
            per = v / len(df.steps) * 100
            logging.info(f"{datetime.datetime.now()}: Class={k}, n={v} ({per:.3f}%)")
        logging.info(f"")

        logging.info(f"{datetime.datetime.now()}: Class breakdown for {base_name} AFTER UNDERSAMPLE")
        for k, v in dict(new_df.steps.value_counts()).items():
            per = v / len(new_df.steps) * 100
            logging.info(f"{datetime.datetime.now()}: Class={k}, n={v} ({per:.3f}%)")
        logging.info(f"")

        logging.info(f"{dict(df.steps.value_counts())}")
        logging.info(f"{dict(new_df.steps.value_counts())}")

        logging.info("=====================================================================\n\n")