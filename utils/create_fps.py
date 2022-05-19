import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

from rdkit.Chem import AllChem
from rdkit import Chem

import time
import logging
import datetime

import os

def create_morgan(df, name_to_csv, radius, bits, column_sm, column_step):
    fps = []
    for smile, step in zip(df[column_sm], df[column_step]):
        mol = Chem.MolFromSmiles(smile)
        fp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits))
        fp.append(step)
        fp.append(smile)
        fps.append(fp)

    #data = {'fps': fps,
    #        'steps': list(df[column_step])}
    df = pd.DataFrame(fps, columns=[*range(bits), "steps", "smiles"])
    df.to_csv(name_to_csv, index=False)
    logging.info(f"{datetime.datetime.now()}: {name_to_csv}")
    logging.info(f"{datetime.datetime.now()} HEADER:")
    logging.info(f"{df.head(2)}")
    logging.info("===========================\n")

output_folder="output"

logging.basicConfig(filename='create_fps.log', level=logging.INFO)
logging.info(f"{datetime.datetime.now()}: Start")

dfs = []
dfs_warn = []
dfs_all = []

# Read HDF5 files and transform them into dataframes
for e in range(5):
    #fil = f"aiZynthOutput_MULTI_{e}.hdf5"
    try:
        filename = f"AiZynthOutData/aiZynthOutput_MULTI_{e}.hdf5"
        f = pd.read_hdf(filename, "table")
        dfs.append(f)
        dfs_all.append(f)
        filename = f"AiZynthOutData/aiZynthOutput_warn_MULTI_{e}.hdf5"
        f = pd.read_hdf(filename, "table")
        dfs_warn.append(f)
        dfs_all.append(f)
    except:
        logging.info(f"{datetime.datetime.now()}: problem while reading {filename}, {e}")
        print("problem", e)
        continue

# Concat DFs into 3 distinct DFs: with no warnings, with warnings and a mixed one
d = pd.concat(dfs, ignore_index=True)
d_warn = pd.concat(dfs_warn, ignore_index=True)
d_all = pd.concat(dfs_all, ignore_index=True)

names = ["no_warn", "warns", "all"]

#Create morgan FPs from dataframe
to_c = []
for df, name in zip([d, d_warn, d_all], names):
    base_name = f"{output_folder}/morgan_" + name
    for rad in [2, 4]:
        for bit in [1024, 2048]:
            name = base_name + f"_{rad}_{bit}.csv"
            if len(to_c) < 4:
                to_c.append(name)
            create_morgan(df[df.is_solved == True], name, rad, bit, "target", 'number_of_steps')