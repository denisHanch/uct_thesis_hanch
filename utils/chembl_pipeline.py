import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt

from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import Chem

import time
import logging
import datetime
import requests 

import os

import random

seed = 45
random.seed(seed)

start = datetime.datetime.now()
output_folder="output"

logging.basicConfig(filename='chembl_pipe.log', level=logging.INFO)
logging.info(f"{datetime.datetime.now()}: Start")

dfs_all = []

# Read HDF5 files and transform them into dataframes
for e in range(5):
    #fil = f"aiZynthOutput_MULTI_{e}.hdf5"
    try:
        filename = f"AiZynthOutData/aiZynthOutput_MULTI_{e}.hdf5"
        f = pd.read_hdf(filename, "table")
        dfs_all.append(f)
        filename = f"AiZynthOutData/aiZynthOutput_warn_MULTI_{e}.hdf5"
        f = pd.read_hdf(filename, "table")
        dfs_all.append(f)
    except Exception as e:
        logging.info(f"{datetime.datetime.now()}\tError {e}")
        logging.info(f"{datetime.datetime.now()}: \tProblem while reading {filename}")
        logging.info("")
        continue

# Concat DFs
df = pd.concat(dfs_all, ignore_index=True)[["target", "number_of_steps", "is_solved"]]

logging.info(f"{datetime.datetime.now()}: DF created")

# List of smiles, canonized with RDKit
smiles = []

# List of steps, corresponding to canonized smiles
steps = []

# Duplicates counter
dup = 0

"""
The three errors below mean the number of times, 
the API response had other code, rather than 200
"""
# Errors during standardization
stand = 0

# Errors during parent generation
par = 0

# Errors during checker
check = 0

# The number of molecules, which returned any error from checker
prob = 0

# API endpoints URLs
stand_url = "https://www.ebi.ac.uk/chembl/api/utils/standardize"
check_url = "https://www.ebi.ac.uk/chembl/api/utils/check"
parent_url = "https://www.ebi.ac.uk/chembl/api/utils/getParent"

# Counts of the checker component errors
errors = {}

logging.info(f"{datetime.datetime.now()}: DF shape {df.shape}")

# This line helps add header to the logging file, in case some molecules cause errors during RDKit processing
problem = True

# Counter of molecules, which caused problems during RDKit processing
proc_prob = 0

# List of structures, which passed the pipeline

final = []
header_final = ["smiles", "orig_smiles", "steps", "orig_index", "is_solved", "pipeline_status"]

for i, el in enumerate(zip(df.target, df.number_of_steps, df.is_solved)):
    smil, num_of_st, solved = el
    try:
        # Create molecule from SMILES
        mol = Chem.MolFromSmiles(smil)

        # Remove stereochemistry
        Chem.rdmolops.RemoveStereochemistry(mol)

        # Convert molecule back into SMILES (canonical)
        smile = Chem.MolToSmiles(mol)

        # Check whether the duplicate of the canonical SMILES exist
        if smile in smiles:
            dup +=1
            logging.info(f"{datetime.datetime.now()}\tError. Duplicated smiles detected after canonization. ID {i}")
            logging.info(f"{datetime.datetime.now()}\tCanonized SMILES: {smile}")
            logging.info(f"{datetime.datetime.now()}\tOriginal  SMILES: {smil}")
            logging.info("")
            try:
                Draw.MolToFile(mol, f"output/duplicates/molecule_id_in_original_df_{i}.png")
            except:
                logging.info(f"{datetime.datetime.now()}: Unable to draw molecule from SMILES")
                logging.info(f"{datetime.datetime.now()}\tCanonized SMILES: {smile}")
                logging.info(f"{datetime.datetime.now()}\tOriginal  SMILES: {smil}")
                logging.info("")
            continue

        # Append canonical SMILES to the list
        smiles.append(smile)

        # Create molfile from SMILES - endpoint require molfiles or SDFs as input
        struct = Chem.MolToMolBlock(mol)
    except Exception as e:
        proc_prob += 1
        if problem:
            logging.info(f"{datetime.datetime.now()}: Molecules with problems:")
            problem = False
        logging.info(f"{datetime.datetime.now()}\tError {e}")
        logging.info(f"{datetime.datetime.now()}\tPROBLEM WITH ID {i}")
        logging.info("")
        try:
            Draw.MolToFile(mol, f"output/problems/gen_prob_{i}.png")
        except:
            logging.info(f"{datetime.datetime.now()}: Unable to draw molecule from SMILES")
            logging.info(f"{datetime.datetime.now()}\tCanonized SMILES: {smile}")
            logging.info(f"{datetime.datetime.now()}\tOriginal  SMILES: {smil}")
            logging.info("")
        continue

    # Pipe molecule through the Standardizer endpoint
    s = requests.post(stand_url, data=struct.encode('utf-8'))
    if s.status_code == 200:
        struct = s.json()[0]["standard_molblock"]
        mol = Chem.MolFromMolBlock(struct)
    else:
        stand += 1
        logging.info(f"{datetime.datetime.now()}\tPROBLEM WITH STANDARDIZATION. ID {i}")
        logging.info("")
        try:
            Draw.MolToFile(mol, f"output/problems/stand_prob_{i}.png")
        except:
            logging.info(f"{datetime.datetime.now()}: Unable to draw molecule from SMILES")
            logging.info(f"{datetime.datetime.now()}\tCanonized SMILES: {smile}")
            logging.info(f"{datetime.datetime.now()}\tOriginal  SMILES: {smil}")
            logging.info("")
        continue
    
    # Get the parent compound
    p = requests.post(parent_url, data=struct.encode('utf-8'))
    if p.status_code == 200:
        struct = p.json()[0]["parent_molblock"]
        mol = Chem.MolFromMolBlock(struct)
        
    else:
        par += 1
        logging.info(f"{datetime.datetime.now()}\tPROBLEM WITH PARENTIZATION. ID {i}")
        logging.info("")
        try:
            Draw.MolToFile(mol, f"output/problems/parent_prob_{i}.png")
        except:
            logging.info(f"{datetime.datetime.now()}: Unable to draw molecule from SMILES")
            logging.info(f"{datetime.datetime.now()}\tCanonized SMILES: {smile}")
            logging.info(f"{datetime.datetime.now()}\tOriginal  SMILES: {smil}")
            logging.info("")
        continue

    # Pipe through the checker component
    c = requests.post(check_url, data=struct.encode('utf-8'))
    if c.status_code == 200:
        if len(c.json()[0]) > 0:
            
            # Add to the final DF only smiles with the following error
            if len(c.json()[0]) == 1 and c.json()[0][0][1] == "InChI: Omitted undefined stereo":
                line = [smile, smil, num_of_st, i, solved, "warn"]
                final.append(line)

            prob += 1

            seven = False
            stereo = False
            for el in c.json()[0]:
                err = str(el[0])
                try:
                    errors[err]["cnt"] += 1
                    try:
                        errors[err]["msgs"][el[1]]["cnt"] += 1
                        errors[err]["msgs"][el[1]]["smiles"].append(smile)
                    except KeyError:
                        errors[err]["msgs"][el[1]] = {
                                                 "smiles": [smile],
                                                 "cnt": 1
                                                }



                except KeyError:
                    errors[err] = {
                                    "cnt": 1,
                                    #"msg": [el[1]]
                                    "msgs": {el[1]:
                                                {
                                                 "smiles": [smile],
                                                 "cnt": 1
                                                }
                                            }

                                  }
        else:
            line = [smile, smil, num_of_st, i, solved, "no_warn"]
            final.append(line)

    else:
        check += 1
        logging.info(f"{datetime.datetime.now()}\tPROBLEM WITH CHECKING. ID {i}")
        logging.info("")


final_df = pd.DataFrame(final, columns=header_final)
final_df.to_csv(f"{output_folder}/final_data_after_pipeline.csv", index=False)

# LOG THE INFO ABOUT PROCESSING ERRORS
if not problem:
    logging.info(f"{datetime.datetime.now()}\t{proc_prob} molecules totally caused problems during RDKit processing (were not pipelined AT ALL)")
    logging.info("")

if stand > 0:
    logging.info(f"{datetime.datetime.now()}\t{stand} molecules totally caused problems during standardization")
    logging.info("")

if par > 0:
    logging.info(f"{datetime.datetime.now()}\t{par} molecules totally caused problems during parentization")
    logging.info("")

if check > 0:
    logging.info(f"{datetime.datetime.now()}\t{check} molecules totally caused problems during checking")
    logging.info("")

if dup > 0:
    logging.info(f"{datetime.datetime.now()}\t{dup} duplicates")
    logging.info("")


logging.info("=======================================\n")
logging.info(f"{datetime.datetime.now()}: Errors dict:")
logging.info(errors)
logging.info("=======================================\n")

logging.info(f"{datetime.datetime.now()}: Errors decomposition:")
for err in errors:
    logging.info(f"\tCode {err}. Counts {errors[err]['cnt']}. Number of unique error messages {len(errors[err]['msgs'])}")
    logging.info("\tMessages:")
    for msg_ind, msg in enumerate(errors[err]['msgs']):
        #logging.info(f"\t\t{msg}")
        logging.info(f"\t\t{msg}| COUNT: {errors[err]['msgs'][msg]['cnt']}")
        if err == "2":
            # As there are too many errors of type 2, take 20 random molecules to draw
            lenn = len(errors[err]['msgs'][msg]["smiles"])
            for i in range(20):
                ind = random.randrange(lenn)
                sm = errors[err]['msgs'][msg]["smiles"][ind]
                mol = Chem.MolFromSmiles(sm)
                try:
                    Draw.MolToFile(mol, f"output/problems/msg/{err}_{msg_ind}_{ind}.png")
                except Exception as e:
                    logging.info(f"{datetime.datetime.now()}: Unable to draw molecule from SMILES")
                    logging.info(f"{datetime.datetime.now()}\tCanonized SMILES: {sm}")
                    logging.info("")

        else:
            # For other types of errors, draw all the molecules
            for ind in range(len(errors[err]['msgs'][msg]["smiles"])):
                sm = errors[err]['msgs'][msg]["smiles"][ind]
                mol = Chem.MolFromSmiles(sm)
                try:
                    Draw.MolToFile(mol, f"output/problems/msg/{err}_{msg_ind}_{ind}.png")
                except Exception as e:

                    logging.info(f"{datetime.datetime.now()}: Unable to draw molecule from SMILES")
                    logging.info(f"{datetime.datetime.now()}\tCanonized SMILES: {sm}")
                    logging.info(f"{datetime.datetime.now()}\t\tException msg: {e}")
                    logging.info("")

    logging.info("=======================================\n")

logging.info(f"{datetime.datetime.now()}: Finished. Processing took {datetime.datetime.now() - start}")