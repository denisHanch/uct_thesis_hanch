import os
import pandas as pd
import seaborn as sns
import itertools as it
import matplotlib.pyplot as plt

from rdkit import Chem
from syba.syba import SybaClassifier

from utils.SA_Score import sascorer
from scscore.standalone_model_numpy import SCScorer

output_path = "/home/hanchard/scores"

def plot_score_plot(dataframe, score_name, x_name):

    if x_name == "steps":
        vs = "number of steps"
        xlab = '\n# of steps'
        w, h = 20, 12
    else:
        vs = f'{x_name.upper()} scores'
        xlab = f'{x_name.upper()} Score'
        w, h = 20, 18

    fig, ax = plt.subplots(figsize=(w, h))
    sns.set(font_scale=1.4)
    sns.scatterplot(data=dataframe, x=x_name, y=score_name)
    plt.xlabel(xlab)
    plt.ylabel(f'{score_name.upper()} score')
    plt.title(f'Distribution of {score_name.upper()} scores against the {vs}')
    plt.savefig(f"{output_path}/{score_name}_vs_{x_name}.png", format="png")

project_root = os.path.dirname(os.path.dirname(__file__))
full_file = "/home/hanchard/benchmarking/fullData/final_data_after_pipeline.csv"

df = pd.read_csv(full_file)

# CALCULATE SYBA
syba = SybaClassifier()
syba.fitDefaultScore()

df['syba'] = df.smiles.apply(lambda smi: syba.predict(smi))
print("CALCULATED SYBA")

# CALCULATE SA
df["sa"] = df.smiles.apply(lambda smi: sascorer.calculateScore(Chem.MolFromSmiles(smi)))
print("CALCULATED SA")

# CALCULATE SC
model = SCScorer()
model.restore(os.path.join(project_root, 'models', 'full_reaxys_model_1024bool', 'model.ckpt-10654.as_numpy.json.gz'))
df['sc'] = df.smiles.apply(lambda smi: model.get_score_from_smi(smi))
print("CALCULATED SC")

score_list = ['syba', 'sa', 'sc']
for score in score_list:
    plot_score_plot(df[df.is_solved == True], score, "steps")


for s1, s2 in list(it.combinations(score_list, 2)):
    plot_score_plot(df[df.is_solved == True], s1, s2)

df.to_csv(f"{output_path}/final_data_after_pipeline.csv", index = False)