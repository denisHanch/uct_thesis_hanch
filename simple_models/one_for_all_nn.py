import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
import time
import logging
import datetime

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
  recall_score, precision_score, roc_curve, auc, f1_score

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

from create_roc import create_roc_graph_OvR

from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras import Input
from keras.models import Sequential

from tensorflow.keras.metrics import AUC, Precision, Recall

from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

from numpy.random import seed
seed(1)
import tensorflow
tensorflow.random.set_seed(2)

# The following imports and lines of code just solve some configuration problems
#

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

prefix = sys.argv[1]

def create_NN(input_layer_size, output_layer_size, mod_name):

    model = Sequential()

    if "nn_1bhl" in mod_name:

        if "2" in mod_name:
          mult = 2
        else:
          mult = 4
        #model.add(Input(shape=(input_layer_size,)))

        layer_size = input_layer_size*mult
        model.add(Dense(layer_size, input_dim=input_layer_size, activation = 'relu'))
        model.add(Dense(output_layer_size, activation='softmax'))

        #Define the model
        #model = Model(inputs=input_l, outputs=output_l)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])

    elif "nn_1shl" in mod_name:

        if "2" in mod_name:
          div = 2
        else:
          div = 4

        #model.add(Input(shape=(input_layer_size,)))

        layer_size = input_layer_size//div
        model.add(Dense(layer_size, input_dim=input_layer_size, activation = 'relu'))

        while layer_size >= 32:
            layer_size = layer_size//div
            dense_l = Dense(layer_size, activation = 'relu')
        
        model.add(Dense(output_layer_size, activation='softmax'))

        #Define the model
        #model = Model(inputs=input_l, outputs=output_l)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])

    else:

        #model.add(Input(shape=(input_layer_size,)))

        layer_size = input_layer_size//4
        model.add(Dense(layer_size, input_dim=input_layer_size, activation = 'relu'))
        model.add(Dense(layer_size, activation = 'relu'))

        while layer_size >= 128:
            layer_size = layer_size//4
            model.add(Dense(layer_size, activation = 'relu'))
            model.add(Dense(layer_size, activation = 'relu'))
        
        model.add(Dense(output_layer_size, activation='softmax'))

        #Define the model
        #model = Model(inputs=input_l, outputs=output_l)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', AUC(), Precision(), Recall()])

    return model


def write_class_report(report, name):
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"output/{name}_report.csv")


def evaluate(model, test_features, test_labels, name, mod):
    """
    Evaluates the accuracy of the given model
    """
    predictions = model.predict(test_features)

    # Get indices of the max elements along axis
    predictions = np.argmax(predictions, axis=1)
    test_labels = np.argmax(test_labels, axis=1)


    accuracy = accuracy_score(test_labels, predictions)*100
    precision = precision_score(test_labels, predictions, average='macro')*100
    recall = recall_score(test_labels, predictions, average='macro')*100
    f1 = f1_score(test_labels, predictions, average="macro")*100

    logging.info(f"{datetime.datetime.now()}: {name}| {mod} model Performance")
    logging.info(f"{datetime.datetime.now()}: {name}| Accuracy = {accuracy:0.2f}%")
    logging.info(f"{datetime.datetime.now()}: {name}| Precision = {precision:0.2f}%")
    logging.info(f"{datetime.datetime.now()}: {name}| Recall = {recall:0.2f}%")
    logging.info(f"{datetime.datetime.now()}: {name}| F1 = {f1:0.2f}%")
    logging.info("\n")

    report = classification_report(test_labels, predictions, output_dict=True)

    return accuracy, precision, recall, f1, report


def conf_matrix(test_feat, test, model, counts, class_names, file=""):
    """
    Creates 2 confusion matices.
    The first one with counts
    The second one with fractions
    """

    postfix = file
    ans = model.predict(test_feat)

    ans = np.argmax(ans, axis=1)
    test = np.argmax(test, axis=1)

    # First, calculate usual confusion matrix
    matrix = confusion_matrix(test, ans)

    fig, ax = plt.subplots(figsize=(12,9))
    #sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={"size": 10},
              cmap=plt.cm.Greens, fmt='d', xticklabels=class_names, yticklabels=class_names)
    #class_names = [str(x+1) for x in range(counts)]
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5

    ax.set_xticks(tick_marks2,      minor=True)
    #ax.set_xticklabels(class_names, minor=True)

    if len(file) > 0:
      postfix = f" (for {postfix})"

    plt.xlabel(f'\nPredicted # of steps{postfix}')
    plt.ylabel('True # of steps')
    plt.title('Confusion Matrix')
    plt.savefig(f"output/{model_name}_{file}_counts.png", format="png")

    plt.clf()

    # Now calculate confusion matrix for predicted counts
    matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 9))
    #sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={"size": 10},
              cmap=plt.cm.Greens, linewidths=0.2, fmt='0.3f', xticklabels=class_names, yticklabels=class_names)
    #class_names = [str(x+1) for x in range(counts)]
    tick_marks = np.arange(len(class_names))
    tick_marks2 = tick_marks + 0.5

    ax.set_xticks(tick_marks2, minor=True)
    #ax.set_xticklabels(class_names, minor=True)

    plt.xlabel(f'\nPredicted # of steps{postfix}')
    plt.ylabel('True # of steps')
    plt.title('Confusion Matrix')
    plt.savefig(f"output/{model_name}_{file}.png", format="png")


def compare_base_vs_best_model(name, path_prefix, feats, class_counts=None):
    """
    Finds best parameters for RF model, and creates confusion matrix
    """

    df = pd.read_csv(f"{path_prefix}/{name}.csv")
    df = df[df.steps > 0].reset_index().drop("index", axis=1)
    cutoff_step = 10
    df = df[df.steps <= cutoff_step].reset_index().drop("index", axis=1)

    train_ind, test_ind = train_test_split(df.index, random_state=seed, stratify=df.steps, test_size=0.10)
    df.iloc[test_ind].to_csv(f"output/test_{name}.csv", index=False)
    df = df.iloc[train_ind].reset_index().drop("index", axis=1)

    offset = 1
    class_names = np.unique(df.steps)

    # Get number of unique classes
    if not class_counts:
        class_counts = len(df.steps.unique())
    X = df[[*feats]]
    X = np.array(X.astype('int64'))
    y = df.steps
    y = LabelEncoder().fit_transform(y)
    y = to_categorical(y, num_classes=class_counts)

    num_of_features = len(feats)

    # Take test data from test dataframe, in case the data was balanced
    if "balanced" in name:
      X_train = X
      y_train = y

      name_test = name.replace("balanced", "test")
      df_test = pd.read_csv(f"{path_prefix}/{name_test}.csv")
      df_test = df_test[df_test.steps <= cutoff_step]
      X_test = df_test[[*feats]]
      y_test = df_test.steps
      y_test = LabelEncoder().fit_transform(y_test)
      y_test = to_categorical(y_test, num_classes=class_counts)
    else:
      # Stratified train/test split
      X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=seed, stratify=y_train)

    #y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)
    #y_val = to_categorical(y_val)


    base_model = create_NN(num_of_features, class_counts, model_name)
    
    history = base_model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
    base_acc, base_precision, base_recall, base_f1, class_report = evaluate(base_model, X_test, y_test, name, "NN")

    conf_matrix(X_test, y_test, base_model, class_counts, class_names, name)
    logging.info("-----------------\n")
    plot_model(base_model, to_file=f"output/{model_name}_architecture.png",
               expand_nested=True, show_shapes=True, show_layer_names=True)

    auc_ovr = create_roc_graph_OvR(X_test, y_test, base_model, "output", f"{model_name}_{name}_ROC_OvR.png", "h", mod='nn', offset=offset, burg_model=False)
    #auc_ovo = create_roc_graph_OvO(X_test, y_test, best_mod, "output", f"{model_name}_{name}_ROC_OvO.png", f"{model_name}_{name}_ROC_AUC_OvO.csv")
    logging.info("-----------------\n")

    ovr_macro = 0
    ovr_weigh = 0
    metric_name = 'AUC_OvR'
    for cl in auc_ovr:
        class_report[str(cl)][metric_name] = auc_ovr[cl]
        ovr_macro += auc_ovr[cl]
        ovr_weigh += auc_ovr[cl]*(class_report[str(cl)]['support']/X_test.shape[0])

    ovr_macro = ovr_macro/len(auc_ovr)
    class_report['macro avg'][metric_name + "_MAN"] = ovr_macro
    #class_report['macro avg'][metric_name] = auc_ovr['macro']
    class_report['weighted avg'][metric_name + "_MAN"] = ovr_weigh
    #class_report['weighted avg'][metric_name] = auc_ovr['weighted']

    write_class_report(class_report, name)

    filename = f"output/{model_name}_model_{name}"
    base_model.save(filename)


seed = 66

prefix = sys.argv[1]
model_name = sys.argv[2]

start = time.time()
logging.basicConfig(filename=f'{model_name}_bench.log', level=logging.INFO)

for bit in [1024, 2048]:
    feats = [str(x) for x in range(bit)]
    for rad in [2, 4]:
      
      base_name = "morgan_fullData"
      name_df = base_name + f"_{rad}_{bit}"
      path_prefix = "fullData"
      compare_base_vs_best_model(name_df, path_prefix, feats)

      if "balanced" in prefix:
        dataset_type = "balanced"
      else:
        dataset_type = "undersampled"
      compare_base_vs_best_model(name_df + f"_{dataset_type}", path_prefix, feats)
      logging.info("===========================================\n")

logging.info(f"Time taken: {time.time() - start}")
