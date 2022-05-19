import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time
import pickle
import logging
import datetime

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, \
  recall_score, precision_score, roc_curve, auc, f1_score


from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from sklearn.model_selection import RandomizedSearchCV

from create_roc import create_roc_graph_OvR


def best_model_search_df(X, y, file, seed, cv):
    """
    Searches for the best parameters through the grid
    """
    # Function to measure the quality of a split
    criterion = ["gini", "entropy"]
    # Strategy used to choose the split at each node
    splitter = ["best", "random"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6, 8, 10]

    # Create the grid
    grid = {'criterion': criterion,
            'splitter': splitter,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'min_samples_split': min_samples_split}

    selector = DecisionTreeClassifier(random_state=seed)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    selector_random = RandomizedSearchCV(estimator=selector, param_distributions=grid,
                                   n_iter=100, cv=cv, verbose=2, random_state=seed, n_jobs=-1)
    # Fit the random search model
    selector_random.fit(X, y)

    logging.info(f"{datetime.datetime.now()}: {file} best parameters:")
    logging.info(f"\t{selector_random.best_params_}")
    logging.info("\n")

    return selector_random.best_estimator_


def best_model_search_rf(X, y, file, seed, cv):
    """
    Searches for the best parameters through the grid
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=1500, num=10)]
    # Function to measure the quality of a split
    criterion = ["gini", "entropy"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4, 6, 8, 10]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the grid
    grid = {'n_estimators': n_estimators,
            'criterion': criterion,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap,
            'min_samples_split': min_samples_split}

    selector = RandomForestClassifier(random_state=seed)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    selector_random = RandomizedSearchCV(estimator=selector, param_distributions=grid,
                                   n_iter=100, cv=cv, verbose=2, random_state=seed, n_jobs=-1)
    # Fit the random search model
    selector_random.fit(X, y)

    logging.info(f"{datetime.datetime.now()}: {file} best parameters:")
    logging.info(f"\t{selector_random.best_params_}")
    logging.info("\n")

    return selector_random.best_estimator_

def best_model_search_svc(X, y, file, seed, cv):
    """
    Searches for the best parameters through the grid
    """
    # Regularization parameter
    c = [1, 10, 100, 1000]
    # ernel type to be used in the algorithm
    kernel = ["linear", "poly", "rbf", "sigmoid"]
    # Degree of the polynomial kernel function
    degree = [x for x in range(10)]
    # Kernel coefficient
    gamma = ['scale', 'auto'] # add float here. log(10e-3) - log(10e3)

    # Create the grid
    grid = {'C': c,
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'probability': [True]}

    selector = SVC(random_state=seed)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    selector_random = RandomizedSearchCV(estimator=selector, param_distributions=grid,
                                   n_iter=100, cv=cv, verbose=2, random_state=seed, n_jobs=-1)
    # Fit the random search model
    selector_random.fit(X, y)

    logging.info(f"{datetime.datetime.now()}: {file} best parameters:")
    logging.info(f"\t{selector_random.best_params_}")
    logging.info("\n")

    return selector_random.best_estimator_

def write_class_report(report, name):
    df = pd.DataFrame(report).transpose()
    df.to_csv(f"output/{name}_report.csv")



def evaluate(model, test_features, test_labels, name, mod):
    """
    Evaluates the accuracy of the given model
    """
    predictions = model.predict(test_features)
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

    # First, calculate usual confusion matrix
    matrix = confusion_matrix(test, ans)
    #class_names = model.classes_
    fig, ax = plt.subplots(figsize=(12,9))
    #sns.set(font_scale=1.4)
    sns.heatmap(matrix, annot=True, annot_kws={"size": 10},
              cmap=plt.cm.Greens, linewidths=0.2, fmt='d', xticklabels=class_names, yticklabels=class_names)
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

    # Get number of unique classes
    if not class_counts:
        class_counts = len(df.steps.unique())
    X = df[[*feats]]
    y = df.steps

    # Take test data from test dataframe, in case the data was balanced
    if "balanced" in name:
      X_train = X
      y_train = y

      name_test = name.replace("balanced", "test")
      df_test = pd.read_csv(f"{path_prefix}/{name_test}.csv")
      df_test = df_test[df_test.steps <= cutoff_step]
      X_test = df_test[[*feats]]
      y_test = df_test.steps
    else:
      # Stratified train/test split
      X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)

    # process X-train with k-fold, and evaluate on y_test

    model = models[model_name]['mod']
    search_func = models[model_name]['func']

    base_model = model
    base_model.fit(X_train, y_train)
    base_acc, base_precision, base_recall, base_f1, base_report = evaluate(base_model, X_test, y_test, name, "base")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    best_mod = search_func(X_train, y_train, name, seed, skf)
    best_acc, best_precision, best_recall, best_f1, class_report = evaluate(best_mod, X_test, y_test, name, "best_params")

    logging.info(f"{datetime.datetime.now()}: {name}| Accuracy improvement of {( 100*(best_acc-base_acc)/base_acc):0.2f}%")
    logging.info(f"{datetime.datetime.now()}: {name}| Precision improvement of {( 100*(best_precision-base_precision)/base_precision):0.2f}%")
    logging.info(f"{datetime.datetime.now()}: {name}| Recall improvement of {( 100*(best_recall-base_recall)/base_recall):0.2f}%")
    logging.info(f"{datetime.datetime.now()}: {name}| F1 improvement of {( 100*(best_f1-base_f1)/base_f1):0.2f}%")
    logging.info("\n")
    conf_matrix(X_test, y_test, best_mod, class_counts, best_mod.classes_, name)
    auc_ovr = create_roc_graph_OvR(X_test, y_test, best_mod, "output", f"{model_name}_{name}_ROC_OvR.png", "d", mod='ml', offset=offset, burg_model=False)
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
    filename = f"output/{model_name}_model_{name}.sav"
    pickle.dump(best_mod, open(filename, 'wb'))

seed = 66

prefix = sys.argv[1]
model_name = sys.argv[2]

models = {"dt": {"mod": DecisionTreeClassifier(criterion="gini", splitter="best", random_state=seed),
                 "func": best_model_search_df},
          "rf": {"mod": RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=seed),
                 "func": best_model_search_rf}, 
          "svc": {"mod": SVC(C=1.0, kernel="rbf", degree=3, gamma="scale", random_state=seed, probability=True),
                  "func": best_model_search_svc}}

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
