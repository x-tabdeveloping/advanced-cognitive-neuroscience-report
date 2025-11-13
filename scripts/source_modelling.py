import json
import os
from collections import defaultdict
from os import chdir
from os.path import join
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
from mord import LogisticAT
from scipy.stats import combine_pvalues, kendalltau
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm


@ignore_warnings(category=ConvergenceWarning)
def get_cv_score(estimator, X, y):
    lr = make_pipeline(StandardScaler(), estimator)

    def tau(y_true, y_pred):
        res = kendalltau(y_true, y_pred)
        return res.statistic

    def p_val(y_true, y_pred):
        res = kendalltau(y_true, y_pred)
        return res.pvalue

    tau_scorer = make_scorer(tau)
    pval_scorer = make_scorer(p_val)
    mean_tau = np.mean(cross_val_score(lr, X, y, cv=4, scoring=tau_scorer))
    combined_p = combine_pvalues(cross_val_score(lr, X, y, cv=4, scoring=pval_scorer))
    return mean_tau, combined_p


def main():
    source_dir = Path("./sources")
    source_files = list(source_dir.glob("*.joblib"))
    subject_to_file = {file.stem: file for file in source_files}
    res = defaultdict(dict)
    subjects = list(subject_to_file.keys())
    for subject in tqdm(subjects, desc="Going through subjects"):
        try:
            print(f"Loading data for {subject}")
            data = joblib.load(subject_to_file[subject])
            y = data["y"]
            for lobe in ["occipital", "frontal"]:
                res[subject][lobe] = dict()
                res[subject][lobe]["tau"] = []
                res[subject][lobe]["p"] = []
                print(f"Running regression for {lobe}")
                # n_epochs, n_sources, n_times
                X = data["Xs"][lobe]
                # (n_epochs, n_times, n_channels)
                X = np.transpose(X, (0, -1, -2))
                n_epochs, n_times, n_channels = X.shape
                print("Reducing dimensionality with PCA")
                pca_model = PCA(30, random_state=42)
                X_pca = pca_model.fit_transform(X.reshape(-1, X.shape[-1]))
                X_pca = X_pca.reshape(n_epochs, n_times, -1)
                exvar = np.sum(pca_model.explained_variance_ratio_) * 100
                print(f"PCA successful and explains {exvar:.2f}% of variance in the data.")
                print(
                    "Trying to predict experience with ordinal regression for each time point"
                )
                # n_times, n_epochs, n_sources
                X_times = np.transpose(X_pca, (-2, 0, -1))
                for X_time in tqdm(
                    X_times, desc="Fitting ordinal regression for all time points"
                ):
                    tau, p = get_cv_score(LogisticAT(alpha=1.0), X_time, y)
                    res[subject][lobe]["tau"].append(tau)
                    res[subject][lobe]["p"].append(p)
                print("Mean tau", np.mean(res[subject][lobe]["tau"]))
                print("Combined p:", np.mean(res[subject][lobe]["p"]))
        except Exception as e:
            print(f"WARNING: Modelling failed for subject {subject} for reason: {e}")
            res[subject]["error"] = str(e)
            continue
        print("Saving intermediate results")
        with open("source_space_results_reduced.json", "w") as out_file:
            out_file.write(json.dumps(res))
    print("DONE")


if __name__ == "__main__":
    main()
