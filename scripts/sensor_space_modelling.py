import json
import os
from collections import defaultdict
from os.path import join
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mord import LogisticAT
from scipy.stats import combine_pvalues, kendalltau
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

subjects = [f"0{subject_id}" for subject_id in range(163, 170 + 1)]
MEG_paths = dict()
for subject in subjects:
    subject_path = f"/work/workshop_data/{subject}"
    folder = next(os.walk(subject_path))[1][0]
    MEG_paths[subject] = os.path.join(subject_path, folder)
behavioral_paths = dict()
logfiles = list(Path("/work/workshop_data/behavioural_logs/").glob("*.csv"))
for subject in subjects:
    for logfile in logfiles:
        if logfile.stem.startswith(subject):
            behavioral_paths[subject] = logfile
subjects_dir = "/work/freesurfer"

BADS = {
    "0163": [],
    "0164": ["MEG2412"],
    "0165": [],
    "0166": ["MEG0722"],
    "0167": ["MEG0722"],
    "0168": [],
    "0169": [],
    "0170": ["MEG0423", "MEG1443", "MEG2621"],
}


def load_training_data(subject_id: str) -> tuple[np.ndarray, np.ndarray]:
    MEG_path = MEG_paths[subject_id]
    raw = mne.io.read_raw_fif(join(MEG_path, "workshop_2025_raw.fif"), preload=True)
    raw.filter(h_freq=40, l_freq=1)
    raw.info["bads"] = BADS[subject_id]
    events = mne.pick_events(
        mne.find_events(
            raw,
            stim_channel="STI101",
            shortest_event=1,
        ),
        include=[1, 3, 4, 6, 8, 10, 12, 14, 16],
    )
    behaviour = pd.read_csv(behavioral_paths[subject_id], index_col=False)
    # Filtering so it's only events for stimuli
    stimulus_clarity_events = np.copy(events[np.isin(events[:, -1], [1, 3])])
    # Remapping to clarity
    stimulus_clarity_events[:, -1] = behaviour["subjective_response"]
    print("Counts: ", behaviour["subjective_response"].value_counts())
    epochs = mne.Epochs(
        raw,
        events=stimulus_clarity_events,
        tmin=-0.2,
        tmax=1.0,
        baseline=(None, -0.050),
        preload=True,
    )
    # (n_epochs, n_channels, n_times)
    X = epochs.get_data(picks="meg")
    y = epochs.events[:, -1]
    # Transposing, so time comes before the channels
    # (n_epochs, n_times, n_channels)
    X = np.transpose(X, (0, -1, -2))
    return X, y


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
    res = defaultdict(dict)
    for subject in tqdm(subjects, desc="Going through subjects"):
        res[subject]["tau"] = []
        res[subject]["p"] = []
        try:
            print(f"Loading data for {subject}")
            X, y = load_training_data(subject)
            n_epochs, n_times, n_channels = X.shape
            print("Reducing dimensionality with PCA")
            X_pca = PCA(50).fit_transform(X.reshape(-1, X.shape[-1]))
            X_pca = X_pca.reshape(n_epochs, n_times, -1)
            print("Trying to predict experience with logreg for each time point")
            # n_times, n_epochs, n_channels
            X_times = np.transpose(X_pca, (-2, 0, -1))
            for X_time in tqdm(
                X_times, desc="Fitting logistic regression for all time points"
            ):
                tau, p = get_cv_score(LogisticAT(alpha=1.0), X_time, y)
                res[subject]["tau"].append(tau)
                res[subject]["p"].append(p)
            print("Mean tau", np.mean(res[subject]["tau"]))
            print("Combined p:", np.mean(res[subject]["p"]))
        except Exception as e:
            print(f"WARNING: Modelling failed for subject {subject} for reason: {e}")
            res[subject]["error"] = str(e)
            continue
        print("Saving intermediate results")
        with open("results.json", "w") as out_file:
            out_file.write(json.dumps(res))
    print("DONE")


if __name__ == "__main__":
    main()
