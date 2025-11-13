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
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.dummy import DummyClassifier
from scipy.stats import kendalltau
from scipy.stats import combine_pvalues
from mord import LogisticAT
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm


subjects = [f"0{subject_id}" for subject_id in range(163, 170 + 1)]
MEG_paths = dict()
for subject in subjects:
    subject_path = f"/work/MEG_data/workshop_data/{subject}"
    folder = next(os.walk(subject_path))[1][0]
    MEG_paths[subject] = os.path.join(subject_path, folder)
behavioral_paths = dict()
logfiles = list(Path("/work/MEG_data/workshop_data/behavioural_logs/").glob("*.csv"))
for subject in subjects:
    for logfile in logfiles:
        if logfile.stem.startswith(subject):
            behavioral_paths[subject] = logfile
subjects_dir = "/work/freesurfer"

SOURCE_paths = dict()
for subject in subjects:
    print(f"subject: {subject}")
    subject_path = Path(f"/work/MEG_data/workshop_data/{subject}")
    sub_files = list(subject_path.glob('*/*'))
    print(f"sub_files {sub_files}")
    trans = [x for x in sub_files if '-trans' in x.name][0]
    # go into freesurfer folder
    freesurfer = Path('/work/freesurfer')
    src = freesurfer / subject / "bem" / f"{subject}-oct-6-src.fif"
    bem = freesurfer / subject / "bem" / f"{subject}-5120-bem-sol.fif"
    # check file paths
    for name, path in [('src', src), ('bem', bem), ('trans', trans)]:
        if not path.exists():
            print(f"⚠️ Warning: {name} file not found for {subject}: {path}")
    # Store in dictionary
    SOURCE_paths[subject] = {
        'freesurfer': freesurfer,
        'trans': trans,
        'src': src,
        'bem': bem
    }

BADS = {
    "0163": [],
    "0164": ["MEG2412"],
    "0165": [],
    "0166": ["MEG0722"],
    "0167": ['MEG0722'],
    "0168": [],
    "0169": [],
    "0170": ['MEG0423', 'MEG1443', 'MEG2621'],
}

def load_epochs(subject_id: str) -> np.ndarray:
    MEG_path = MEG_paths[subject_id]
    raw = mne.io.read_raw_fif(join(MEG_path, "workshop_2025_raw.fif"), preload=True)
    raw.filter(h_freq=40, l_freq=1)
    raw.info["bads"] = BADS[subject_id]

    # Extract events
    events = mne.pick_events(mne.find_events(
        raw,
        stim_channel="STI101",
        shortest_event=1,
    ), include=[1, 3, 4, 6, 8, 10, 12, 14, 16])

    # Load behavioral data
    behaviour = pd.read_csv(behavioral_paths[subject_id], index_col=False)

    # Only keep clarity events
    stimulus_clarity_events = np.copy(events[np.isin(events[:, -1], [1, 3])])

    # Replace event IDs with subjective responses (1–4, etc.)
    stimulus_clarity_events[:, -1] = behaviour["subjective_response"]

    print("Counts: ", behaviour["subjective_response"].value_counts())

    epochs = mne.Epochs(
        raw,
        events=stimulus_clarity_events,
        tmin=-0.2,
        tmax=1.0,
        baseline=(None, -0.05),
        preload=True,
    )

    epochs_mag = epochs.copy().pick_types(meg='mag')

    numpy_epochs = epochs_mag.get_data()  # shape = (n_epochs, n_sensors, n_times)
    y = epochs.events[:, -1]
    return numpy_epochs, y


def main():
    out_dir = Path("epochs")
    out_dir.mkdir(exist_ok=True)

    for subject in tqdm(subjects, desc="Source reconstruction for all subjects"):

        print(f"\nRunning epoching for subject {subject}")
        numpy_epochs, y = load_epochs(subject)
        res = {"subject_id": subject, "epochs": numpy_epochs, "y": y}
        print("Saving")
        joblib.dump(res, out_dir.joinpath(f"{subject}.joblib"))
        print(f"Saved {subject}_epochs.npy with shape {numpy_epochs.shape}")

    print("DONE")


if __name__ == "__main__":
    main()
