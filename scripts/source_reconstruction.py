import os
from os.path import join
from pathlib import Path

import joblib
import mne
import numpy as np
import pandas as pd
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

SOURCE_paths = dict()
for subject in subjects:
    subject_path = Path(f"/work/workshop_data/{subject}")
    sub_files = list(subject_path.glob("*/*"))
    trans = [x for x in sub_files if "-trans" in x.name][0]
    # go into freesurfer folder
    freesurfer = Path("/work/freesurfer")
    src = freesurfer / subject / "bem" / f"{subject}-oct-6-src.fif"
    bem = freesurfer / subject / "bem" / f"{subject}-5120-bem-sol.fif"
    # check file paths
    for name, path in [("src", src), ("bem", bem), ("trans", trans)]:
        if not path.exists():
            print(f"⚠️ Warning: {name} file not found for {subject}: {path}")
    # Store in dictionary
    SOURCE_paths[subject] = {
        "freesurfer": freesurfer,
        "trans": trans,
        "src": src,
        "bem": bem,
    }

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


def load_sources(
    subject_id: str,
) -> tuple[np.ndarray, np.ndarray]:
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
        # event_id=dict(no_experience=1, weak_glimpse=2, almost_clear=3, clear_experience=4),
        tmin=-0.2,
        tmax=1.0,
        baseline=(None, -0.050),
        preload=True,
    )
    stcs = source_reconstruction(
        epochs=epochs,
        trans_path=SOURCE_paths[subject_id]["trans"],
        src_path=SOURCE_paths[subject_id]["src"],
        bem_path=SOURCE_paths[subject_id]["bem"],
    )
    lobel_labels = build_lobe_labels(
        subject=subject_id, subjects_dir=SOURCE_paths[subject_id]["freesurfer"]
    )
    lobe_stcs = create_lobe_stcs(stcs=stcs, lobe_labels=lobel_labels)
    # Extract features (X) per lobe
    X_dict = {}
    for lobe_name, stcs_list in lobe_stcs.items():
        X, _, _ = extract_features(epochs, stcs_list)  # y is same for all
        X_dict[lobe_name] = X  # shape: (n_epochs, n_vertices_in_lobe, n_times)
    y = epochs.events[:, -1]
    return X_dict, y


# -------
# SOURCE RECONSTRUCTION
def source_reconstruction(epochs, trans_path, src_path, bem_path):
    info = epochs.info
    fwd = mne.make_forward_solution(info, trans_path, src_path, bem_path)

    noise_cov = mne.compute_covariance(epochs, tmin=None, tmax=0)
    noise_cov.plot(epochs.info)
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, noise_cov
    )
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs, inverse_operator, lambda2=1.0, method="MNE"  # this is required
    )
    return stcs


def create_lobe_stcs(stcs, lobe_labels):
    """
    Create lobe_stcs as a dict[lobe] object where
    """
    # Extract for all lobes
    lobe_stcs = {}
    print("=" * 60)

    for lobe_name, label in lobe_labels.items():
        lobe_stcs[lobe_name] = extract_label_stcs(label, stcs)

        # Get stats from the first STC (assume all have same source space)
        example_stc = lobe_stcs[lobe_name][0]
        n_vertices = example_stc.data.shape[0]
        n_times = example_stc.data.shape[1]
        n_epochs = len(lobe_stcs[lobe_name])

        def get_label_vertices(label):
            """Return total number of vertices in a Label or BiHemiLabel."""
            if hasattr(label, "lh"):  # It's a BiHemiLabel
                return len(label.lh.vertices) + len(label.rh.vertices)
            else:  # It's a regular Label (lh or rh)
                return len(label.vertices)

        def get_label_hemi(label):
            """Return 'lh', 'rh', or 'both'."""
            if hasattr(label, "lh"):
                return "both"
            else:
                return label.hemi

        # Use helper functions to handle BiHemiLabel vs Label
        n_label_vertices = get_label_vertices(label)
        hemi = get_label_hemi(label)

        print(f"Lobe: {lobe_name.capitalize()}")
        print(f"  - Hemisphere: {hemi}")
        print(f"  - # Vertices in label: {n_label_vertices}")
        print(f"  - STC shape (vertices × time): {n_vertices} × {n_times}")
        print(f"  - # Epochs: {n_epochs}")
        print(
            f"  - Time range: {example_stc.times[0]:.3f} to {example_stc.times[-1]:.3f} sec"
        )

    return lobe_stcs


def extract_features(epochs, stcs):
    y = epochs.events[:, 2]
    n_events = len(y)
    n_times = stcs[0].data.shape[1]
    n_vertices = stcs[0].data.shape[0]
    X = np.zeros((n_events, n_vertices, n_times), dtype=np.float32)
    for i, stc in enumerate(stcs):
        X[i] = stc.data  # no need for .copy() unless modifying
    return X, y, n_times


def build_lobe_labels(subject, subjects_dir):
    labels = mne.read_labels_from_annot(subject, subjects_dir=subjects_dir)
    label_dict = {lbl.name: lbl for lbl in labels}

    lobe_rois = {
        "frontal": [
            "superiorfrontal",
            "rostralmiddlefrontal",
            "caudalmiddlefrontal",
            "parsopercularis",
            "parstriangularis",
            "parsorbitalis",
            "lateralorbitofrontal",
            "medialorbitofrontal",
            "precentral",
            "paracentral",
            "frontalpole",
        ],
        "parietal": [
            "superiorparietal",
            "inferiorparietal",
            "supramarginal",
            "postcentral",
            "precuneus",
        ],
        "temporal": [
            "superiortemporal",
            "middletemporal",
            "inferiortemporal",
            "bankssts",
            "fusiform",
            "transversetemporal",
            "entorhinal",
            "temporalpole",
            "parahippocampal",
        ],
        "occipital": ["lateraloccipital", "lingual", "cuneus", "pericalcarine"],
    }

    lobe_labels = {}
    for lobe, rois in lobe_rois.items():
        combined = None
        for roi in rois:
            for hemi in ["lh", "rh"]:
                name = f"{roi}-{hemi}"
                if name in label_dict:
                    combined = (
                        label_dict[name]
                        if combined is None
                        else combined + label_dict[name]
                    )
        if combined is not None:
            combined.name = lobe  # optional: set meaningful name
            lobe_labels[lobe] = combined
    return lobe_labels


# ---
# Helpers for source reconstruction
def extract_label_stcs(
    label: mne.Label, stcs: list[mne.SourceEstimate]
) -> list[mne.SourceEstimate]:
    """Extract source time courses within a given label."""
    return [stc.in_label(label) for stc in stcs]


# ---
# -------


def main():
    out_dir = Path("sources")
    out_dir.mkdir(exist_ok=True)
    cached_files = out_dir.glob("*.joblib")
    cached_subjects = set([path.stem for path in cached_files])
    for subject in tqdm(subjects, desc="Source reconstruction for all subjects."):
        if subject in cached_subjects:
            print(f"Subject {subject} already cached, skipping forward")
            continue
        print(f"Running reconstruction for subject {subject}")
        Xs, y = load_sources(subject)
        res = {"subject_id": subject, "Xs": Xs, "y": y}
        print("Saving")
        joblib.dump(res, out_dir.joinpath(f"{subject}.joblib"))
    print("DONE")


if __name__ == "__main__":
    main()
