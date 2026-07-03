"""Build evaluation-ready and representative mapping tables for NeuroVault datasets."""

import argparse
import json
import os
import os.path as op
from pathlib import Path

import pandas as pd


IBC_REPRESENTATIVE_RULES = {
    "archi_emotional": "nv40641-sub-01-ses-07-expression-intention-gender",
    "archi_social": "nv40616-sub-01-ses-00-triangle-mental-random",
    "archi_spatial": "nv40602-sub-01-ses-00-saccades",
    "archi_standard": "nv40564-sub-01-ses-00-computation-sentences",
    "hcp_emotion": "nv40035-sub-01-ses-03-face-shape",
    "hcp_gambling": "nv40020-sub-01-ses-03-reward",
    "hcp_language": "nv40012-sub-01-ses-03-story-math",
    "hcp_motor": "nv40023-sub-01-ses-03-left-hand-avg",
    "hcp_relational": "nv40038-sub-01-ses-04-relational-match",
    "hcp_social": "nv40016-sub-01-ses-04-mental-random",
    "hcp_wm": "nv40041-sub-01-ses-04-2back-0back",
    "language_nsp": "nv40654-sub-01-ses-05-complex-simple",
}

CNP_REPRESENTATIVE_RULES = {
    "BART": "nv49974-BART-Accept",
    "PAMRET": "nv49999-PAMRET-All",
    "SCAP": "nv50048-SCAP-All",
    "STOPSIGNAL": "nv50088-STOPSIGNAL-Go-StopSuccess",
    "TASKSWITCH": "nv50104-TASKSWITCH-ALL",
}


def _load_full_task_set(task_snapshot_fn):
    with open(task_snapshot_fn, "r") as file_obj:
        tasks = json.load(file_obj)
    return {task["name"].strip() for task in tasks if task.get("name")}


def _materialize_representative_dir(mapping_df, source_dir, destination_dir):
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for row in mapping_df.itertuples(index=False):
        source_path = Path(row.local_path)
        link_path = destination_dir / source_path.name
        if link_path.exists() or link_path.is_symlink():
            continue
        rel_target = os.path.relpath(source_path, start=destination_dir)
        link_path.symlink_to(rel_target)


def _build_mapping_tables(data_dir, dataset_name, representative_rules):
    dataset_dir = Path(data_dir) / dataset_name
    mapping_df = pd.read_csv(dataset_dir / "mapping.csv")
    mapping_df["task_name"] = mapping_df["task_name"].fillna("").astype(str).str.strip()

    reduced_df = pd.read_csv(Path(data_dir) / "cognitive_atlas" / "reduced_tasks.csv")
    reduced_tasks = {task.strip() for task in reduced_df["task"].tolist()}
    full_tasks = _load_full_task_set(Path(data_dir) / "cognitive_atlas" / "task_snapshot-02-19-25.json")

    mapping_df["in_reduced_ontology"] = mapping_df["task_name"].isin(reduced_tasks)
    mapping_df["in_full_ontology"] = mapping_df["task_name"].isin(full_tasks)

    reduced_mapping_df = mapping_df.loc[mapping_df["in_reduced_ontology"]].copy()
    full_mapping_df = mapping_df.loc[mapping_df["in_full_ontology"]].copy()

    representative_rows = []
    for task_family, prediction_label in representative_rules.items():
        matched = mapping_df.loc[mapping_df["prediction_label"] == prediction_label]
        if matched.empty:
            raise KeyError(f"Could not find representative label {prediction_label!r} for {dataset_name}:{task_family}.")
        representative_rows.append(matched.iloc[0].to_dict())
    representative_df = pd.DataFrame(representative_rows).sort_values(by="task_family").reset_index(drop=True)

    reduced_representative_df = representative_df.loc[representative_df["in_reduced_ontology"]].copy()
    full_representative_df = representative_df.loc[representative_df["in_full_ontology"]].copy()

    reduced_mapping_df.to_csv(dataset_dir / "mapping_reduced.csv", index=False)
    full_mapping_df.to_csv(dataset_dir / "mapping_full.csv", index=False)
    representative_df.to_csv(dataset_dir / "mapping_representative.csv", index=False)
    reduced_representative_df.to_csv(dataset_dir / "mapping_representative_reduced.csv", index=False)
    full_representative_df.to_csv(dataset_dir / "mapping_representative_full.csv", index=False)

    representative_dir = Path(data_dir) / f"{dataset_name}_representative"
    _materialize_representative_dir(representative_df, dataset_dir, representative_dir)

    return {
        "mapping": mapping_df,
        "reduced_mapping": reduced_mapping_df,
        "full_mapping": full_mapping_df,
        "representative_mapping": representative_df,
        "reduced_representative_mapping": reduced_representative_df,
        "full_representative_mapping": full_representative_df,
    }


def _get_parser():
    parser = argparse.ArgumentParser(description="Build evaluation-ready dataset mapping tables.")
    parser.add_argument(
        "--project_dir",
        dest="project_dir",
        default=op.abspath(op.join(op.dirname(__file__), "..")),
        help="Path to the repository root.",
    )
    return parser


def main(project_dir):
    data_dir = op.join(op.abspath(project_dir), "data")
    _build_mapping_tables(data_dir, "ibc", IBC_REPRESENTATIVE_RULES)
    _build_mapping_tables(data_dir, "cnp", CNP_REPRESENTATIVE_RULES)


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


if __name__ == "__main__":
    _main()
