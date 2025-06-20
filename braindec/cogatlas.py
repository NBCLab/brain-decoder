import json

import numpy as np
import pandas as pd
import requests
from nimare import extract

COGATLAS_URLS = {
    "task": "https://www.cognitiveatlas.org/api/v-alpha/task",
    "concept": "https://www.cognitiveatlas.org/api/v-alpha/concept",
}

CLASSES_MAPPING = {
    "ctp_C1": "Perception",
    "ctp_C2": "Attention",
    "ctp_C3": "Reasoning and decision making",
    "ctp_C4": "Executive cognitive control",
    "ctp_C5": "Learning and memory",
    "ctp_C6": "Language",
    "ctp_C7": "Action",
    "ctp_C8": "Emotion",
    "ctp_C9": "Social function",
    "ctp_C10": "Motivation",
}


def _get_cogatlas_dict(url):
    try:
        # Send a GET request to the API
        response = requests.get(url)

        # Raise an exception for bad responses
        response.raise_for_status()

        # Parse the JSON response into a Python dictionary
        return response.json()

    except requests.RequestException as e:
        print(f"Error retrieving tasks: {e}")
        return None


def _get_concepts_to_tasks(relationships_df, concept_to_task=None):
    if concept_to_task is not None:
        concepts, tasks = zip(*concept_to_task.items())
        n_new_rel = len(concepts)
        new_relationships = pd.DataFrame(
            {"input": concepts, "output": tasks, "rel_type": ["measuredBy"] * n_new_rel}
        )
        relationships_df = pd.concat([relationships_df, new_relationships], ignore_index=True)
        relationships_df = relationships_df.drop_duplicates()

    concepts_to_tasks_df = relationships_df.loc[relationships_df["rel_type"] == "measuredBy"]
    concepts_to_tasks_df = concepts_to_tasks_df.drop(columns=["rel_type"])
    concepts_to_tasks_df.columns = ["id", "measuredBy"]
    return concepts_to_tasks_df


class CognitiveAtlas:
    def __init__(
        self,
        data_dir=None,
        task_snapshot=None,
        concept_snapshot=None,
        concept_to_task=None,
        concept_to_process=None,
        reduced_tasks=None,
    ):
        if task_snapshot is None:
            self.task = _get_cogatlas_dict(COGATLAS_URLS["task"])
        else:
            with open(task_snapshot, "r") as file:
                self.task = json.load(file)

        if concept_snapshot is None:
            self.concept = _get_cogatlas_dict(COGATLAS_URLS["concept"])
        else:
            with open(concept_snapshot, "r") as file:
                self.concept = json.load(file)

        # Convert dicts to DataFrame
        self.task_df = pd.DataFrame(self.task)
        self.task_df = self.task_df.replace("", np.nan)
        self.task_df = self.task_df.dropna(subset=["name", "definition_text"])

        if reduced_tasks is not None:
            task_to_keep = reduced_tasks["task"].to_list()
            self.task_df = self.task_df.loc[self.task_df["name"].isin(task_to_keep)]

            """
            # Drops tasks with short definitions
            self.task_df = self.task_df.loc[self.task_df["definition_text"].str.len() > 90]
            reduced_tasks = reduced_tasks.loc[
                reduced_tasks["task"].isin(self.task_df["name"])
            ].reset_index(drop=True)
            """

        self.task_ids = self.task_df["id"].to_list()
        self.task_names = self.task_df["name"].to_list()
        self.task_definitions = self.task_df["definition_text"].to_list()

        self.concept_df = pd.DataFrame(self.concept)
        self.concept_df = self.concept_df.replace("", np.nan)
        self.concept_df = self.concept_df.dropna(subset=["name", "definition_text"])
        self.concept_ids = self.concept_df["id"].to_list()
        self.concept_names = self.concept_df["name"].to_list()
        self.concept_definitions = self.concept_df["definition_text"].to_list()

        self.process_ids = list(CLASSES_MAPPING.keys())
        self.process_names = list(CLASSES_MAPPING.values())

        if concept_to_process is not None:
            mask = self.concept_df["name"].isin(concept_to_process.keys())
            self.concept_df.loc[mask, "id_concept_class"] = self.concept_df.loc[mask, "name"].map(
                concept_to_process
            )

        # Add Cognitive Process name to concept dataframe
        cog_proc_mapping_df = pd.DataFrame(
            CLASSES_MAPPING.items(),
            columns=["id_concept_class", "cognitive_process"],
        )

        self.concept_df = pd.merge(
            self.concept_df,
            cog_proc_mapping_df,
            how="left",
            on="id_concept_class",
        )

        if reduced_tasks is not None:
            concepts_to_tasks_df = self._get_concepts_to_tasks_red(reduced_tasks)
        else:
            cogatlas = extract.download_cognitive_atlas(data_dir=data_dir, overwrite=False)
            relationships_df = pd.read_csv(cogatlas["relationships"])

            concepts_to_tasks_df = _get_concepts_to_tasks(
                relationships_df,
                concept_to_task=concept_to_task,
            )

        self.concept_to_task_idxs = []
        for concept in self.concept_ids:
            sel_df = concepts_to_tasks_df.loc[concepts_to_tasks_df["id"] == concept]
            if len(sel_df) == 0:
                # append empty numpy array to
                self.concept_to_task_idxs.append(np.array([]))
                continue

            sel_tasks = sel_df["measuredBy"].values
            indices = np.where(np.in1d(self.task_df["id"].values, sel_tasks))[0]

            self.concept_to_task_idxs.append(indices)

        self.process_to_concept_idxs = []
        for process in self.process_names:
            sel_df = self.concept_df.loc[self.concept_df["cognitive_process"] == process]
            indices = np.where(np.in1d(self.concept_df["id"].values, sel_df["id"].values))[0]

            self.process_to_concept_idxs.append(indices)

        self.task_to_concept_idxs = []
        for task in self.task_df["id"]:
            sel_df = concepts_to_tasks_df.loc[concepts_to_tasks_df["measuredBy"] == task]
            if len(sel_df) == 0:
                self.task_to_concept_idxs.append(np.array([]))
                continue

            sel_concepts = sel_df["id"].values
            indices = np.where(np.in1d(self.concept_df["id"].values, sel_concepts))[0]

            self.task_to_concept_idxs.append(indices)

        self.task_to_process_idxs = []
        for task in self.task_df["id"]:
            sel_concepts = concepts_to_tasks_df.loc[concepts_to_tasks_df["measuredBy"] == task][
                "id"
            ].values
            if task == "trm_550b54a8b30f4":
                print(sel_concepts)

            sel_df = self.concept_df.loc[self.concept_df["id"].isin(sel_concepts)]
            indices = np.where(np.in1d(self.process_ids, sel_df["id_concept_class"].values))[0]

            self.task_to_process_idxs.append(indices)

    def get_concept_id_from_name(self, names):
        if isinstance(names, str):
            return self.concept_ids[self.concept_names.index(names)]

        return [self.concept_ids[self.concept_names.index(name)] for name in names]

    def get_task_id_from_name(self, names):
        if isinstance(names, str):
            return self.task_ids[self.task_names.index(names)]

        return [self.task_ids[self.task_names.index(name)] for name in names]

    def get_task_idx_from_names(self, names):
        if isinstance(names, str):
            return np.where(np.in1d(self.task_names, names))[0][0]

        return [np.where(np.in1d(self.task_names, task_name))[0][0] for task_name in names]

    def get_concept_idx_from_names(self, names):
        if isinstance(names, str):
            return np.where(np.in1d(self.concept_names, names))[0][0]

        return [
            np.where(np.in1d(self.concept_names, concept_name))[0][0] for concept_name in names
        ]

    def get_process_idx_from_names(self, names):
        if isinstance(names, str):
            return np.where(np.in1d(self.process_names, names))[0][0]

        return [
            np.where(np.in1d(self.process_names, process_name))[0][0] for process_name in names
        ]

    def get_task_names_from_idx(self, task_idx):
        return np.array(self.task_names)[task_idx]

    def get_concept_names_from_idx(self, concept_idx):
        return np.array(self.concept_names)[concept_idx]

    def get_process_names_from_idx(self, process_idx):
        return np.array(self.process_names)[process_idx]

    def get_task_idx_from_concept_idx(self, concept_idx):
        return self.concept_to_task_idxs[concept_idx]

    def get_concept_idx_from_task_idx(self, task_idx):
        return self.task_to_concept_idxs[task_idx]

    def get_concept_idx_from_process_idx(self, process_idx):
        return self.process_to_concept_idxs[process_idx]

    def _get_concepts_to_tasks_red(self, reduced_tasks):
        concepts, tasks = [], []
        for _, row in reduced_tasks.iterrows():
            task = row["task"]
            task = self.get_task_id_from_name(task)
            for col in ["concept_1", "concept_2", "concept_3"]:
                concept = row[col]
                if pd.isna(concept):
                    continue
                concept = self.get_concept_id_from_name(concept)

                concepts.append(concept)
                tasks.append(task)

        # Create the new dataframe
        return pd.DataFrame(
            {
                "id": concepts,
                "measuredBy": tasks,
            }
        )
