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

MISSING_CONCEPTS_MAPPING = {
    "anticipation": "ctp_C7",
    "arousal": "ctp_C8",
    "arithmetic processing": "ctp_C3",
    "concept": "ctp_C3",
    "effort": "ctp_C7",
    "creative thinking": "ctp_C3",
    "emotion regulation": "ctp_C8",
    "emotional enhancement": "ctp_C8",
    "guilt": "ctp_C8",
    "imagination": "ctp_C3",
    "phonological processing": "ctp_C6",
    "semantic categorization": "ctp_C6",
    "story comprehension": "ctp_C6",
    "visual orientation": "ctp_C4",
    "strategy": "ctp_C3",
    "thought": "ctp_C3",
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


class CognitiveAtlas:
    def __init__(self, data_dir=None, task_snapshot=None, concept_snapshot=None):
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
        self.task_names = self.task_df["name"].to_list()
        self.task_definitions = self.task_df["definition_text"].to_list()

        self.concept_df = pd.DataFrame(self.concept)
        self.concept_df = self.concept_df.replace("", np.nan)
        self.concept_df = self.concept_df.dropna(subset=["name", "definition_text"])
        self.concept_names = self.concept_df["name"].to_list()
        self.concept_definitions = self.concept_df["definition_text"].to_list()

        mask = self.concept_df["name"].isin(MISSING_CONCEPTS_MAPPING.keys())
        self.concept_df.loc[mask, "id_concept_class"] = self.concept_df.loc[mask, "name"].map(
            MISSING_CONCEPTS_MAPPING
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

        cogatlas = extract.download_cognitive_atlas(data_dir=data_dir, overwrite=False)
        relationships_df = pd.read_csv(cogatlas["relationships"])

        concepts_to_tasks_df = relationships_df.loc[relationships_df["rel_type"] == "measuredBy"]
        concepts_to_tasks_df = concepts_to_tasks_df.drop(columns=["rel_type"])
        concepts_to_tasks_df.columns = ["id", "measuredBy"]

        self.concept_to_task_idxs = []
        for concept in self.concept_df["id"]:
            sel_df = concepts_to_tasks_df.loc[concepts_to_tasks_df["id"] == concept]
            if len(sel_df) == 0:
                # append empty numpy array to
                self.concept_to_task_idxs.append(np.array([]))
                continue

            sel_tasks = sel_df["measuredBy"].values
            indices = np.where(np.in1d(self.task_df["id"].values, sel_tasks))[0]

            self.concept_to_task_idxs.append(indices)

        self.process_names = list(CLASSES_MAPPING.values())
        self.process_to_concept_idxs = []
        for process in self.process_names:
            sel_df = self.concept_df.loc[self.concept_df["cognitive_process"] == process]
            indices = np.where(np.in1d(self.concept_df["id"].values, sel_df["id"].values))[0]

            self.process_to_concept_idxs.append(indices)
