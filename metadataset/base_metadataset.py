from __future__ import annotations

from abc import abstractmethod

import numpy as np
import torch


META_SPLITS = {
    0: [(0, 1, 2), (3,), (4,)],
    1: [(1, 2, 3), (4,), (0,)],
    2: [(2, 3, 4), (0,), (1,)],
    3: [(3, 4, 0), (1,), (2,)],
    4: [(4, 0, 1), (2,), (3,)],
}


class BaseMetaDataset:
    metadataset_name = "base"
    def __init__(
        self,
        data_dir: str,
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "nll",
        data_version: str = None
    ):
        """Initialize the BaseMetaDataset.

        Args:
            data_dir (str): Path to the directory containing the meta-dataset.
            data_pct (tuple(float, float, float ), optional):
                Percentage of data to use for meta-train, meta-val, and meta-test.
                Defaults to (0.6, 0.2, 0.2).
            seed (int, optional): Random seed. Defaults to 42.
            split (str, optional): Dataset split name. Defaults to "valid".
            metric_name (str, optional): Name of the metric. Defaults to "nll".

        Attributes:
            data_dir (str): Directory path for the dataset.
            dataset_names (list): List of dataset names present in the meta-dataset.
            split_indices (dict): Dictionary containing splits for meta-training,
                                   meta-validation, and meta-testing.
            seed (int): Random seed.
            split (str): Dataset split name.
            metric_name (str): Name of the metric.
            meta_splits (dict): Dictionary containing meta train, val, and test splits.
            meta_split_ids (tuple): Tuple containing meta train, val, and test split ids.
            logger (logging.Logger): Logger.

        """

        self.data_dir = data_dir
        self.seed = seed
        self.split = split
        self.metric_name = metric_name
        self.meta_split_ids = meta_split_ids
        self.meta_splits: dict[str, list[str]] = {}
        self.data_version = data_version
        
        self.feature_dim: int | None = None

        # To initialize call _initialize() in the child class
        self.dataset_names: list[str] = []

        # To initialize call set_dataset(dataset_name) in the child class
        self.dataset_name: str
        self.hp_candidates: torch.Tensor
        self.hp_candidates_ids: torch.Tensor
        self.best_performance: torch.Tensor | None = None
        self.worst_performance: torch.Tensor | None = None


    @abstractmethod
    def get_dataset_names(self) -> list[str]:
        """Fetch the dataset names present in the meta-dataset.

        Returns:
            list: List of dataset names present in the meta-dataset.
        """

        raise NotImplementedError

    @abstractmethod
    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch hyperparameter candidates for a given dataset.

        Args:
            dataset_name (str): Name of the dataset for which hyperparameters are required.
        Returns:
            tuple[torch.Tensor, torch.Tensor]:
            A tuple of tensors, namely:
                - hp_candidates: tensor [N, F]
                - hp_candidates_ids: tensor [N]


            where N is the number of possible pipelines evaluated for a specific dataset
            and F is the number of features per pipeline. The hp_candidates_ids tensor
            contains the pipeline ids for the corresponding hyperparameter candidates.

        Raises:
            NotImplementedError: This method should be overridden by the child class.

        """

        raise NotImplementedError

    @abstractmethod
    def _get_worst_and_best_performance(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch the worst and best performance for a given dataset.


        Returns:
            tuple[torch.Tensor, torch.Tensor]:
            A tuple of tensors, namely:
                - worst_performance
                - best_performance

        """

        raise NotImplementedError

    @abstractmethod
    def evaluate_ensembles(
        self,
        ensembles: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given ensemble configurations.

        Args:
            ensembles (list[list[int]]): Ensemble configuration.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            A tuple of tensors, namely:
                - pipeline_hps: tensor [B, N, F]
                - metric: tensor [B]
                - metric_per_pipeline: tensor [B, N]
                - time_per_pipeline: tensor [B, N]

        Raises:
            NotImplementedError: This method should be overridden by the child class.

        """

        raise NotImplementedError

    @abstractmethod
    def get_num_samples(self) -> int:
        """
        Returns the number of samples for the current loaded dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_classes(self) -> int:
        """
        Returns the number of classes for the current loaded dataset.
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_pipelines(self) -> int:
        """
        Returns the number of classes for the current loaded dataset.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_predictions(self, ensembles: list[list[int]]) -> torch.Tensor:

        """
        Returns the ensemble predictions for every sample in the active dataset.

        B: number of ensembles
        N: number of models per ensemble
        M: number of samplers per ensemble
        C: number of classes per ensemble

        Args:
            ensembles: List of list with the base model index to evaluate: [B, N]

        Returns:
            prediction: torch tensor with the probabilistic prediction per class: [B, N, M, C]    
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_targets(self) -> torch.Tensor:
        """
        Returns the target associated to every sample in the active dataset.

        M: number of samplers per ensemble
        Returns:
            target: torch tensor with the target per sample: [M]    
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_time(self, ensembles: list[list[int]]) -> torch.Tensor:
        """
        Returns the target associated to every sample in the active dataset.

        B: number of ensembles
        N: number of models per ensemble
        Args:
            ensembles: List of list with the base model index to evaluate: [B, N]

        Returns:
            time: torch tensor with the time per pipeline and ensemble: [B, N]    
        """
        raise NotImplementedError
    
    def _initialize(self):
        """Initialize the meta-dataset. This method should be called in the child class."""
        self.dataset_names = self.get_dataset_names()
        self.meta_splits = self._get_meta_splits()

    def _get_meta_splits(self) -> dict[str, list[str]]:
        """Internal method to get meta splits for datasets.

        Args:
            data_pct (tuple(tuple(float), float, float ), optional):
                ID of the cross validation partition assigned to meta-train, meta-val and meta-test.
                The dfault assumes 5-fold cross (meta-) validation.
        Returns:
            dict[str, list[str]]: Dictionary containing meta train, val, and test splits.

        """
        rnd_gen = np.random.default_rng(self.seed)
        dataset_names = self.dataset_names.copy()
        rnd_gen.shuffle(dataset_names)

        meta_train_splits, meta_val_splits, meta_test_splits = self.meta_split_ids

        meta_splits: dict[str, list[str]] = {
            "meta-train": [],
            "meta-valid": [],
            "meta-test": [],
        }
        num_splits = len(meta_train_splits) + len(meta_test_splits) + len(meta_val_splits)
        for i, dataset in enumerate(dataset_names):
            split_id = i % num_splits
            if split_id in meta_train_splits:
                meta_splits["meta-train"].append(dataset)
            elif split_id in meta_test_splits:
                meta_splits["meta-test"].append(dataset)
            elif split_id in meta_val_splits:
                meta_splits["meta-valid"].append(dataset)
            else:
                raise ValueError("Dataset not assigned to any split")
        return meta_splits

    def set_state(self, dataset_name: str,
                  split: str | None = None):
        """
        Set the dataset to be used for training and evaluation.
        This method should be called before sampling.

            Args:
                dataset_name (str): Name of the dataset.

        """

        self.dataset_name = dataset_name
        self.split = split
        self.hp_candidates, self.hp_candidates_ids = self._get_hp_candidates_and_indices()
        (
            self.worst_performance,
            self.best_performance,
        ) = self._get_worst_and_best_performance()

    def compute_normalized_score(self, score: torch.Tensor) -> torch.Tensor:
        """Compute the normalized score.

        Args:
            score (torch.Tensor): The score tensor.

        Returns:
            torch.Tensor: The normalized score tensor.
        """
        if self.best_performance is not None and self.worst_performance is not None:
            score = (score - self.best_performance) / (
                self.worst_performance - self.best_performance
            )

        return score
    
    def normalize_performance(self, performance: float,
                              best_reference_performance: float = None,
                              worst_reference_performance: float = None):

        if best_reference_performance is None:
            if self.best_performance is None:
                best_reference_performance = 0.
            else:
                best_reference_performance = self.best_performance


        if worst_reference_performance is None:
            if self.worst_performance is None:
                worst_reference_performance = 1.
            else:
                worst_reference_performance = self.worst_performance


        if best_reference_performance != worst_reference_performance:
            normalized_value = (performance-best_reference_performance)/(worst_reference_performance-best_reference_performance)
        else:
            normalized_value = performance
            
        return normalized_value
    

    def get_logits_from_probabilities(self, probabilities: torch.Tensor) -> torch.Tensor:
        """
        Get the logits given the probabilities.

        Args:
            probabilities (torch.Tensor): probability Tensor of shape (num_ensembles, num_pipelines, num_samples, num_classes).

        Returns:
            logits (torch.Tensor): probability Tensor of shape (num_ensembles, num_pipelines, num_samples, num_classes)

        """
        log_p = torch.log(probabilities + 10e-8)
        C = -log_p.mean(-1)
        logits = log_p + C.unsqueeze(-1)
        return logits

    
    def recommend_pipelines(self, num_pipelines: int) -> list[int]:
        return np.random.randint(0, self.get_num_pipelines(), num_pipelines)