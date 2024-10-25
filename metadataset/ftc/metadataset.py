import numpy as np
import pandas as pd
import torch
from pathlib import Path
from collections import defaultdict

from .ftc_args import ftc_args, ftctest_args, mini_ftc_args
from .hub import MODELS, DATASETS
from ..evaluator import Evaluator

class FTCMetadataset(Evaluator):

    metadataset_name = "ftc"

    def __init__(
        self,
        data_dir: str = None,
        seed: int = 42,
        split: str = "valid",
        metric_name: str = "error",
        meta_split_ids: tuple[tuple, tuple, tuple] = ((0, 1, 2), (3,), (4,)),
        data_version: str = "mini", 
        device: str = 'cpu',
        **kwargs
    ):
        self.seed = seed
        self.split = split
        self.metric_name = metric_name
        self.data_version = data_version
        self.models = MODELS
        
        super().__init__(
            data_dir=data_dir,
            meta_split_ids=meta_split_ids,
            seed=seed,
            split=split,
            metric_name=metric_name,
            device=device,
            data_version=data_version
        )

        if self.data_version == "mini":
            self.data_dir = Path(data_dir) / "mini"
        elif self.data_version == "extended":
            self.data_dir = Path(data_dir) / "ftc"
        else:
            raise ValueError("Data version is not valid.")
        
        self.files = ["times.csv",
                     "test_predictions.csv",
                     "val_predictions.csv"]
        self._load_data()
        self._initialize()

    def get_dataset_names(self):
        if self.data_version == "ftctest":
            return ["ag_news"]
        return DATASETS.copy()
    
    def _load_predictions(self, config_id):
        path = self.data_dir / config_id
        if path.exists():
            return [torch.FloatTensor(pd.read_csv(path / file, index_col=0).values) for file in self.files]
        else:
            return None

    def _preprocess_hps(self, hps):
        hps =  pd.get_dummies(
                        pd.DataFrame(hps)
                    )
        hps_names =  hps.columns
        hps = hps.fillna(0) \
                    .astype(float).values
        return torch.FloatTensor(hps), hps_names
    
    def export_failed_configs(self, args_path):

        with open(args_path+f'/failed_ftc_{self.data_version}.args', 'w') as file:
            # Write each sentence to the file
            for arg in self.failed_configs:
                arg += f" --finetuning_config_file failed_finetuning_args"
                file.write(arg + '\n')

    def _load_data(self):
        if self.data_version == "ftctest":
            args_dict, args_list = ftctest_args("ftctest")
        elif self.data_version == "mini":
            args_dict, args_list = mini_ftc_args("mini_ftc")
        elif self.data_version == "extended":
            args_dict, args_list = ftc_args("ftc")
        else:
            raise ValueError("No valid data version.")
        
        self.val_predictions = defaultdict(list)
        self.test_predictions = defaultdict(list)
        self.times = defaultdict(list)
        self.curves = defaultdict(list)
        self.all_hp_candidates = defaultdict(list)
        self.all_hp_names = defaultdict(list)
        self.row_hp_candidates = defaultdict(list)
        self.all_hp_candidates_ids = defaultdict(list)
        self.failed_hps = defaultdict(list)
        self.failed_configs = []
        self.val_targets = {}
        self.test_targets = {}

        for (config_id, config), bash_args in zip(args_dict.items(), args_list):
            data = self._load_predictions(config_id)
            dataset_name = config.pop("dataset_name")

            if data is not None:
                times, test_predictions, val_predictions = data

                if test_predictions.isnan().sum()>0 or \
                    val_predictions.isnan().sum()>0:
                    self.failed_hps[dataset_name].append(config)
                    self.failed_configs.append(bash_args)
                else:
                    self.times[dataset_name].append(times.T.unsqueeze(0))
                    self.val_predictions[dataset_name].append(val_predictions.unsqueeze(0))
                    self.test_predictions[dataset_name].append(test_predictions.unsqueeze(0))
                    self.row_hp_candidates[dataset_name].append(config)
            else:
                self.failed_hps[dataset_name].append(config)
                self.failed_configs.append(bash_args)

        for dataset_name in self.val_predictions.keys():
            self.val_targets[dataset_name] = torch.cat(self.val_predictions[dataset_name])[0,:,-1].long()
            self.test_targets[dataset_name] = torch.cat(self.test_predictions[dataset_name])[0,:,-1].long()
            self.val_predictions[dataset_name] = torch.cat(self.val_predictions[dataset_name])[...,:-1]
            self.test_predictions[dataset_name] = torch.cat(self.test_predictions[dataset_name])[...,:-1]
            self.times[dataset_name] = torch.cat(self.times[dataset_name])

            self.failed_hps[dataset_name] = pd.DataFrame(self.failed_hps[dataset_name])
            self.all_hp_candidates[dataset_name], self.all_hp_names[dataset_name] = self._preprocess_hps(self.row_hp_candidates[dataset_name])
            self.all_hp_candidates_ids[dataset_name] =  torch.arange(len(self.row_hp_candidates[dataset_name]))
    
    def get_features(self, ensembles):
        hp_candidates, hp_candidates_ids = self._get_hp_candidates_and_indices()
        return hp_candidates[torch.LongTensor(ensembles)]

    def set_state(self, dataset_name: str,
                    split: str = "valid"
                  ):

        self.split = split
        self.dataset_name = dataset_name
        super().set_state(dataset_name=dataset_name,
                          split=split)


    def _get_hp_candidates_and_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.all_hp_candidates[self.dataset_name], self.all_hp_candidates_ids[self.dataset_name]
    
    def get_targets(self) -> torch.Tensor:
        if self.split == "valid":
            return self.val_targets[self.dataset_name]
        elif self.split == "test":
            return self.test_targets[self.dataset_name]
        else:
            raise NameError("Split name is not specified.")

    def get_num_pipelines(self) -> int:
        return len(self.val_predictions[self.dataset_name])

    def get_num_classes(self) -> int:
        return self.val_predictions[self.dataset_name].shape[-1]

    def get_num_samples(self) -> int:
        if self.split == "valid":
            return self.val_targets[self.dataset_name].shape[-2]
        elif self.split == "test":
            return self.test_targets[self.dataset_name].shape[-2]
        else:
            raise NameError("Split name is not specified.")


    def _get_worst_and_best_performance(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, hp_candidates_ids = self._get_hp_candidates_and_indices()
        unique_pipelines = hp_candidates_ids.unsqueeze(0).T.tolist()
        _, metrics, _, _ = self.evaluate_ensembles(unique_pipelines)

        self.worst_performance = metrics.max().item()
        self.best_performance = metrics.min().item()
        return self.worst_performance, self.best_performance

    def get_time(self, ensembles: list[list[int]]) -> torch.Tensor:
        return self.times[self.dataset_name][torch.LongTensor(ensembles)]
    
    def get_predictions(self, ensembles: list):
        if self.split == "valid":
            pred = self.val_predictions[self.dataset_name]
        elif self.split == "test":
            pred = self.test_predictions[self.dataset_name]
        else:
            raise NameError("Split name is not specified.")
    
        pred = torch.nn.Softmax(dim=-1)(pred[torch.LongTensor(ensembles)])
        
        return pred
    