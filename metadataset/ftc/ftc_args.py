from metadataset.ftc.hub import MODELS, DATASETS
experiment_name = "ftc"

def ftc_args(experiment_name):
    hps_dict = {"lora_r": [ 8, 16, 32, 64, 128],
                "learning_rate": [ 0.005, 0.001, 0.0005, 0.0001, 0.00001],
                }
    args_list = []
    args_dict = {}
    config_id = 0
    for model in MODELS:
        for dataset in DATASETS:
            for lora_r in hps_dict["lora_r"]:
                for lr in hps_dict["learning_rate"]:
                    args_list.append(
                        f"--model_name {model} " + \
                        f"--dataset_name {dataset} " + \
                        f"--lora_r {lora_r} " + \
                        f"--learning_rate {lr} " + \
                        f"--experiment_name {experiment_name}_{config_id}" 
                    )
                    temp_experiment_name = f"{experiment_name}_{config_id}"
                    config_id += 1
                    args_dict[temp_experiment_name]= {"model": model,
                                                        "dataset_name": dataset,
                                                        "lora_r": lora_r,
                                                        "learning_rate": lr,
                                                        }

    return args_dict, args_list

def mini_ftc_args(experiment_name):
    hps_dict = {"lora_r": [ 8, 16, 32, 64, 128],
                "learning_rate": [ 0.005, 0.001, 0.0005, 0.0001, 0.00001]
                }
    pct_train = 0.1
    args_list = []
    args_dict = {}
    config_id = 0
    for model in MODELS:
        for dataset in DATASETS:
            for lora_r in hps_dict["lora_r"]:
                for lr in hps_dict["learning_rate"]:
                    args_list.append(
                        f"--pct_train {pct_train} " + \
                        f"--model_name {model} " + \
                        f"--dataset_name {dataset} " + \
                        f"--lora_r {lora_r} " + \
                        f"--finetuning_config_file mini_finetuning_args  " + \
                        f"--learning_rate {lr} " + \
                        f"--experiment_name {experiment_name}_{config_id}" 
                    )
                    temp_experiment_name = f"{experiment_name}_{config_id}"
                    config_id += 1
                    args_dict[temp_experiment_name]= {"model": model,
                                                        "dataset_name": dataset,
                                                        "lora_r": lora_r,
                                                        "learning_rate": lr,
                                                        }

    return args_dict, args_list


def ftctest_args(experiment_name):
    hps_dict = {"lora_r": [ 8, 16, 32, 64, 128],
                "learning_rate": [ 0.005],
                "lora_dropout": [0.]}
    args_list = []
    args_dict = {}
    config_id = 0
    datasets = ["ag_news"]
    for model in MODELS:
        for dataset in datasets:
            for lora_r in hps_dict["lora_r"]:
                for lr in hps_dict["learning_rate"]:
                    #for lora_dropout in args_dict["lora_dropout"]:
                        args_list.append(
                            f"--model_name {model} " + \
                            f"--dataset_name {dataset} " + \
                            f"--lora_r {lora_r} " + \
                            #   f"--lora_dropout {lora_dropout} " + \
                            f"--learning_rate {lr} " + \
                            f"--experiment_name {experiment_name}_{config_id}" 
                        )
                        temp_experiment_name = f"{experiment_name}_{config_id}"
                        config_id += 1
                        args_dict[temp_experiment_name]= {"model": model,
                                                            "dataset_name": dataset,
                                                            "lora_r": lora_r,
                                                            "learning_rate": lr,
                                                            }
    
    return args_dict, args_list

