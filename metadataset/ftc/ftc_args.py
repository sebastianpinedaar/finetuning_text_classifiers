from metadataset.ftc.hub import MODELS, DATASETS
experiment_name = "ftc"

def ftc_args(experiment_name):
    hps_dict = {"lora_r": [ 8, 16, 32, 64, 128],
                "learning_rate": [ 0.005, 0.001, 0.0005, 0.0001, 0.00001],
                "lora_dropout": [0., 0.1, 0.5]}
    args_list = []
    args_dict = {}
    config_id = 0
    for model in MODELS:
        for dataset in DATASETS:
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

