from hub import models, datasets

args_dict = {"lora_r": [128]}
args_list = []
experiment_name = "test"
config_id = 0
for model in models:
    for dataset in datasets:
        for lora_r in args_dict["lora_r"]:
            args_list.append(
                f"--model_name {model} " + \
                f"--dataset_name {dataset} " + \
                f"--lora_r {lora_r} " + \
                f"--experiment_name {experiment_name}_{config_id}" 
            )
            config_id += 1

with open(f'bash_args/{experiment_name}.args', 'w') as file:
    # Write each sentence to the file
    for arg in args_list:
        file.write(arg + '\n')