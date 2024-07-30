from metadataset.ftc.ftc_args import ftc_args, mini_ftc_args

if __name__ == "__main__":
    experiment_name = "mini_ftc"
    args_dict, args_list = mini_ftc_args(experiment_name=experiment_name)
    with open(f'bash_args/{experiment_name}.args', 'w') as file:
        # Write each sentence to the file
        for arg in args_list:
            file.write(arg + '\n')