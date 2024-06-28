from metadataset.ftc.ftc_args import ftc_args

if __name__ == "__main__":
    experiment_name = "ftc"
    args_dict, args_list = ftc_args(experiment_name=experiment_name)
    with open(f'bash_args/{experiment_name}.args', 'w') as file:
        # Write each sentence to the file
        for arg in args_list:
            file.write(arg + '\n')