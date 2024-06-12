#how to save the predictino per dataset?
import os
os.environ["HF_DATASETS_CACHE"] = "/cephfs/workspace/v101be15-finetuning"
os.environ["HF_HOME"] = "/cephfs/workspace/v101be15-finetuning"
os.environ['TRANSFORMERS_CACHE'] = "/cephfs/workspace/v101be15-finetuning"

from transformers import  Trainer, TrainingArguments
from datasets import load_dataset
import argparse
import pandas as pd
from pathlib import Path
import argparse
import yaml

from utils import get_model_and_tokenizer_for_classification,  \
                    get_text_and_label_field, get_tokenize_function, \
                    get_predictions, get_num_labels

SEED = 42

if __name__ == "__main__":
    current_path = Path(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--finetuning_config_file", type=str, default="default_finetuning_args")
    args = parser.parse_args()

    print(args)
    experiment_name=args.experiment_name
    model_name = args.model_name
    dataset_name = args.dataset_name
    finetuning_config = args.finetuning_config_file
    lora_r = args.lora_r
    test_size = args.test_size

    with open(current_path / "config" / (finetuning_config+".yml"), 'r') as file:
        finetuning_args = yaml.safe_load(file)

    if lora_r is not None:
        finetuning_args["lora_args"]["r"] = lora_r
        
    # Load and tokenize the dataset
    train_dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")
    train_val_split = train_dataset.train_test_split(test_size=test_size)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    text_field, label_field = get_text_and_label_field(dataset_name)
    num_labels = get_num_labels(train_dataset, dataset_name, label_field)
    model, tokenizer = get_model_and_tokenizer_for_classification(model_name, num_labels=num_labels, lora_args=finetuning_args["lora_args"])
    tokenize_function = get_tokenize_function(tokenizer, text_field)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)


    if experiment_name.startswith("test"):
        train_dataset = train_dataset.select(range(1000))
        val_dataset = val_dataset.select(range(1000))

    training_args = TrainingArguments(
        **finetuning_args["training_args"]
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,

    )

    # Train and evaluate the model
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Making predictions
    val_predictions = get_predictions(model, tokenizer, val_dataset, text_field, label_field)
    test_predictions = get_predictions(model, tokenizer, test_dataset, text_field, label_field)

    experiment_path = current_path / "data" / experiment_name.split("_")[0] / experiment_name
    experiment_path.mkdir(exist_ok=True, parents=True)

    val_predictions.to_csv(experiment_path / "val_predictions.csv")    # Prepare DataLoader
    test_predictions.to_csv(experiment_path / "test_predictions.csv")    # Prepare DataLoader

    pd.DataFrame(trainer.state.log_history).to_csv(experiment_path / "curves.csv")