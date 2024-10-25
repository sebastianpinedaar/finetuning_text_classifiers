import os
workspace_path = "/anvme/workspace/v101be15-ftc_data"
os.environ["HF_DATASETS_CACHE"] = workspace_path
os.environ["HF_HOME"] = workspace_path
os.environ['TRANSFORMERS_CACHE'] = workspace_path

from transformers import  Trainer, TrainingArguments
import argparse
import pandas as pd
from pathlib import Path
import argparse
import yaml

from utils import TimeCallback
from utils import get_predictions
from utils import get_model_tokenizer_dataset

SEED = 42

if __name__ == "__main__":
    current_path = Path(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset_name", type=str, default="imdb")
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default="test")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--pct_train", type=float, default=1.)
    parser.add_argument("--finetuning_config_file", type=str, default="default_finetuning_args")
    args = parser.parse_args()

    print(args)
    experiment_name=args.experiment_name
    model_name = args.model_name
    dataset_name = args.dataset_name
    finetuning_config = args.finetuning_config_file
    lora_r = args.lora_r
    test_size = args.test_size
    max_length = args.max_length
    learning_rate = args.learning_rate
    lora_dropout = args.lora_dropout
    pct_train = args.pct_train

    is_test = experiment_name.startswith("test")

    with open(current_path / "config" / (finetuning_config+".yml"), 'r') as file:
        finetuning_args = yaml.safe_load(file)

    if lora_r is not None:
        finetuning_args["lora_args"]["r"] = lora_r

    if lora_dropout is not None:
        finetuning_args["lora_args"]["lora_dropout"] = lora_dropout
        
    if learning_rate is not None:
        finetuning_args["training_args"]["learning_rate"] = learning_rate

    time_callback = TimeCallback()

    training_args = TrainingArguments(
        **finetuning_args["training_args"]
    )

    (model, tokenizer, train_dataset, 
     val_dataset, test_dataset, dataset_info) = get_model_tokenizer_dataset(model_name=model_name,
                                                                  dataset_name=dataset_name,
                                                                  test_size=test_size,
                                                                  max_length=max_length,
                                                                  pct_train=pct_train,
                                                                  lora_args=finetuning_args["lora_args"])

    text_field = dataset_info["text_field"]
    label_field = dataset_info["label_field"]

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[time_callback]
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
    pd.DataFrame(time_callback.epoch_times).to_csv(experiment_path / "times.csv")
