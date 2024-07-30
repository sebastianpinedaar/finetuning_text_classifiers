import os
import torch
import pandas as pd

os.environ["HF_DATASETS_CACHE"] = "/cephfs/workspace/v101be15-finetuning"
os.environ["HF_HOME"] = "/cephfs/workspace/v101be15-finetuning"
os.environ['TRANSFORMERS_CACHE'] = "/cephfs/workspace/v101be15-finetuning"

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType,get_peft_model
from transformers import TrainerCallback
import time

class TimeCallback(TrainerCallback):
    def __init__(self):
        self.epoch_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        print(f"Epoch {state.epoch} took {epoch_time:.2f} seconds")


def get_extra_lora_args(model_name, ):
    extra_args_for_lora = {}

    if model_name == "albert-large-v2":
        extra_args_for_lora["target_modules"] = ["query", "value", "key", "ffn"]

    return extra_args_for_lora

def get_text_and_label_field(dataset_name):
    text_field = "text"
    label_field = "label"

    if dataset_name == "stanfordnlp/sst2":
        text_field = "sentence"

    if dataset_name == "community-datasets/yahoo_answers_topics":
        text_field = "question_content"
        label_field = "topic"

    if dataset_name == "dbpedia_14":
        text_field = "content"
        label_field = "label"

    return text_field, label_field

def get_tokenize_function(tokenizer, text_field, max_length=None):
    kwargs = {}
    if max_length is not None:
        kwargs = {"max_length" : max_length}

    def tokenize_function(examples):
        return tokenizer(examples[text_field],
                          padding='max_length', 
                          truncation=True,
                          **kwargs)
    return tokenize_function

def get_model_and_tokenizer_for_classification(model_name, num_labels, lora_args):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    if model_name =="gpt2-medium" or model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id


    lora_args.update(get_extra_lora_args(model_name))
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        #inference_mode=False, 
        **lora_args
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    return model, tokenizer


def get_predictions(model, tokenizer, eval_dataset, text_field, label_field):
    outputs = []
    labels = []
    with torch.no_grad():
        for sample in eval_dataset:
            text = sample.pop(text_field)
            label = sample.pop(label_field)
            inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to("cuda")
            temp_outputs = model(**inputs)
            outputs.append(temp_outputs.logits.cpu().numpy()[0].tolist())
            labels.append(label)

    df = pd.DataFrame(outputs)
    df["label"] = labels

    return df

def get_num_labels(dataset, dataset_name, label_field):

    if dataset_name == "mteb/tweet_sentiment_extraction":
        num_labels = 3
    elif dataset_name == "SetFit/mnli":
        num_labels = 3
    else:
        num_labels = dataset.features[label_field].num_classes

    return num_labels

def combine_texts(example):
    text1 = example["text1"]
    text2 = example["text2"]
    example["text"] = f"Text1: {text1}, text2: {text2}"
    return example

def get_model_tokenizer_dataset(model_name, dataset_name, 
                                test_size=0.1, 
                                max_length=512, 
                                pct_train=1., 
                                seed=42,
                                lora_args={}):

    train_dataset = load_dataset(dataset_name, split="train")
    test_dataset = load_dataset(dataset_name, split="test")
    train_val_split = train_dataset.train_test_split(test_size=test_size,
                                                    seed=seed)
    train_dataset = train_val_split["train"]
    val_dataset = train_val_split["test"]

    if dataset_name in ["SetFit/mnli", "stanfordnlp/sst2"]:
        #This datasets have hidden labels in the test set
        train_val_split = train_dataset.train_test_split(test_size=test_size,
                                                         seed=seed)
        train_dataset = train_val_split["train"]
        test_dataset = train_val_split["test"]       
    
    if dataset_name == "SetFit/mnli":
        train_dataset = train_dataset.map(combine_texts)
        val_dataset = val_dataset.map(combine_texts)
        test_dataset = test_dataset.map(combine_texts)

    text_field, label_field = get_text_and_label_field(dataset_name)
    num_labels = get_num_labels(train_dataset, dataset_name, label_field)
    model, tokenizer = get_model_and_tokenizer_for_classification(model_name, num_labels=num_labels, lora_args=lora_args)
    tokenize_function = get_tokenize_function(tokenizer, text_field, max_length=max_length)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    if pct_train < 1.:
        train_size = pct_train*len(train_dataset)
        train_dataset = train_dataset.shuffle(seed=seed).select(range(int(train_size)))

    dataset_info = {
        "text_field" : text_field,
        "label_field" : label_field
    }

    return model, tokenizer, train_dataset, val_dataset, test_dataset, dataset_info