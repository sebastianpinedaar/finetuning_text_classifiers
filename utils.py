import os
import torch
import pandas as pd

os.environ["HF_DATASETS_CACHE"] = "/cephfs/workspace/v101be15-finetuning"
os.environ["HF_HOME"] = "/cephfs/workspace/v101be15-finetuning"
os.environ['TRANSFORMERS_CACHE'] = "/cephfs/workspace/v101be15-finetuning"

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, TaskType,get_peft_model


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

def get_tokenize_function(tokenizer, text_field):
    def tokenize_function(examples):
        return tokenizer(examples[text_field], padding='max_length', truncation=True)
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
    else:
        num_labels = dataset.features[label_field].num_classes

    return num_labels
