# Finetuning Text Classifiers

A simple code-base for finetuning Large Language Models for classification tasks.

## Install

```bash
conda -n quick_tune python=3.10
conda activate ftc
pip install -r requirements.txt
```

## Run finetuning pipeline

You can generate data for a specific configuration by running the finetuning script `finetune.py`.

```bash
python finetune.py --model_name gpt2 --dataset_name imdb --lora_r 8 --learning_rate 0.005 --experiment_name ftc_0
```

Training and LoRA arguments can be passed as a config file as in `config/default_finetuning_args.yml`.


## FTC Metadataset


We create a search space with the following hyperparmaeters.

* Datasets: `imdb, mteb/tweet_sentiment_extraction, ag_news, dbpedia_14, stanfordnlp/sst2, SetFit/mnli`.
* Models: `gpt2, bert-large-uncased, albert-large-v2, facebook/bart-large, google-t5/t5-large`.
* Learning rate: ` 0.005, 0.001, 0.0005, 0.0001, 0.00001`.
* Lora R: `8, 16, 32, 64, 128`.

To generate the metadataset, you can run the scripts with all the arguments in `bash_args/ftc.args`.
