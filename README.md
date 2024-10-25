# Finetuning Text Classifiers

A simple code-base for finetuning Large Language Models for classification tasks.

## Install

```bash
conda -n ftc python=3.10
conda activate ftc
pip install -r requirements.txt
```

## Run finetuning pipeline

You can generate data for a specific configuration by running the finetuning script `finetune.py`. Remember to change the path to save the HuggingFace data and models , i.e. variable *workspace_path*.

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

We generate two metadataset versions: **mini** containing 10% of the data, and **extended**, containing the full data. To generate the metadataset in **mini** version, you can run the scripts with all the arguments in `bash_args/mini_ftc.args`. For the **extended** version use the arguments listed in `bash_args/ftc.args`. Some configurations have failed due to memory limits. We listed them in the files `bash_args/failed_ftc_extended.args` and `bash_args/failed_ftc_mini.args`.

## Download Metadataset

You can skip the generation of the metadataset by downloading them.

```bash
mkdir data
cd data
wget https://rewind.tf.uni-freiburg.de/index.php/s/QGjKFQf42FeZCag/download/ftc.zip
wget https://rewind.tf.uni-freiburg.de/index.php/s/G2KTnSYcC42gzwK/download/mini.zip
```
## Evaluate Metadataset

After downloading the data, you can simulate ensembles by using an *FTCMetadataset* object, as shown below.


```python
from metadataset.ftc.metadataset import FTCMetadataset
from pathlib import Path

data_dir = Path(os.path.dirname(os.path.abspath(__file__))) /  "data" 
data_version = "extended"
metadataset = FTCMetadataset(data_dir=str(data_dir), metric_name="error",
                                data_version=data_version)
metadataset.set_state(dataset_name=metadataset.get_dataset_names()[1]
                        )
output = metadataset.evaluate_ensembles([[1,2],[3,4]])
ensemble_score = output[1]
per_pipeline_score = output[2]
```

## Citation

If this repo is helpful to you, consider citing us.

```bibtex
@inproceedings{
arango2024ensembling,
title={Ensembling Finetuned Language Models for Text Classification},
author={Sebastian Pineda Arango and Maciej Janowski and Lennart Purucker and Arber Zela and Frank Hutter and Josif Grabocka},
booktitle={NeurIPS 2024 Workshop on Fine-Tuning in Modern Machine Learning: Principles and Scalability},
year={2024},
url={https://openreview.net/forum?id=oeUE4Of8e8}
}
```
