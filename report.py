from pathlib import Path
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sys.path.append("..")
from metadataset.ftc.metadataset import FTCMetadataset
from metadataset.ftc.hub import MODELS

from matplotlib import rc


data_dir = "data" 
data_version = "mini"
fontsize = 30
#data_version = "extended"

metadataset = FTCMetadataset(data_dir=str(data_dir), 
                             metric_name="error",
                             data_version=data_version)
dataset_names = metadataset.get_dataset_names()
dataset_name = dataset_names[0]

results_per_dataset_val = {}
results_per_dataset_test = {}
all_results = []
best_configs = []
for dataset_name in dataset_names:
    metadataset.set_state(dataset_name=dataset_name)
    hps = metadataset.row_hp_candidates[dataset_name]
    ids = np.arange(len(hps))
    hps_df = pd.DataFrame(hps)
    results_per_dataset_val[dataset_name] = {}
    results_per_dataset_test[dataset_name] = {}
    best_metric = np.inf
    best_conf = None
    for model in MODELS:
        X_obs = ids[hps_df.model ==model]

        if len(X_obs)>0:
            _, metric, metric_per_pipeline, _ = metadataset.evaluate_ensembles([X_obs])
            metric_per_pipeline = metric_per_pipeline.numpy()
            val_metric = np.min(metric_per_pipeline[0])
            best_val_model_id = np.argmin(metric_per_pipeline[0])
            
            if val_metric < best_metric:
                best_metric = val_metric
                best_conf = hps_df.iloc[X_obs[best_val_model_id]]

            results_per_dataset_val[dataset_name][model]= {
                                                "best_val_model_id": best_val_model_id,
                                                "val_metric": val_metric
                                                }
            for i, metric in enumerate(metric_per_pipeline[0]):
                all_results.append({
                    "mode": model,
                    "dataset_name": dataset_name,
                    "learning_rate": hps[i]["learning_rate"],
                    "lora_r":  hps[i]["lora_r"],
                    "metric": metric_per_pipeline[0][i]
                })
        else:
            results_per_dataset_val[dataset_name][model]= {
                                                "best_val_model_id": np.nan,
                                                "val_metric": np.nan
                                                    }           
                                            
        
    best_configs.append((dataset_name, best_conf.values.tolist()))
    metadataset.set_state(dataset_name=dataset_name,
                          split="test")
    
    for model in MODELS:
        X_obs = ids[hps_df.model ==model]

        if len(X_obs)>0:
            best_val_model_id = X_obs[results_per_dataset_val[dataset_name][model]["best_val_model_id"]]
            _, metric, metric_per_pipeline, _ = metadataset.evaluate_ensembles([[best_val_model_id]])
            metric_per_pipeline = metric_per_pipeline.numpy()
            results_per_dataset_test[dataset_name][model] =metric_per_pipeline[0][0]
        else:
            results_per_dataset_test[dataset_name][model] = np.nan

all_results = pd.DataFrame(all_results)
heatmap_aggregated = all_results[["learning_rate", "lora_r", "metric"]].groupby(["learning_rate", "lora_r"]).mean().reset_index().pivot_table(columns="lora_r", values="metric", index="learning_rate")  

plt.figure(figsize=(8, 8))
plt.tick_params(axis='both', which='major', labelsize=fontsize*0.7)

ax = sns.heatmap(heatmap_aggregated, annot=True, fmt=".4f", cmap="viridis",  annot_kws={"size": 20}, cbar=False)

plt.xlabel("LoRA Rank", fontsize=fontsize)
plt.ylabel("Learning Rate", fontsize=fontsize)
plt.tight_layout()
plt.savefig(f"results/heatmap_{data_version}.png", bbox_inches='tight')
plt.savefig(f"results/heatmap_{data_version}.pdf", bbox_inches='tight')

df_results = pd.DataFrame(results_per_dataset_test)
df_results.to_latex(f"results/{data_version}_model_comparison.tex")
print("Done.")


    