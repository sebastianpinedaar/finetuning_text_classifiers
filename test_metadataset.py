from pathlib import Path
import os

from metadataset.ftc.metadataset import FTCMetadataset

if __name__ == "__main__":
    data_dir = Path(os.path.dirname(os.path.abspath(__file__))) /  "data" / "ftc"
    metadataset = FTCMetadataset(data_dir=str(data_dir))
    metadataset.set_state(dataset_name=metadataset.get_dataset_names()[4]
                          )
    metadataset.evaluate_ensembles([[1,2],[3,4]])
    metadataset.export_failed_configs('bash_args')
    print("Done")