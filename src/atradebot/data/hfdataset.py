
import pandas as pd
import json
import os
from atradebot.config import FILES_D
from datasets import Dataset as HFDataset
pd.options.mode.chained_assignment = None

def collect_files_hf(directory: str) -> HFDataset:
    """
    Collect all JSONL files that start with the given prefix, combine them into one dataset, and convert to a HuggingFace dataset.

    Args:
        directory (str): The directory to search for files.
        prefix (str): The prefix of the files to collect.

    Returns:
        HFDataset: The combined dataset as a HuggingFace dataset.
    """
    prefix = 'atradebot_collect_'
    combined_data = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.jsonl'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                for line in file:
                    combined_data.append(json.loads(line))
    hfdata = HFDataset.from_pandas(pd.DataFrame(combined_data))
    return hfdata

if __name__ == "__main__":
    hfdata = collect_files_hf(FILES_D)
    hfdata.save_to_disk(os.path.join(FILES_D, 'atradebot_dataset'))
    print('Done')