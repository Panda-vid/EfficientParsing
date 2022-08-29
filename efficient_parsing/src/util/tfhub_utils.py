import re
import shutil

import tensorflow as tf

from pathlib import Path


def delete_matching_tfhub_cache(error: tf.errors.DataLossError):
    match = re.search(r"\/tmp\/tfhub_modules\/([a-zA-Z\d]+)\/", error.message)
    folder_name = match.group(0)
    model_folder = Path(folder_name)
    delete_tfhub_cache(model_folder)


def delete_tfhub_cache(model_path: Path):
    cache_path = Path("/tmp/tfhub_modules") / model_path
    if cache_path.is_dir():
        print(f"Deleting {str(cache_path)}.")
        shutil.rmtree(cache_path)
