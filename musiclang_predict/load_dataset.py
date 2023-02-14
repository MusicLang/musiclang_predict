import os
import shutil
from .constants import DEFAULT_OUTPUT, DATASET_DICT

def download_model(name, output=DEFAULT_OUTPUT, update=False, **kwargs):
    """
    Download a model using the dataset dict
    """
    output_path = os.path.join(output, name)
    if os.path.exists(output_path) and not update:
        print('Info: Model is already downloaded, if you want to force download set "update" parameter to true')
        return output_path

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    print(f'Downloading model "{name}"...')
    # git clone
    git_url = DATASET_DICT[name]
    from git import Repo
    Repo.clone_from(git_url, output_path)
    return output_path

def save_model(name):
    pass
