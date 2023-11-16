import argparse
import inspect
import itertools
from pathlib import Path
import re
import shutil
import tempfile

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader

from audio_binary_dataset import AudioDumbDataset
from cloned_audio_detector import ClonedAudioDetector
from utils import get_workspace, mounted_datastore, upload_files_to_datastore


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_command(function):
    parser = argparse.ArgumentParser()
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        parser.add_argument(f"--{parameter_name}", type=parameter.annotation if parameter.annotation != inspect._empty else None)
    
    def wrapper():
        args, unknown = parser.parse_known_args()
        non_null_args = {key: value for key, value in vars(args).items() if value is not None}
        return function(**non_null_args)
    
    return wrapper


def get_model(job_name, download_path):
    ws = get_workspace()
    ml_client = MLClient(
        credential=DefaultAzureCredential(),
        subscription_id=ws.subscription_id,
        resource_group_name=ws.resource_group,
        workspace_name=ws.name,
    )
    ml_client.jobs.download(
        name=job_name,
        output_name='checkpoint',
        download_path=download_path
    )
    checkpoint_path = Path(download_path) / "named-outputs" / "checkpoint" / "checkpoint"
    detector_loaded = ClonedAudioDetector.load_from_checkpoint(checkpoint_path=checkpoint_path , map_location=device)
    detector_loaded.eval();
    return detector_loaded


def get_file_batch_indices(file):
    match = re.match(r"^logits_batch_(\d+)_(\d+)$", Path(file).stem)
    if match:
        return match.groups()


def predict_macro_batch(
    model,
    dataset,
    predictions_directory,
    batch_size: int = 100,
    macro_batch_size: int = 10,
    overwrite: bool = False
):
    with mounted_datastore(
        datastore_name="workspaceblobstore",
        relative_path=predictions_directory
    ) as predictions_path:
        files = pd.Series(Path(predictions_path).iterdir())
        
    if not overwrite:
        batch_indices = pd.DataFrame(files.map(get_file_batch_indices).dropna().tolist(), columns=["batch_size", "batch_index"]).astype("int")

        batch_indices_of_size = batch_indices[batch_indices.batch_size == batch_size].batch_index
        if len(batch_indices_of_size) > 0:
            last_index = batch_indices_of_size.sort_values(ascending=False).iloc[0]
        else:
            last_index = -1
    else:
        last_index = -1
    
    batch_start_index = last_index + 1
    batch_end_index = batch_start_index + macro_batch_size

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    iterable = itertools.islice(dataloader, batch_start_index, batch_end_index)

    Path("temp").mkdir(exist_ok=True)
    for batch_index, batch in enumerate(tqdm(iterable, total=macro_batch_size), start=batch_start_index):
        batch_logits = model.forward(batch.to(device))
        pd.DataFrame(batch_logits.cpu().detach().numpy()).to_csv(
            Path("temp") / f"logits_batch_{batch_size}_{batch_index}.csv",
            index=False,
            header=False
        )
        print(f"Batch ({batch_index}/{macro_batch_size}) completed")
    upload_files_to_datastore("workspaceblobstore", predictions_directory, "temp", "*.csv")
    shutil.rmtree("temp", ignore_errors=True)
        

@make_command
def main(job_name, model_download_path, data_path, audio_files_csv, audio_files_folder, predictions_path, batch_size: int = 1000, macro_batch_size: int =100):
    audio_files_folder = Path(data_path) / audio_files_folder
    audio_files_csv = audio_files_csv
    
    audio_files = pd.read_csv(str(audio_files_csv)).iloc[:, 0].map(
        lambda path: str(Path(audio_files_folder) / path)
    )
    
    detector_loaded = get_model(job_name, model_download_path)
    dumb_dataset = AudioDumbDataset(audio_files, 16000, 64000)
    predict_macro_batch(detector_loaded, dumb_dataset, predictions_path, batch_size=batch_size, macro_batch_size=macro_batch_size)

    
if __name__=="__main__":
    main()
