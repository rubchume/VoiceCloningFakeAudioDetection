import argparse
import inspect
import os

from azureml.core import Workspace, Dataset, Datastore


def make_command(function):
    parser = argparse.ArgumentParser()
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        parser.add_argument(f"--{parameter_name}", type=parameter.annotation if parameter.annotation != inspect._empty else None)
    
    def wrapper():
        args, unknown = parser.parse_known_args()
        non_null_args = {key: value for key, value in vars(args).items() if value is not None}
        return function(**non_null_args)
    
    return wrapper


from contextlib import contextmanager
from pathlib import Path
import sys

from azureml.core import Workspace, Dataset as AzureCoreDataset, Datastore


@contextmanager
def suppress_error_print():
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
    

@contextmanager
def mounted_datastore(datastore_name, relative_path):
    workspace = get_workspace()
    datastore = Datastore.get(workspace, datastore_name)
    dataset = AzureCoreDataset.File.from_files(path=(datastore, relative_path))
    mounted_path = dataset.mount()
    
    with suppress_error_print():
        mounted_path.start()

    yield mounted_path.mount_point
    mounted_path.stop()
    
    
def upload_files_to_datastore(datastore_name, destination_path, source_path, pattern=None):
    workspace = get_workspace()

    datastore = Datastore.get(workspace, "workspaceblobstore")
    Dataset.File.upload_directory(
        src_dir=source_path,
        target=(datastore, destination_path),
        pattern=pattern,
        overwrite=True,
        show_progress=True
    )
    
    
from azureml.core import Run, Workspace


def get_workspace():
    try:
        run = Run.get_context()
        return run.experiment.workspace
    except AttributeError:
        return Workspace.from_config()