from pathlib import Path


# Define paths to folders
data_path = Path("data")
job_definitions_path = Path("job_definitions")
computes_path = job_definitions_path / "computes"
environments_path = job_definitions_path / "environments"
source_path = Path("src")
audio_output_path = Path("outputs")
training_artifacts_path = Path("artifacts")

# Create folders if they do not exist
Path(computes_path).mkdir(exist_ok=True, parents=True)
Path(environments_path).mkdir(exist_ok=True, parents=True)
audio_output_path.mkdir(exist_ok=True, parents=True)
training_artifacts_path.mkdir(exist_ok=True, parents=True)