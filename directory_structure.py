from pathlib import Path


job_definitions_path = Path("job_definitions")
computes_path = job_definitions_path / "computes"
environments_path = job_definitions_path / "environments"
source_path = Path("src")

Path(computes_path).mkdir(exist_ok=True, parents=True)
Path(environments_path).mkdir(exist_ok=True, parents=True)