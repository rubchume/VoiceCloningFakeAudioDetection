from pathlib import Path

from utilities import get_module_declared_variables, create_variables_in_module
    
    
paths = dict(
    data_path=Path("data"),
    job_definitions_path=Path("job_definitions"),
    pipelines_path=Path("pipelines"),
    computes_path=Path("job_definitions") / "computes",
    environments_path=Path("job_definitions") / "environments",
    source_path=Path("src"),
    audio_output_path=Path("outputs"),
    training_artifacts_path=Path("artifacts"),
    models_path=Path("models"),
)


for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)

    
create_variables_in_module(paths)    