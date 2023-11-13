from pathlib import Path

from utilities import get_module_declared_variables, create_variables_in_module
    
    
paths = dict(
    data_path=Path("data/datasets"),
    job_definitions_path=Path("job_definitions"),
    computes_path=Path("job_definitions") / "computes",
    environments_path=Path("job_definitions") / "environments",
    cloning_source_path=Path("src/cloning"),
    classification_source_path=Path("src/classification"),
    audio_output_path=Path("data/cloned_audios"),
    training_artifacts_path=Path("data/clone_training_artifacts"),
    models_path=Path("data/voice_cloning_models"),
    external_libraries_path=Path("external_libraries")
)


for path in paths.values():
    path.mkdir(parents=True, exist_ok=True)

    
create_variables_in_module(paths)    