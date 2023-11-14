import argparse
import inspect
from pathlib import Path

import pandas as pd


def make_command(function):
    parser = argparse.ArgumentParser()
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        parser.add_argument(f"--{parameter_name}", type=parameter.annotation if parameter.annotation != inspect._empty else None)
    
    def wrapper():
        args = parser.parse_args()
        return function(**vars(args))
    
    return wrapper


@make_command
def main(common_voice_dataset, cloned_voice_dataset, real_voice_files, cloned_voice_files):
    real_voices_path = Path(common_voice_dataset)
    real_voices_info_file = Path(real_voices_path) / "validated.tsv"
    real_info = pd.read_csv(real_voices_info_file, delimiter="\t").path.map(
        lambda path: str(real_voices_path / "clips" / path)
    )
    
    cloned_info = pd.Series([str(path) for path in Path(cloned_voice_dataset).glob("*.wav")]).rename("path")
    
    real_info.str.removeprefix(common_voice_dataset).to_csv(real_voice_files, header=False, index=False)
    cloned_info.str.removeprefix(cloned_voice_dataset).to_csv(cloned_voice_files, header=False, index=False)
    

if __name__ == "__main__":
    main()
