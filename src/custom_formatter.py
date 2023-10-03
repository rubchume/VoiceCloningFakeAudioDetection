import csv
from pathlib import Path


def custom_formatter(root_path, meta_file, **kwargs):
    speaker_name = "Eva"
    with (Path(root_path) / meta_file).open("r", encoding="utf-8") as file:
        return [
            {"text": text, "audio_file": str(Path(root_path) / file_name), "speaker_name": speaker_name, "root_path": root_path}
            for file_name, text in csv.reader(file, delimiter="|")
        ]
