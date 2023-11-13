from pathlib import Path

import pandas as pd

from utils import make_command


@make_command
def main(common_voice_dataset, files_info_tsv, audio_files_csv):
    validated_tsv_path = Path(common_voice_dataset) / files_info_tsv
    pd.read_csv(validated_tsv_path, delimiter="\t").path.to_csv(audio_files_csv, header=False, index=False)
    
    
if __name__ == "__main__":
    main()
