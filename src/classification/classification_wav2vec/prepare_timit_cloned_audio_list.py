from pathlib import Path

import pandas as pd

from utils import make_command


@make_command
def main(timit_cloned_dataset, audio_files_csv):
    cloned_info = pd.Series([path.name for path in Path(timit_cloned_dataset).glob("*.wav")]).rename("path")
    cloned_info.to_csv(audio_files_csv, header=False, index=False)
    
    
if __name__ == "__main__":
    main()
