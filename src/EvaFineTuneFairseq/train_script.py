import runpy

from custom_formatter import custom_formatter
from TTS.tts import datasets


def main():
    datasets.custom_formatter = custom_formatter
    runpy.run_module("TTS.bin.train_tts", run_name='__main__', alter_sys=True)


if __name__ == "__main__":
    main()
