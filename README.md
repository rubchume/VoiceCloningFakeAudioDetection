# Introduction

This is a project about human voice cloning and audio classification using huge amounts of data with the help of Microsoft Azure ML platform.

There are two parts in this project: cloning and classification.

The first one is about cloning voices from the [TIMIT dataset](https://paperswithcode.com/dataset/timit), which contains a huge sample of speakers with different accents in the English language.

The second part is the training of a machine learning model that can classify audio recordings as real or cloned voices. The cloned voices in the first part and real recordings from the [CommonVoice dataset](https://commonvoice.mozilla.org/en/datasets) were used.

# Data description

As said before, two datasets were used.

The [TIMIT](https://paperswithcode.com/dataset/timit) dataset is very structured, with multiple speakers from different accents saying out loud some specific sentences. As we have information about the speaker and the accent, this allows for the training of models with different accents or even train a model with multiple sentences (which is desirable).
There are examples (in some notebooks) of cloned voices with a big amount of voice recordings, and that allows for creating a high quality synthetic voice that copies with high accuracy the original voice.
However, for our purpose we are going to focus on *one-shot* cloning, i.e. using just one sentence.

The [CommonVoice dataset](https://commonvoice.mozilla.org/en/datasets) is less structured and actually the quality of recordings is reviewed by the community, so I focused only on the recordings that were validated. Nonetheless, even with that restriction the amount of data is huge (tenths of gigabytes). This fact makes it difficult to work with this dataset. A laptop or a personal computer might not be enough to cope with the training over such a huge amount of data. Cloud technologies helped with this issue.

# Approach(es)

Most files are Python scripts or Jupyter Notebooks that have been created in the process of trying different approaches to the problem. They are left here just as a reference and starting point for future development.

However, the final approach used only requires to follow a few Jupyter Notebooks. You can use this file as a guide to navigate them.

In the cloning step, a voice cloning model (YourTTS) is used to clone the audio samples and sentences from the TIMIT dataset. You can follow this process in the notebook [notebooks_cloning/Synthesize - Chosen model.ipynb](notebooks_cloning/Synthesize%20-%20Chosen%20model.ipynb). The synthetic audios were transcribed with OpenAI whisper and the WER (Word Error Rate) was calculated against the real words.

The classification step trains a CNN (Convolutional Neural Network) model on the MEL-spectorgrams of each audio for differentiating between cloned voices and real ones, using the TIMIT and Common Voice datasets.
The training of the model is done in [notebooks_classification/CNN_voice_classification.ipynb](notebooks_classification/CNN_voice_classification.ipynb).
Then the trained model is used to classify the whole TIMIT dataset and Common Voice datasets. You can follow this proces in [notebooks_classification/CNN_voice_prediction.ipynb](notebooks_classification/CNN_voice_prediction.ipynb).

As said above, the amount of data to be trained on and classify was huge. I coped with this issue by creating Microsoft Azure ML jobs that could run for hours (more than 1 day in some cases) on powerful compute clusters with GPUs. Without the help of cloud solutions this project can be very inconvenient due to the high demand in computing resources for a personal computer.

# Setup
For executing the notebooks in Linux, several packages must be installed. Use:
```bash
source setup.sh
```

# Conclusions

Regarding the cloning of voices, I must say that the state of the art is very good (2023), but even the most used Python libraries (like CoquiTTS) still need to mature much more.
Dealing with compatibility issues between packages, long and complex configurations for each model, the fact that a lot of the code was developed as part of PhD Thesis (focused on the academic value but not focused on usability and not following the best practices in software development), a documentation that easily gets outdated given the fast pace of evolution of the code... All those features of immature code made it hard to complete this part.

And I am afraid that with the passing of the months and the years, the code I used will become old and buggy (unless all package versions are preserved as they are in the `pyproject.toml` file).

That being said, once I made the chosen algorithm work for my case, the performance was very good. The WER was about 20%, which is quite good considering that there was the cloning step and the transcribing step adding inaccuracies.

With respect to the classification step, different algorithms were used but the most convenient and accurate way of doing it is still the classic approach of computing MEL-spectograms and classifying them with a simple CNN. At least for this use case, the performance was very good.

![image](https://github.com/rubchume/VoiceCloningFakeAudioDetection/assets/83817841/4ebbbfad-0efa-44da-a5bf-4be82e0c33e2)

As stated above, the use of Azure jobs was crucial for this to be completed.
Another difficulty I found was that the cutting of audios (the offset) had to be adjusted to avoid overfitting, since all synthetic voices shared the same amount of initial offset.
