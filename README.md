# Introduction
This is a project about human voice cloning and audio classification.

Most files are Python scripts or Jupyter Notebooks that have been created in the process of trying different approaches to the problem. They are left here just as a reference and starting point for future development.

However, the final approach used only requires to follow a few Jupyter Notebooks. You can use this file as a guide to navigate them.

The project contains two main steps: cloning and classification. In the cloning step, a voice cloning model (YourTTS) is used to clone the audio samples and sentences from the TIMIT dataset. You can follow this process in the notebook [notebooks_cloning/Synthesize - Chosen model.ipynb](notebooks_cloning/Synthesize%20-%20Chosen%20model.ipynb).

The classification step trains a CNN model for differentiating between cloned voices and real ones, using the TIMIT and Common Voice datasets.
The training of the model is done in [notebooks_classification/CNN_voice_classification.ipynb](notebooks_classification/CNN_voice_classification.ipynb).
Then the trained model is used to classify the whole TIMIT dataset and Common Voice datasets. You can follow this proces in [notebooks_classification/CNN_voice_prediction.ipynb](notebooks_classification/CNN_voice_prediction.ipynb).

# Setup
Execute
```bash
source setup.sh
```