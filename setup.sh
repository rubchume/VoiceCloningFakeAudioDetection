is_azure_environment() {
    if [ "$USER" == "azureuser" ]; then
        return 0
    else
        return 1
    fi
}


set_up_environment_for_azure() {
    ENV_NAME=voicecloningenv
    conda init bash
    conda deactivate
    conda env remove --name $ENV_NAME
    conda create -n $ENV_NAME python=3.11 --yes
    conda activate $ENV_NAME
    conda install -c anaconda ipykernel --yes
    python -m ipykernel install --user --name=$ENV_NAME

    curl -sSL https://install.python-poetry.org | POETRY_HOME=~/poetry python3 -
    export PATH="/home/azureuser/poetry/bin:$PATH"
    poetry install
}


if is_azure_environment; then
    echo "Setting up Python virtual environment for Azure compute instance"
    set_up_environment_for_azure
fi

pip install -U openai-whisper
sudo apt update && sudo apt install ffmpeg
