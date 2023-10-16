is_azure_environment() {
    if [ "$USER" == "azureuser" ]; then
        return 0
    else
        return 1
    fi
}


get_current_python_virtual_environment() {
    conda info | extract_row_fields ": " "active environment :" "2"
}

extract_row_fields() {
    local separator="$1"
    local presence_filter="$2"
    local output_field_index="$3"
    
    awk -F "${separator}" "/${presence_filter}/ { print \$${output_field_index} }"
}


conda_environment_exists() {
  local environment="$1"
  
  conda env list | grep -q "^${environment}[[:space:]]"
  return $?
}


is_python_package_installed() {
    local package="$1"
    
    pip show "${package}" > /dev/null 2>&1
    
    return $?
}


set_up_environment_for_azure() {
    conda init bash
    conda deactivate
    conda env remove --name $ENV_NAME
    conda create -n $ENV_NAME python=3.10 --yes
    conda activate $ENV_NAME
    conda install -c anaconda ipykernel --yes
    python -m ipykernel install --user --name=$ENV_NAME
    poetry install
}


activate_poetry() {
    curl -sSL https://install.python-poetry.org | POETRY_HOME="${POETRY_GLOBAL_PATH}" python3 -
    export PATH="${POETRY_GLOBAL_PATH}/bin:$PATH"
}

activate_ssh_agent() {
    eval "$(ssh-agent -s)"
    ssh-add ~/.ssh/github_rubchume
}

install_tts() {
    git clone https://github.com/coqui-ai/TTS/
    cd TTS
    make system-deps  # only on Linux systems.
    make install
    cd ..
}

install_piper_tts() {
    sudo apt-get install python3-dev
    git clone https://github.com/rhasspy/piper.git
    cd piper/src/python
    conda create -n piperttsenv python=3.10 --yes
    conda activate piperttsenv
    pip3 install --upgrade pip
    pip3 install --upgrade wheel setuptools
    pip3 install -e .
    ./build_monotonic_align.sh
    conda deactivate
    sudo apt-get -y install espeak-ng
    cd -
}


ENV_NAME=voicecloningenv
POETRY_GLOBAL_PATH=/home/azureuser/poetry

if is_azure_environment; then
    echo "Setting up Python virtual environment for Azure compute instance"
    activate_poetry
    activate_ssh_agent
    activePythonEnvironment=$(get_current_python_virtual_environment)
    if [ "$activePythonEnvironment" != "$ENV_NAME" ]; then
        if conda_environment_exists $ENV_NAME; then
            conda activate "$ENV_NAME"
        else
            set_up_environment_for_azure
        fi
    else
        echo "Virtual environment is already active"
    fi
fi

if ! is_python_package_installed "openai-whisper"; then
    pip install -U openai-whisper
    sudo apt update && sudo apt -y install ffmpeg
fi

# if ! is_python_package_installed "TTS"; then
#     install_tts
# fi

if ! conda_environment_exists "piperttsenv"; then
    install_piper_tts
fi