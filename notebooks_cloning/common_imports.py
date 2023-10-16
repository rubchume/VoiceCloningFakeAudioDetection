import csv
from functools import reduce
import math
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
from typing import Callable, Optional
import warnings

import auditok
import azure
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential
import mlflow
from num2words import num2words
import pandas as pd
import plotly.express as px
import plotly.offline as pyo
import subprocess
from tqdm.notebook import tqdm
import whisper

import directory_structure


pyo.init_notebook_mode()