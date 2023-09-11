import argparse
import inspect
import logging
from pathlib import Path

import mlflow


logging.basicConfig(level=logging.INFO)


def make_command(function):
    parser = argparse.ArgumentParser()
    for parameter_name, parameter in inspect.signature(function).parameters.items():
        parser.add_argument(f"--{parameter_name}", type=parameter.annotation if parameter.annotation != inspect._empty else None)
    
    def wrapper():
        args = parser.parse_args()
        return function(**vars(args))
    
    return wrapper


@make_command
def main(audio_dataset):
    logging.info("Start training")
    with mlflow.start_run() as run:
        parameters = dict(test_parameter="test_value")
        mlflow.log_params(parameters)
        mlflow.log_metric("some_metric", 123)
        
        print("HELLOOOOO")
        print(list(Path(audio_dataset).iterdir())[:5])
    
    logging.info("Finished training")
    

if __name__=="__main__":
    main()
