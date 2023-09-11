import mlflow
import logging


logging.basicConfig(level=logging.INFO)


def main():
    logging.debug("Start training")
    with mlflow.start_run() as run:
        parameters = dict(test_parameter="test_value")
        mlflow.log_params(parameters)
    
    logging.debug("Finished training")
    

if __name__=="__main__":
    main()
