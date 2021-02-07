import pandas as pd
from loguru import logger

class Scorer:
    def __init__(self,model):
        #Do something here if you need to on initialization. This code will be run once on class instance creation.
        logger.debug("Initialized Scorer")

    def score_model(self, model, input_data: pd.DataFrame) -> pd.DataFrame:
        '''
            Model that predicts heart disease in patients.
        '''
        logger.debug("Inside score_model method of scoring.py...")
        
        ### Start Prepare
        X_score = input_data

        ### End Prepare

        ### Start Predict

        # Apply predictions to scoring file
        dfOut = input_data
        dfOut['tgt_heart_disease'] = model.predict(X_score)

        output_data = dfOut

        ### End Predict

        return output_data