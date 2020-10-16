import pandas as pd
from loguru import logger

def score_model(model, input_data: pd.DataFrame) -> pd.DataFrame:
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