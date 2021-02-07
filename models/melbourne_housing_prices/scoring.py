import pandas as pd
from loguru import logger
import os

class Scorer:
    def __init__(self,model):
        #Do something here if you need to on initialization. This code will be run once on class instance creation.
        logger.debug("Initialized Scorer")

    def score_model(self, model, input_data: pd.DataFrame) -> pd.DataFrame:
        '''
            Model that predicts the prices of houses in the Melbourne housing market. Test case for joblib format with CSV input and embedded model
        '''
        logger.debug("Inside score_model method of scoring.py...")
        
        # Temp file paths
        temp_le_model_filepath = os.path.join(os.path.dirname(__file__), 'temp/labelencoding.pkl')

        ### Start Prepare
        X_score = input_data

        # Prepare numeric columns
        colsNum = ['Rooms','Distance','Postcode','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','Lattitude','Longtitude','Propertycount']
        Xnum_score = X_score[colsNum]
        Xnum_score_repnull = Xnum_score.fillna(0)
        X_score[colsNum] = Xnum_score_repnull[colsNum]

        # Prepare categorical columns
        colsCat = ['Type', 'Method', 'Regionname']

        # load the encoder file
        import pickle 
        le_file = open(temp_le_model_filepath, 'rb')
        le = pickle.load(le_file) 
        le_file.close()

        # apply the encoder
        Xle_score = X_score.copy()
        for col in colsCat:
            Xle_score[col] = le.fit_transform(X_score[col])

        ### End Prepare

        ### Start Predict

        dfOut = model.predict(Xle_score[colsNum+colsCat])

        output_data = dfOut

        ### End Predict

        return output_data