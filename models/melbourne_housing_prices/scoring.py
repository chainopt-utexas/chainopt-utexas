import pandas as pd
import os
import joblib

def predict_price(house_metadata):
    '''
        Model that predicts the prices of houses in the Melbourne housing market. Test case for joblib format with CSV input and embedded model
    '''
    # Load Pretrained Model
    model_filepath = os.path.join(os.path.dirname(__file__), 'pred_house_prices.joblib')
    pretrained_model_file = open(model_filepath, 'rb')
    pretrained_model = joblib.load(pretrained_model_file) 
    pretrained_model_file.close()
    
    # Temp file paths
    temp_le_model_filepath = os.path.join(os.path.dirname(__file__), 'temp/labelencoding.pkl')

    ### Start Prepare
    X_score = house_metadata

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

    dfOut = pretrained_model.predict(Xle_score[colsNum+colsCat])

    ### End Predict

    return dfOut