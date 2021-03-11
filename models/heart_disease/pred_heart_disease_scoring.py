import pandas as pd
import pickle 
import os

def predict_heart_disease(patient_metrics):

    # Load Pretrained Model
    model_filepath = os.path.join(os.path.dirname(__file__), 'pred_heart_disease.pkl')
    pretrained_model_file = open(model_filepath, 'rb')
    pretrained_model = pickle.load(pretrained_model_file) 
    pretrained_model_file.close()

    # Apply predictions to scoring file
    dfOut = patient_metrics
    dfOut['tgt_heart_disease'] = pretrained_model.predict(patient_metrics)

    ### End Predict

    return dfOut