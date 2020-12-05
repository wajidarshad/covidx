# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 17:03:06 2020

@author: Dr_Wajid
"""

import numpy as np
import joblib
from sklearn import preprocessing


from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

def extract_transfer_learning_features(img_path):
    model = DenseNet121(weights='imagenet', include_top=False)
    img = image.load_img(img_path, target_size=(331, 331))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x).flatten()
    return features
def standarized_normalized_h_vs_u(features, norm='yes'):
    features=(features-np.load('trained_models/mean_helathy_vs_unhealth.npy'))/(np.load('trained_models/std_helathy_vs_unhealth.npy')+0.0001)
    #features=preprocessing.scale(features)
    if norm=='yes':
        features=preprocessing.normalize(features)
    return features
def standarized_normalized_c_vs_p(features, norm='yes'):
    features=(features-np.load('trained_models/mean_covid_vs_pneu.npy'))/(np.load('trained_models/std_covid_vs_pneu.npy')+0.0001)
    #features=preprocessing.scale(features)
    if norm=='yes':
        features=preprocessing.normalize(features)
    return features
def apply_covidx(image_path):
    feats=extract_transfer_learning_features(image_path)
    trained_model_h_vs_u=joblib.load('trained_models/helathy_vs_unhealth_svm_densene_unhealth_pos_helath_neg_reduced.joblib')
    if trained_model_h_vs_u.predict(standarized_normalized_h_vs_u([feats]))[0]<0:
        print("Healthy")
    else:
        trained_model_C_vs_P=joblib.load('trained_models/covid_vs_pneumonia_svm_densenet_covid_pos_pneu_neg.joblib')
        if trained_model_C_vs_P.predict(standarized_normalized_c_vs_p([feats]))[0]<0:
            print("Pneumonia")
        else:
            print('COVID-19')
            print('Model for severity prediction is coming soon.....')
    
if __name__ == "__main__":
    apply_covidx('input_image.jpg')
