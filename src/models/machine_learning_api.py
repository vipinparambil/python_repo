
from flask import Flask,request
import pandas as pd
import os
import numpy as np
import pickle
import json

app =Flask(__name__)

model_path = os.path.join(os.path.pardir,os.path.pardir,'models')

model_file_path = os.path.join(model_path, 'lr_model.pkl')

saclar_file_path = os.path.join(model_path, 'lr_scaler.pkl')

#Import Model
with open(model_file_path, 'rb') as file:
    model = pickle.load(file)

with open(saclar_file_path, 'rb') as file:
    scalar = pickle.load(file)

# columns
columns = [ u'Age', u'Fare', u'FamilySize', \
       u'IsMother', u'IsMale', u'Deck_A', u'Deck_B', u'Deck_C', u'Deck_D', \
       u'Deck_E', u'Deck_F', u'Deck_G', u'Deck_Z', u'Pclass_1', u'Pclass_2', \
       u'Pclass_3', u'Title_Lady', u'Title_Master', u'Title_Miss', u'Title_Mr', \
       u'Title_Mrs', u'Title_Officer', u'Title_Sir', u'Fare_Bin_very_low', \
       u'Fare_Bin_low', u'Fare_Bin_high', u'Fare_Bin_very_high', u'Embarked_C', \
       u'Embarked_Q', u'Embarked_S', u'AgeState_Adult', u'AgeState_Child'] 

#prediction api
@app.route('/api', methods=['POST'])
def make_prediction():
    # read json object and conver to json string
    data = request.get_json()
    # create pandas dataframe using json string
    df = pd.read_json(data)
    # extract passengerIds
    #passenger_ids = df['PassengerId'].ravel()
    # actual survived values
    actuals = df['Survived'].ravel()
    # extract required columns based and convert to matrix
    X = df[columns].values.astype('float')
    # transform the input
    X_scaled = scalar.transform(X)
    # make predictions
    predictions = model.predict(X_scaled)
    # create response dataframe
    df_response = pd.DataFrame({'Predicted' : predictions, 'Actual' : actuals})
    # return json 
    return df_response.to_json()

if __name__ == '__main__':
     app.run(host = '0.0.0.0', port = 10001, debug=False)
    
