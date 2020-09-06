from flask import Flask, render_template,request,send_file

import pickle
import pandas as pd
import numpy as np

model=pickle.load(open('Santander_Prediction_model.model','rb'))
scaler=pickle.load(open('scaler.model','rb'))
columns_to_drop=pickle.load(open('columns_to_drop.list','rb'))

app = Flask(__name__)

def predict(test_data):
    test_data.drop(columns=columns_to_drop,inplace=True,axis=1)
    num_cols=test_data.drop(columns='ID_code').columns
    test_data[num_cols]=scaler.transform(test_data.drop(columns='ID_code'))
    pred=model.predict(test_data.drop(columns='ID_code'))
    pred=np.where(pred>=0.5,1,0)
    test_data['target']=pred
    test_data=test_data[['ID_code','target']]
    return test_data
    

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['GET','POST'])
def upload_predict():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        shape = df.shape
        df=predict(df)
        df.to_csv('predictions.csv',index=False)
        return render_template('predict.html', shape=shape,d=df.values)
    else:
        return render_template('predict.html')
        
@app.route('/csv')  
def download_csv():  
    return send_file('predictions.csv',
                     mimetype='text/csv',
                     attachment_filename='predictions.csv',
                     as_attachment=True)

    

if __name__=='__main__':
    app.run(debug=True) 


