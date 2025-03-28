#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, render_template, request
import joblib
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

Hier_model = joblib.load('HierModel.pkl')
rf_model = joblib.load('RF_Classifier.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('Home_AD4.html')

@app.route('/predict', methods=['POST'])
def result():
    try:
        input_features = [float(request.form[key]) for key in request.form.keys()]
        input_features = np.array(input_features).reshape(1, -1)

        cluster_label = Hier_model.fit_predict(input_features)[0]

        segment = rf_model.predict(input_features)[0]

        return render_template('Result_AD4.html', segment=segment, cluster=cluster_label)
    except Exception as e:
        error_message = f"Error occurred: {str(e)}"
        return render_template('Result_AD4.html', segment=error_message, cluster=None)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




