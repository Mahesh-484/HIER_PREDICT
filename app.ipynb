{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7f0a6e26-2c26-4a6f-9692-49935987d772",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app '__main__'\n",
      " * Debug mode: on\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.\n",
      " * Running on http://127.0.0.1:5000\n",
      "Press CTRL+C to quit\n",
      " * Restarting with watchdog (windowsapi)\n"
     ]
    },
    {
     "ename": "SystemExit",
     "evalue": "1",
     "output_type": "error",
     "traceback": [
      "An exception has occurred, use %tb to see the full traceback.\n",
      "\u001b[1;31mSystemExit\u001b[0m\u001b[1;31m:\u001b[0m 1\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template, request\n",
    "import joblib\n",
    "import numpy as np\n",
    "import warnings \n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "Hier_model = joblib.load('HierModel.pkl')\n",
    "rf_model = joblib.load('RF_Classifier.pkl')\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template('Home_AD4.html')\n",
    "\n",
    "@app.route('/Result_AD4', methods=['POST'])\n",
    "def result():\n",
    "    try:\n",
    "        input_features = [float(request.form[key]) for key in request.form.keys()]\n",
    "        input_features = np.array(input_features).reshape(1, -1)\n",
    "\n",
    "        cluster_label = Hier_model.fit_predict(input_features)[0]\n",
    "\n",
    "        segment = rf_model.predict(input_features)[0]\n",
    "\n",
    "        return render_template('Result_AD4.html', segment=segment, cluster=cluster_label)\n",
    "    except Exception as e:\n",
    "        error_message = f\"Error occurred: {str(e)}\"\n",
    "        return render_template('Result_AD4.html', segment=error_message, cluster=None)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "27c40cd8-fa22-4b2b-bf58-a738d1b50d27",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
