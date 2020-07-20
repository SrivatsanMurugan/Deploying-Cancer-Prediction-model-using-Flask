from flask import Flask, render_template, request

from model import train_svm, test_svm, predict_svm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from time import time


app = Flask(__name__)

@app.route('/')
def hello_method():
	return render_template('form.html')

@app.route('/predict', methods=['POST']) 
def login_user():


	data = []
	string = 'field'
	for i in range(1,31):
		data.append(float(request.form['field'+str(i)]))

	#for i in range(30):
		#print(data[i])


	df_numpy = np.asarray(data, dtype = float)
	df_numpy = df_numpy.reshape(1,30)
	output, acc= predict_svm(clf, df_numpy)

	if(output==1):
		final_output = 'Malignant'
	else:
		final_output = 'Benign'

	accuracy_x = acc[0][0]
	accuracy_y = acc[0][1]
	if(accuracy_x>accuracy_y):
		acc = accuracy_x
	else:
		acc=accuracy_y
	return render_template('result.html', output=final_output, accuracy=acc*100)


if __name__=='__main__':
	global clf 
	clf = train_svm()
	test_svm(clf)
	print("Done")
	app.run(port=4995)

