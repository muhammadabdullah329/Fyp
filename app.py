from flask import Flask, render_template, Response
import cv2
import sys
import os
from sys import platform
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.models import load_model
import time
import warnings
warnings.filterwarnings("ignore")


app = Flask(__name__)

camera = cv2.VideoCapture(0)  # use 0 for web camera

model = load_model('../output/har_simpleLSTM.hdf5')
model.summary()

inputarray = []
outresult = ['kneeling', 'overhead', 'sitting', 'squatting', 'stooping']#, 'sitting'


def printKeypoints(datums):
    datum = datums[0]
    test = []
    a = datum.poseKeypoints       #[0,:,:2]
    a = np.array(a)
    p= a.flatten()
    p= np.delete(p, slice(2, None, 3))
    #print(p)
    
    scaler = MinMaxScaler()
    p = p.reshape(-1,1)
    p = scaler.fit_transform(p)
    p = p.flatten()
    if len(p) > 50:
        p = p[:50]
    
    inputarray.append(p)
	
    
    
    if len(inputarray) > 12:
        inputarray.pop(0) if inputarray else False
        #print (inputarray)
        if np.array(inputarray).shape != (12,50):
            return
        temp = np.array(inputarray)
        
        c=model.predict(np.expand_dims(temp,axis=0))
        c = c.flatten()
        c = np.array(c)
        #print (c)
        
        max_val = np.argmax(c)
        maxy = np.max(c)
        #print(maxy)
        print(outresult[max_val])
        return outresult[max_val]


TextCounter=0
tempText="" 
img = None
def display(datums):
    datum = datums[0]
    global img
    global tempText
    global TextCounter
    img = datum.cvOutputData
    text = printKeypoints(datums)
    
    if text == tempText:
        TextCounter+=1
    else:
        TextCounter=0

    tempText = text
    
    if TextCounter >= 40:  #minimum time required to detect 
        img = cv2.putText(img, text, (300,200),cv2.FONT_HERSHEY_COMPLEX,1,(0,150,0),2)
    frame = cv2.imencode('.jpg', img)[1].tobytes()
    return frame



def gen_frames():  # generate frame by frame from camera
	try:
			# Import Openpose (Windows/Ubuntu/OSX)
			dir_path = os.path.dirname(os.path.realpath(__file__))
			try:
				# Change these variables to point to the correct folder (Release/x64 etc.)
				sys.path.append(dir_path + '/../bin/python/openpose/Release');
				os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' +  dir_path + '/../bin;'
				import pyopenpose as op
			except ImportError as e:
				print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
				raise e

			# Flags
			parser = argparse.ArgumentParser()
			parser.add_argument("--no-display", action="store_true", help="Disable display.")
			args = parser.parse_known_args()

			# Custom Params (refer to include/openpose/flags.hpp for more parameters)
			params = dict()
			params["model_folder"] = "../models/"

			# Add others in path?
			for i in range(0, len(args[1])):
				curr_item = args[1][i]
				if i != len(args[1])-1: next_item = args[1][i+1]
				else: next_item = "1"
				if "--" in curr_item and "--" in next_item:
					key = curr_item.replace('-','')
					if key not in params:  params[key] = "1"
				elif "--" in curr_item and "--" not in next_item:
					key = curr_item.replace('-','')
					if key not in params: params[key] = next_item

			# Construct it from system arguments
			# op.init_argv(args[1])
			# oppython = op.OpenposePython()

			# Starting OpenPose
			opWrapper = op.WrapperPython(op.ThreadManagerMode.AsynchronousOut)
			opWrapper.configure(params)
			opWrapper.start()


			
			while True:
				# Pop frame
				datumProcessed = op.VectorDatum()
				if opWrapper.waitAndPop(datumProcessed):
					if not args[0].no_display:
						# Display image
						
						frame = display(datumProcessed)
						yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
					
				else:
					break
	except Exception as e:
		print(e)
		sys.exit(-1)
            

img_counter = 0
@app.route('/imgCaptured')
def imgCaptured():
    global img_counter
    global img
    img_name = "opencv_frame_{}.png".format(img_counter)
    cv2.imwrite(img_name, img)
    img_counter += 1
    return render_template('imgCaptured.html')
    


@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """home page"""
    return render_template('index.html')
	
@app.route('/index.html')
def home():
    """home page"""
    return render_template('index.html')
	
@app.route('/live.html')
def live():
    """Video streaming page."""
    return render_template('live.html')


if __name__ == '__main__':
    app.run(debug=True)