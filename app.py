from flask import *
import os
from werkzeug.utils import secure_filename
from keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import pyttsx3
import recognize
app = Flask(__name__)
# Classes of trafic signs
classes = { 0:'Speed limit (20km/h)',
			1:'Speed limit (30km/h)',
			2:'Speed limit (50km/h)',
			3:'Speed limit (60km/h)',
			4:'Speed limit (70km/h)',
			5:'Speed limit (80km/h)',
			6:'End of speed limit (80km/h)',
			7:'Speed limit (100km/h)',
			8:'Speed limit (120km/h)',
			9:'No passing',
			10:'No passing veh over 3.5 tons',
			11:'Right-of-way at intersection',
			12:'Priority road',
			13:'Yield',
			14:'Stop',
			15:'No vehicles',
			16:'Vehicle > 3.5 tons prohibited',
			17:'No entry',
			18:'General caution',
			19:'Dangerous curve left',
			20:'Dangerous curve right',
			21:'Double curve',
			22:'Bumpy road',
			23:'Slippery road',
			24:'Road narrows on the right',
			25:'Road work',
			26:'Traffic signals',
			27:'Pedestrians',
			28:'Children crossing',
			29:'Bicycles crossing',
			30:'Beware of ice/snow',
			31:'Wild animals crossing',
			32:'End speed + passing limits',
			33:'Turn right ahead',
			34:'Turn left ahead',
			35:'Ahead only',
			36:'Go straight or right',
			37:'Go straight or left',
			38:'Keep right',
			39:'Keep left',
			40:'Roundabout mandatory',
			41:'End of no passing',
			42:'End no passing vehicle > 3.5 tons',
			43:'No Sign Detected'
            }

"""def getClassName(classNo):
    if classNo==0: return 'Speed Limit 20 km/h'
    elif classNo==1: return 'Speed Limit 30 km/h'
    elif classNo==2: return 'Speed Limit 50 km/h'
    elif classNo==3: return 'Speed Limit 60 km/h'
    elif classNo==4: return 'Speed Limit 70 km/h'
    elif classNo==5: return 'Speed Limit 80 km/h'
    elif classNo==6: return 'End of Speed Limit 30 km/h'
    elif classNo==7: return 'Speed Limit 100 km/h'
    elif classNo==8: return 'Speed Limit 120 km/h'
    elif classNo==9: return 'No Passing'
    elif classNo==10: return 'No Passing for vehicle over 3.5 tons'
    elif classNo==11: return 'Right-of way at intersection'
    elif classNo==12: return 'Priority Road'
    elif classNo==13: return 'Yield'
    elif classNo==14: return 'Stop'
    elif classNo==15: return 'No Vehicles'
    elif classNo==16: return 'Vehicles over 3.5 tons prohibited'
    elif classNo==17: return 'No entry'
    elif classNo==18: return 'General Caution'
    elif classNo==19: return 'Dangerous Curve to the left'
    elif classNo==20: return 'Dangerous Curve to the right'
    elif classNo==21: return 'Double Curve'
    elif classNo==22: return 'Bumpy Road'
    elif classNo==23: return 'Slippery Road'
    elif classNo==24: return 'Road Narrows on the right'
    elif classNo==25: return 'Road Work'
    elif classNo==26: return 'Traffic Signals'
    elif classNo==27: return 'Pedestrians'
    elif classNo==28: return 'Children Crossing'
    elif classNo==29: return 'BiCycles Crossing'
    elif classNo==30: return 'Beware of ice or snow'
    elif classNo==31: return 'Wild animals crossing'
    elif classNo==32: return 'End of all speed and passing limits'
    elif classNo==33: return 'Turn Right Ahead'
    elif classNo==34: return 'Turn Left Ahead'
    elif classNo==35: return 'Ahead only'
    elif classNo==36: return 'Go Straight or Right'
    elif classNo==37: return 'Go Straight or Left'
    elif classNo==38: return 'Keep Right'
    elif classNo==39: return 'Keep Left'
    elif classNo==40: return 'Roundabout mandatory'
    elif classNo==42: return 'End of no passing'
    elif classNo==42: return 'End of no passing by vehicles over 3.5 metric tons' """
def image_processing(img):
	model = load_model('./model/traffic.h5')
	data=[]
	image = Image.open(img)
	image = image.resize((30,30))
	data.append(np.array(image))
	X_test=np.array(data)
	#Y_pred = model.predict_classes(X_test)
	predict_x = model.predict(X_test)
	Y_pred = np.argmax(predict_x, axis=1)
	return Y_pred

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template("login.html")
@app.route('/first', methods=['GET'])
def first():
    # Main page
    return render_template('first.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
        # Get the file from post request
		f = request.files['file']
		file_path = secure_filename(f.filename)
		f.save(file_path)
        # Make prediction
		result = image_processing(file_path)
		s = [str(i) for i in result]
		a = int("".join(s))
		result = "Predicted Traffic Sign is: " +classes[a]
		os.remove(file_path)
		engineio = pyttsx3.init()
		voices = engineio.getProperty('voices')
		engineio.setProperty('voice', voices[1].id)
		engineio.say(result)
		results=engineio.runAndWait()
		return result
		return results        
	return None
@app.route('/live',methods=['GET','POST'])
def live():
    if request.method=='POST':
        recognize.camera()          
    return True
        
        
        
            
        
@app.route('/performance')

def performance():
    return render_template("performance.html")

@app.route('/chart')
def chart():
    return render_template("chart.html")    

if __name__ == '__main__':
    app.run(port=5000,debug=True)
