
import numpy as np
import cv2
import pickle

frameWidth=640
frameHeight=480
brightness=180
threshold=0.90
font=cv2.FONT_HERSHEY_SIMPLEX

#SET UP THE VIDEO CAMERA
cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,brightness)

from tensorflow import keras
my_model=keras.models.load_model('my_model')
my_model.load_weights("weights.h5")

def grayscale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img=cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img=grayscale(img)
    img=equalize(img)
    img=img/255
    return img
def getClassName(classNo):
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
    elif classNo==12: return 'Priotity Road'
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
    elif classNo==42: return 'End of no passing by vehicles over 3.5 metric tons'    

def camera():
    while True:
    #Read image
        success,imgOriginal=cap.read()
        img=np.asarray(imgOriginal)
        img=cv2.resize(img,(32,32))
        img=preprocessing(img)
        #cv2.imshow("processed image",img)
        img=img.reshape(1,32,32,1)
        cv2.putText(imgOriginal,"Traffic sign: ",(20,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(imgOriginal,"Probability: ",(20,75),font,0.75,(0,0,255),2,cv2.LINE_AA)
    
        #predict image
        predictions=my_model.predict(img)
        classIndex=np.argmax(predictions,axis=1)
        probabilityvalue=np.amax(predictions)
    
        if probabilityvalue>threshold:
            #print(getClassName(classIndex))
            cv2.putText(imgOriginal,str(classIndex)+" "+str(getClassName(classIndex)),(120,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(imgOriginal,str(round(probabilityvalue*100,2))+"%",(180,75),font,0.75,(0,0,255),2,cv2.LINE_AA)
            """result="Predicted Traffic Sign is: "+getClassName(classIndex)
            print(result)
            engineio = pyttsx3.init()
            engineio.say(result)
            results=engineio.runAndWait()
            #return result
            #return results"""
        cv2.imshow("Result",imgOriginal)
        if cv2.waitKey(1) and 0xFF==ord('q'):
            break




