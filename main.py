from cProfile import label
from flask import Flask, request
from keras.models import load_model
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO
import io
import cv2
import base64
import numpy as np
from sympy import sequence

mp_holistic = mp.solutions.holistic

app = Flask(__name__)

model = load_model('model.h5')
labels = ['good', 'hello', 'how are you', 'I am', 'I love you', 'Nice to meet you', 'nill', 'no', 'sick', 'yes']

holistic = mp_holistic.Holistic(
    # static_image_mode=True,
    # model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
def mediapipe_detection(image,model):
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)#Blue green red to red green blue transformation
    image.flags.writeable=False
    results=model.process(image)
    image.flags.writeable=False
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results

def extract_keypoints(result):
    pose=np.array([[res.x,res.y,res.z,res.visibility] for res in result.pose_landmarks.landmark]).flatten() if result.pose_landmarks else np.zeros(33*4)
    # face=np.array([[res.x,res.y,res.z] for res in result.face_landmarks.landmark]).flatten() if result.face_landmarks else np.zeros(468*3)
    lh=np.array([[res.x,res.y,res.z] for res in result.left_hand_landmarks.landmark]).flatten() if result.left_hand_landmarks else np.zeros(21*3)
    rh=np.array([[res.x,res.y,res.z] for res in result.right_hand_landmarks.landmark]).flatten() if result.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,lh,rh])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    sequence = []
    data = data['images'][0].split(',')
    print("Len of json: ",len(data))
    for i in data:
        img = base64.b64decode(i)
        file = open("frame.jpg", "wb")
        file=file.write(img)
        img = cv2.imread('frame.jpg')
        image,result = mediapipe_detection(img,holistic)
        img = extract_keypoints(result)
        sequence.append(img)
    
    print('length of seq: ',len(sequence))
    images = np.array(sequence)
    pre = model.predict(np.expand_dims(images, axis=0))
    print("Prediction: ",pre)
    res = np.argmax(pre)
    print("Result: ",res)
    return labels[res]


app.run(port=5000, debug=False)