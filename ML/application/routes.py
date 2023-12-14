from application import app
from flask import redirect, render_template, url_for, request,session
import secrets
import os
import cv2
from flask import jsonify
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

model_dict = pickle.load(open(app.config['MODEL_PATH_RFC'], 'rb'))
model = model_dict['model']



mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

@app.route('/')
def index():
    return 'API and CI/CD is it running though?'


@app.route('/sign_language_translation/upload_image/<user>', methods=['POST'])
def upload_image(user):

    # sentence= ""
    if request.method == 'POST' and 'file' in request.files:
        file = request.files['file']
        filename = secure_filename(file.filename)
        extension = filename.split(".")[-1]
        
        save_path = os.path.join(app.config['UPLOADED_IMAGE'], user)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        new_filename = str(len(os.listdir(save_path))) + f".{extension}"
        
        file.save(os.path.join(save_path, new_filename))
        
        response_data = {
            "response": "upload successfully",
        }

        return jsonify(response_data)
    
@app.route('/sign_language_translation/predict/<user>', methods=['POST'])
def predict_sign_language(user):
    # auto_crop_image
    x_ = []
    y_ = []
    crop_dir = os.path.join(app.config['CROPED_IMAGE'], user)
    raw_dir = os.path.join(app.config['UPLOADED_IMAGE'], user)
    if not os.path.exists(crop_dir):
            os.makedirs(crop_dir)
            
    counter = 0
    
    for img_path in os.listdir(os.path.join(raw_dir)):
        data_aux = []
        img= cv2.imread(os.path.join(raw_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    H, W, _ = img.shape
                    x_.append(x)
                    y_.append(y)
                
                x1 =int(min(x_) * W) 
                y1 =int(min(y_) * H) 
                x2 =int(max(x_) * W) 
                y2 =int(max(y_) * H) 
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                
                if (x2 - x1) > (y2 - y1):
                    y2 = y1 + (x2 - x1)
                    y2 = int(0.75 * y2 ) 
                    y1 = int((y1 - (0.25 * y2) ))     
                
                else:
                    x2 = x1 + (y2 - y1)
                    x2 = int(0.7 * x2)
                    x1 = int((x1 - (0.25 * x2)))
                
                x1=int(x1-(x1*0.4))
                x2=int(x2*1.4)
                y1=int(y1-(y1*0.4))
                y2=int(y2*1.4)
             
                hand_crop = img_rgb[y1:y2, x1:x2]
                cv2.waitKey(25)
                cv2.imwrite(os.path.join(crop_dir, '{}.jpg'.format(counter)),  cv2.cvtColor(hand_crop, cv2.COLOR_RGB2BGR))
        counter +=1
    # predict image
    model = load_model(app.config["MODEL_PATH"],compile=False)

    files = os.listdir(crop_dir)
    characters = []
    
    input_size = (64,64)

    labels = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N',
            'O','P','Q','R','S','T','U','V','W','X','Y','Z']
    
    def preprocess(img, input_size):
        nimg = img.convert('L').resize(input_size, resample=Image.BICUBIC)
        img_arr = np.array(nimg) / 255.0
        return img_arr

    def reshape(imgs_arr):
        return np.stack(imgs_arr, axis=0)

    for data in files:
        if data.endswith('.jpg') or data.endswith('.png'):
            print(data)
            imgg = Image.open(os.path.join(crop_dir, data))
            X = preprocess(imgg,input_size)
            X = reshape([X])
            y = model.predict(X)
            print( labels[np.argmax(y)], np.max(y) )
            characters.append(labels[np.argmax(y)])
    separated = ""
    word_cnn = separated.join(characters)
    
    labels =  [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    
    
    model_dict = pickle.load(open(app.config['MODEL_PATH_RFC'], 'rb'))
    model = model_dict['model']

    characters_rfc = []
    for img_path in os.listdir(os.path.join(raw_dir)):
        data_aux = []   
        img= cv2.imread(os.path.join(raw_dir, img_path))
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)
        prediction_proba = model.predict_proba([np.asarray(data_aux)[:42]])
        characters_rfc.append(labels[np.argmax(prediction_proba)])
        
    word_rfc = separated.join(characters_rfc)
    
    
            

    
    
    
    response_data = {
        "cnn": word_cnn,
        "rfc" : word_rfc
    }

    return jsonify(response_data)


    


