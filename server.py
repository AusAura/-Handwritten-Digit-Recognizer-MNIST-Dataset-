from flask import Flask, jsonify, request, render_template, url_for
from keras.models import load_model
import numpy as np
import io
from PIL import Image
import cv2

app = Flask(__name__)
model = load_model('mnist-recognize.h5.old')

## this image reading and preprocessing function is to be tested yet
# def cv2_read_image(img):
#     img = cv2.imread('example.jpg')
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## to grayscale
#     _, black_and_white_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY) ## threshold prepr for black/white
#     cv2.imwrite('example_black_and_white.jpg', black_and_white_img) ## saving image

def resize_image_cv2(img, target_size=(28, 28)):
    resized_img = cv2.resize(255-img, target_size, interpolation=cv2.INTER_AREA) #resizing and inverting, 255-img to increase efficiency
    return resized_img

def preprocess_image(img):
    # Ensure that the input is a PIL Image
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img))

    img_array = np.array(img)
    img_array = resize_image_cv2(img_array)
    # print('resized in preproc')
    resized_img = Image.fromarray((img_array * 255).astype(np.uint8)) ## reverse convertation for testing
    # print('got image back')
    img_array = img_array.reshape(1, 28, 28, 1) / 255.0 ## for 1st model, prepr for ANN input
    return img_array, resized_img

## Predicting route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # get the name
        for n in request.files:
            name = n

        file = request.files[name] ## get the file
        img_data = file.read()
        img = Image.open(io.BytesIO(img_data)) ## converting to PIL Image
        # print('opened')
        img = img.convert('L') ## convert to grayscale
        # print('convert')

        ## saving to tmp original image
        img_filename = 'tmp/' + name + '_orig' + '.jpg'
        img.save('static/' + img_filename)
        img_url = url_for('static', filename=img_filename) 
        # img_url = 'static/' + img_filename ## the same

        # PREPROCESSING
        # print('stage 1 done - now preprocessing')
        processed_data, resized_img = preprocess_image(img) ## also returning reverse-converted image for visual checkout
        # print('processed')

        ## saving to tmp processed image
        img_filename = 'tmp/' + name + '.jpg'
        resized_img.save('static/' + img_filename)
        img_url = url_for('static', filename=img_filename)
        
        # PREDICTION
        # print('stage 2 done - now predicting')
        prediction = model.predict(processed_data)
        label = str(np.argmax(prediction))
        print(prediction)
        # print(label)
        
        # OUTPUT
        # print('stage 3 done - now rendering')
        return render_template('index.html', img=img_url, prediction=label)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)