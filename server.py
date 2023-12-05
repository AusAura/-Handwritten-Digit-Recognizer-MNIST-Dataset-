from flask import Flask, jsonify, request, render_template, url_for
from keras.models import load_model
import numpy as np
import io
from PIL import Image
from skimage import transform

def resize_image_skimage(img, target_size=(28, 28)):
    resized_img = transform.resize(img, target_size, mode='constant')
    print('done resize')
    return resized_img

# def resize_image_cv2(img, target_size=(28, 28)):
#     resized_img = cv2.resize(img, target_size)
#     return resized_img

app = Flask(__name__)

model = load_model('mnist-recognize.h5')


def preprocess_image(img):
    # Ensure that the input is a PIL Image
    if not isinstance(img, Image.Image):
        img = Image.open(io.BytesIO(img))

    img_array = np.array(img)
    img_array = resize_image_skimage(img_array)
    print('resized in preproc')
    resized_img = Image.fromarray((img_array * 255).astype(np.uint8))
    print('got image back')
    # img_array = img_array.reshape(1, 28, 28, 1) / 255.0
    img_array = img_array.reshape(1, 784) / 255.0
    # print(img_array)
    return img_array, resized_img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        for n in request.files:
            name = n
            print(n)

        # Получение файла из запроса
        file = request.files[name]
        
    
        img_data = file.read()
        img = Image.open(io.BytesIO(img_data))
        print('open')
        img = img.convert('L')
        print('convert')
        # img = img.resize((28, 28), Image.LANCZOS) ## Bleak

        
        # Предобработка изображения
        print('stage 1')
        processed_data, resized_img = preprocess_image(img)
        print('processed')

        img_filename = 'tmp/' + name + '.jpg'
        print(img_filename)
        resized_img.save('static/' + img_filename)
        print('stage 2')
        # img_url = url_for('static', filename=img_filename) ## the same
        img_url = 'static/' + img_filename
        
        # Предсказание с использованием модели
        prediction = model.predict(processed_data)
        print(prediction)
        label = str(np.argmax(prediction))
        print(label)
        
        # Отправка предсказания
        print('stage 3')
        return render_template('index.html', img=img_url, prediction=label)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000)