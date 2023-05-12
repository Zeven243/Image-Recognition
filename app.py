from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)
model = tf.keras.applications.MobileNetV2(weights='imagenet')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Handle the uploaded image
        image = request.files['image']
        image_path = 'static/' + image.filename
        image.save(image_path)

        # Load and preprocess the image
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        # Make predictions
        predictions = model.predict(image)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=5)[0]

        # Prepare labels and probabilities for display
        results = [(label, probability * 100) for (_, label, probability) in decoded_predictions]

        return render_template('index.html', image_path=image_path, results=results)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
