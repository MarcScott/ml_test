from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import wikipedia as wiki

model = tf.keras.applications.VGG16()
size = 224, 224
IMAGE_SIZE = size[0]

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/')
def upload_f():
   return render_template('upload.html')

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      file_location = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(f.filename))
      f.save(f"static/uploads/{f.filename}")
      resize_image(f"static/uploads/{f.filename}")
      classification = identify_image(f"static/uploads/{f.filename}")
      print(classification)
      title, details, images = get_info(classification)
      return f'''
      <h1>{title}</h1>
      <img src="static/uploads/{f.filename}" width=500>
      <p>T{details}</p>
      <img src={images[0]} width = 500>
      '''

def resize_image(image_path):
    im = Image.open(image_path)
    im = im.convert('RGB')
    im.thumbnail(size)
    im.save(image_path, "JPEG")

def get_info(classification):
    article = wiki.page(classification)
    title = article.title
    details = article.summary
    images = article.images
    return(title, details, images)

def identify_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path,
    target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    classifications = model.predict(image, batch_size=1)
    best_classification = tf.keras.applications.imagenet_utils.decode_predictions(classifications, top=1)
    return best_classification[0][0][1]

app.run(debug = True)
