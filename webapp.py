from flask import Flask ,render_template, request
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage
import cv2
import numpy as np
import os

from flask_cors import CORS, cross_origin



names=["Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust","Apple___healthy","Blueberry___healthy","Cherry_(including_sour)___healthy","Cherry_(including_sour)___Powdery_mildew","Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot","Corn_(maize)___Common_rust_","Corn_(maize)___healthy","Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot","Grape___Esca_(Black_Measles)","Grape___healthy","Grape___Leaf_blight_(Isariopsis_Leaf_Spot)","Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot","Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy","Potato___Early_blight","Potato___healthy","Potato___Late_blight","Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew","Strawberry___healthy","Strawberry___Leaf_scorch","Tomato___Bacterial_spot","Tomato___Early_blight","Tomato___healthy","Tomato___Late_blight","Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot","Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot","Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

# Process image and predict label
def processImg(IMG_PATH):
    # Read image
    model = load_model("C:/Users/DELL/my_model2.h5")
    
    # Preprocess image
    image = cv2.imread(IMG_PATH)
    print(image)
    image = cv2.resize(image, (30, 30))
    #image = image.astype("float") / 255.0
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    res = model.predict(image)
    label = np.argmax(res)
    print("Label", label)
    labelName = names[label-1]
    print("Label name:", labelName)
    return labelName


# Initializing flask application
app = Flask(__name__,template_folder='template')
cors = CORS(app)

@app.route("/")
def main():
    
    return render_template("upload.html")


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      fname=secure_filename(f.filename);
      f.save("static/"+secure_filename(f.filename))
      
      new_path = os.path.abspath(fname)
      resp = processImg(new_path)

         
   return  render_template("upload.html",value=resp,finame="static/"+fname)
     


# About page with render template
@app.route("/about")
def postsPage():
    return render_template("about.html")




if __name__ == "__main__":
    app.run(debug=True)