from flask import Flask,render_template,request
# import the necessary packages
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
#the model is prebuild so we just need to load it

app =  Flask(__name__)
@app.route('/', methods= ['GET'])
def hello():
    return render_template("index.html")

@app.route('/', methods= ['post'])
def predict():
    model= VGG16()
    imagefile= request.files['imagefile']
    image_path= "./images/"+ imagefile.filename
    imagefile.save(image_path)

    image= load_img(image_path, target_size=(224,224))
    image= img_to_array(image)
    image= image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image= preprocess_input(image)
    yhat= model.predict(image)
    label= decode_predictions(yhat)
    label= label[0][0]
    classification= "%s (%.2f%%)" % (label[1], label[2]*100)
    return render_template("index.html", prediction= classification, image_path= image_path)
if __name__=='__main__':
    app.run(port= 3000, debug= True)
