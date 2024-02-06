from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np 
import tensorflow as tf 
import io
import cv2 as cv


app = FastAPI(title='Dog Cat Classifier')


model = tf.keras.models.load_model("classifier.h5", compile=False)

def preprocess_img(pet):
    pet_array = np.array(pet)
    print("Type of pet variable:", type(pet_array))
    print("Shape of pet variable:", pet_array.shape)
    img_array = pet.resize((224, 224)) 
    img_array = tf.keras.utils.img_to_array(img_array)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array / 255.0


@app.get("/")
async def root():
    return {"message": "Hello World!"}


@app.post("/upload-image/")
async def upload_image(file: UploadFile):
    contents = await file.read()
    pet = Image.open(io.BytesIO(contents))

    processed_image = preprocess_img(pet)

    prediction = model.predict(processed_image)

    prediction_label = "dog" if prediction[0][0] > 0.5 else "cat"

    
    return JSONResponse(content={"prediction": prediction_label,
                                 "confidence": prediction[0][0] * 100})
