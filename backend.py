from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import tensorflow as tf 
import io
from fastapi.middleware.cors import CORSMiddleware

#create an API instance
app = FastAPI(title='Dog Cat Classifier')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

#load saved model
model = tf.keras.models.load_model("classifier.h5", compile=False)

#preproces uploaded image
def preprocess_img(pet):
    img_array = pet.resize((224, 224)) 
    img_array = tf.keras.utils.img_to_array(img_array)
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array / 255.0


@app.get("/")
async def root():
    return {"message": "Hello World!"}

#post method to upload image and predict classes
@app.post("/upload-image/")
async def upload_image(file: UploadFile):
    contents = await file.read()
    pet = Image.open(io.BytesIO(contents))

    processed_image = preprocess_img(pet)

    prediction = model.predict(processed_image)

    prediction_label = "dog" if prediction[0][0] > 0.5 else "cat"

    
    return JSONResponse(content={"prediction": prediction_label,
                                 "confidence": prediction[0][0] * 100})
