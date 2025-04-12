import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import io
import uvicorn

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

# Initialize FastAPI
app = FastAPI()

# Load the trained model
model = load_model("colon.h5")  # Ensure colon.h5 is in the same directory

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Preprocessing settings
datagen = ImageDataGenerator(rescale=1.0 / 255)
class_labels = ["Colon_adenocarcinoma", "Colon_benign_tissue"]
img_size = (224, 224)
batch_size = 1

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    temp_image_path = "temp_image.jpg"
    image.save(temp_image_path)

    # Temp dataframe to simulate original training setup
    temp_df = pd.DataFrame({
        "filepaths": [temp_image_path],
        "labels": ["unknown"]
    })

    generator = datagen.flow_from_dataframe(
        dataframe=temp_df,
        x_col="filepaths",
        y_col="labels",
        target_size=img_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False,
        batch_size=batch_size
    )

    preprocessed_img = next(generator)

    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = class_labels[predicted_class_index]

    prediction_probabilities = {
        class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))
    }

    return {
        "predicted_class": predicted_class,
        "prediction_probabilities": prediction_probabilities
    }

# REMOVE this block when deploying to Railway or Render
# You only need it if running locally for testing
# Use "uvicorn main:app --host 0.0.0.0 --port 8106" as the start command instead

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8106)
