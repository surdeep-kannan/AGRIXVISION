# =========================================================================================
# === AgriSmart AI Backend (v28) with Multilingual/Tanglish Chatbot ===
# =========================================================================================

# --- 1. Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, joblib, pandas as pd, ee, re, time, json
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from datetime import datetime

# TensorFlow / Keras for Plant Disease
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications import MobileNetV2
import numpy as np

# LangChain for QA Chatbot
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain

# --- 2. FastAPI Init ---
app = FastAPI(title="AgriSmart AI Backend")
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
os.makedirs("uploads", exist_ok=True)

# --- 3. Load Models ---
yield_model = None
geolocator = None
plant_disease_model = None
CHROMA_DB_DIR = "db"

# Initialize Earth Engine
try:
    ee.Initialize(project='agrix-469307')
    geolocator = Nominatim(user_agent="agri_smart_app_v28")
    if os.path.exists("yield_model.pkl"):
        yield_model = joblib.load("yield_model.pkl")
    print("Yield prediction and geolocation initialized successfully.")
except Exception as e:
    print(f"Error initializing services: {e}")

# Plant disease model (21 classes)
try:
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(224,224,3))
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(224,224,3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(21, activation="softmax")(x)
    plant_disease_model = Model(inputs, outputs)
    if os.path.exists("plant_disease_model.keras"):
        plant_disease_model.load_weights("plant_disease_model.keras")
        print("Plant disease model loaded (21 classes).")
except Exception as e:
    print(f"Plant disease model error: {e}")
    plant_disease_model = None

plant_disease_class_names = [
    "Apple__Apple_scab","Apple__Black_rot","Apple__Cedar_apple_rust","Apple__healthy",
    "Cherry__healthy","Corn__Common_rust_","Corn__Northern_Leaf_Blight","Corn__healthy",
    "Grape__Black_rot","Grape__Esca_Black_Measles","Grape__healthy","Peach__Bacterial_spot",
    "Peach__healthy","Pepper_bell__Bacterial_spot","Pepper_bell__healthy","Potato__Early_blight",
    "Potato__Late_blight","Potato__healthy","Strawberry__healthy","Tomato__Bacterial_spot",
    "Tomato__Early_blight"
]

# --- 4. Load Knowledge Base VectorStore ---
embeddings = OllamaEmbeddings(model="llama3")
vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

chatbot_chain = ConversationalRetrievalChain.from_llm(
    llm=OllamaLLM(model="llama3"),
    retriever=retriever,
    return_source_documents=True,
    verbose=False
)

# --- 5. Helper Functions ---
def parse_coordinates(location_str: str):
    try:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", location_str)
        if len(numbers) >= 2: return float(numbers[0]), float(numbers[1])
    except: return None, None
    return None, None

def get_environmental_data(aoi, start_date, end_date):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            elevation_image = ee.Image('USGS/SRTMGL1_003').rename('elevation')
            temp_collection = ee.ImageCollection('MODIS/061/MOD11A1').filterDate(start_date, end_date).select('LST_Day_1km')
            mean_temp_image = temp_collection.map(lambda img: img.multiply(0.02).subtract(273.15)).mean().rename('mean_temp')
            rainfall_collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY').filterDate(start_date, end_date)
            total_rainfall_image = rainfall_collection.sum().rename('total_rainfall')
            combined_image = ee.Image.cat([elevation_image, mean_temp_image, total_rainfall_image])
            mean_data = combined_image.reduceRegion(ee.Reducer.mean(), geometry=aoi, scale=100, maxPixels=1e9).getInfo()
            return mean_data
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {}

# --- 6. Yield Prediction Endpoint ---
@app.post("/process-and-predict")
async def process_and_predict(location: str, audio: UploadFile = File(...)):
    if yield_model is None or geolocator is None:
        raise HTTPException(status_code=500, detail="Yield prediction service not ready.")
    
    filepath = os.path.join("uploads", audio.filename)
    with open(filepath, "wb") as buffer:
        buffer.write(await audio.read())

    lat, lon = parse_coordinates(location)
    if lat is None or lon is None:
        raise HTTPException(status_code=400, detail="Could not parse location.")
    
    try:
        aoi = ee.Geometry.Point(lon, lat).buffer(20000)
        env_data = get_environmental_data(aoi, f"{datetime.now().year-1}-06-01", f"{datetime.now().year-1}-10-31")
        features_df = pd.DataFrame([{'mean_ndvi': env_data.get('NDVI', 0.7), 'total_rainfall_mm': env_data.get('total_rainfall', 100)}])
        predicted_yield = yield_model.predict(features_df)
        return {"predicted_yield_kg_per_hectare": float(predicted_yield[0]), "env_data": env_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 7. Plant Disease Prediction Endpoint ---
@app.post("/predict-disease")
async def predict_disease(image_file: UploadFile = File(...)):
    if plant_disease_model is None: raise HTTPException(status_code=503, detail="Plant disease model not ready.")
    try:
        contents = await image_file.read()
        img = tf.image.decode_image(contents, channels=3)
        img = tf.image.resize(img, [224,224])
        img = img/255.0
        pred = plant_disease_model.predict(np.expand_dims(img,0))
        idx = int(np.argmax(pred[0]))
        return {"predicted_class": plant_disease_class_names[idx], "confidence": float(np.max(pred[0]))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 8. Conversational Chatbot Endpoint ---
class ChatRequest(BaseModel):
    user_id: str
    question: str

@app.post("/ask-chatbot")
async def ask_chatbot(req: ChatRequest):
    try:
        result = chatbot_chain({"question": req.question, "chat_history": []})
        answer = result.get("answer", "")
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
