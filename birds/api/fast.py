from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from birds.predict import generate_mel_spectrogram_prediction, get_top_predictions_dict, get_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
@app.get("/")
def index():
    return {"ok": True}

@app.get("/predict")
def predict(audiofile_path):
    
    # Generate mel spectrogram and get model
    spectrogram = generate_mel_spectrogram_prediction(audiofile_path)
    model = get_model('model/model_densenet169_v1')
    
    # Get predictions and return top 3 predictions
    dict_predict = get_top_predictions_dict(spectrogram, model)
    
    return dict_predict

@app.post("/files/")
async def create_file(file: bytes = File(...)):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Generate mel spectrogram and get model
    spectrogram = generate_mel_spectrogram_prediction(file.filename)
    model = get_model('model/model_densenet169_v1')
    
    # Get predictions and return top 3 predictions
    dict_predict = get_top_predictions_dict(spectrogram, model)
    
    return dict_predict

if __name__=="__main__":
    dict_predict = predict('raw_data/data_10s/test/Hirundo-rustica-157282_tens.ogg')
    print(dict_predict)