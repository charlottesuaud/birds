from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from birds.predict import generate_mel_spectrogram_prediction, get_top_predictions_dict, get_model

import shutil

app = FastAPI()
model = get_model('model/model_densenet169_v1') # load model when starting API to avoid waiting at each prediction

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


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    
    # Create generic 'ouput' + extension filename to avoid writing too many files on disk
    # As model can handle severeal audio file types we retrieve the extension form provided filename
    filename = 'output.' + str(file.filename)[-3:]
    print(filename)
    with open(filename,'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Generate mel spectrogram
    spectrogram = generate_mel_spectrogram_prediction(filename)
    
    # Get predictions and return top 3 predictions
    dict_predict = get_top_predictions_dict(spectrogram, model)
    
    # FastApi can only manage python object and prediction were np.float32 so we convert back to python float in ouput dictionnary
    return { k:float(v) for k, v in dict_predict.items() }