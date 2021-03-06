import tensorflow_io as tfio
import numpy as np
import tensorflow as tf

TARGET_SAMPLE_RATE = 16_000
TARGET_SPLIT_DURATION_SEC = 10

TARGET_DICT = {'Sonus naturalis': 0,
               'Fringilla coelebs': 1,
               'Parus major': 2,
               'Turdus merula': 3,
               'Turdus philomelos': 4,
               'Sylvia communis': 5,
               'Emberiza citrinella': 6,
               'Sylvia atricapilla': 7,
               'Emberiza calandra': 8,
               'Phylloscopus trochilus': 9,
               'Luscinia megarhynchos': 10,
               'Strix aluco': 11,
               'Phylloscopus collybita': 12,
               'Carduelis carduelis': 13,
               'Erithacus rubecula': 14,
               'Chloris chloris': 15,
               'Sylvia borin': 16,
               'Acrocephalus arundinaceus': 17,
               'Acrocephalus dumetorum': 18,
               'Oriolus oriolus': 19,
               'Troglodytes troglodytes': 20,
               'Bubo bubo': 21,
               'Ficedula parva': 22,
               'Linaria cannabina': 23,
               'Luscinia svecica': 24,
               'Alauda arvensis': 25,
               'Luscinia luscinia': 26,
               'Phoenicurus phoenicurus': 27,
               'Aegolius funereus': 28,
               'Cyanistes caeruleus': 29,
               'Hirundo rustica': 30,
               'Emberiza cirlus': 31,
               'Locustella naevia': 32,
               'Cuculus canorus': 33,
               'Sylvia curruca': 34,
               'Loxia curvirostra': 35,
               'Emberiza hortulana': 36,
               'Carpodacus erythrinus': 37,
               'Athene noctua': 38,
               'Crex crex': 39,
               'Acrocephalus schoenobaenus': 40,
               'Acrocephalus palustris': 41,
               'Periparus ater': 42,
               'Phylloscopus sibilatrix': 43,
               'Emberiza schoeniclus': 44,
               'Hippolais icterina': 45,
               'Pyrrhula pyrrhula': 46,
               'Caprimulgus europaeus': 47,
               'Ficedula hypoleuca': 48,
               'Glaucidium passerinum': 49}

REVERSE_DICT = {value : key for (key, value) in TARGET_DICT.items()}

def generate_mel_spectrogram_prediction(file_path, 
                                    output_rate=TARGET_SAMPLE_RATE,
                                    transpose=True,
                                    nfft=400, window=400, stride=100,                      # spectrogram params
                                    rate=TARGET_SAMPLE_RATE, mels=128, fmin=0, fmax=8000): # mel spectrogram params
    '''
    Objective : Generate spectrogram from an audio file path
    Input : file_path , label = integer between 0 and 49
    Ouput : Spectrogram tf.Tensor shape (x,y)
    '''
    # 1 - Generate tensor from file path
    ## a) create AudioTensor
    audio_tensor = tfio.audio.AudioIOTensor(file_path, dtype='float32')
    print(f'Audio tensor generated : shape {audio_tensor.shape}')

    ## b) convert AudioTensor to tf Tensor and get rate
    tensor = audio_tensor.to_tensor()
    input_rate = tf.cast(audio_tensor.rate, tf.int64)
    print(f'Audio tensor converted to tensor : shape {tensor.shape}')

    ## c) resample to output_rate
    output_rate = np.int64(output_rate)
    tensor = tfio.audio.resample(tensor, input_rate, output_rate, name=None)
    print(f'Tensor resampled : shape {tensor.shape}')

    ## d) pad if too short -> not necessary
    
    ## e) split syst??matique ?? 10sec car c'est la dur??e d'entra??nement
    split_index = output_rate * TARGET_SPLIT_DURATION_SEC
    tensor = tensor[:split_index]
    print(f'Tensor splited at 10s : shape {tensor.shape}')

    ## f) harmonize tensor shape
    if tensor.dtype == tf.int16:
        tensor = tf.cast(tensor, tf.float32)
    
    ## g) convert stereo to mono and remove last dimension
    tensor = tf.reduce_mean(tensor, 1)
    print(f'Tensor converted to mono output : shape {tensor.shape}')

    # 2 - Generate spectrogram
    spectrogram = tfio.audio.spectrogram(tensor, nfft=nfft, window=window, stride=stride)
    print(f'Spectrogram generated from tensor : shape {spectrogram.shape}')
    
    # 3 - Convert to mel spectrogram
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=rate, mels=mels, fmin=fmin,fmax=fmax)
    print(f'Mel_spectrogram generated from spectrofram : shape {mel_spectrogram.shape}')

    # 3 - Transpose output if asked
    if transpose == True:
        mel_spectrogram = tf.transpose(mel_spectrogram, perm=[1, 0])
    print(f'Mel_spectrogram transposed : shape {mel_spectrogram.shape}')
    
    # 4 - Expand dim to get channel dimension
    mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=-1)
    print(f'Mel_spectrogram expanded with channel : shape {mel_spectrogram.shape}')

    # 5 - Convert gray to RGB (requested shape for densenet)
    mel_spectrogram = tf.image.grayscale_to_rgb(mel_spectrogram)
    print(f'Mel_spectrogram expanded with 3 channels to mimic RGB : shape {mel_spectrogram.shape}')

    # 6 - Expand dim to have similar shape as model was trained in batches
    mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=0)
    print(f'Mel_spectrogram expanded with 1 dim as training was done in batch : shape {mel_spectrogram.shape}')

    return mel_spectrogram


def get_top_predictions_dict(spectrogram, model):
    '''
    Objective : Generate dictionnary with top 3 predictions from model
    Input : spectrogram, model
    Ouput : dict
    '''
    # Get prediction array (len : 50 -> number of classes)
    prediction = model.predict(spectrogram)[0]
    print('Prediction obtained from model')
    # Retrieve top 3 predictions with associated values
    top3_pred_indexes = prediction.argsort()[-3:]
    #top3_pred_indexes = np.argpartition(prediction, -3)[-3:]
    top3_pred_values = prediction[top3_pred_indexes]
    print('Top 3 predictions retrieved')

    # Convert target number back into scientific name
    top3_pred_names = [REVERSE_DICT[k] for k in top3_pred_indexes]

    # Associate result in a dictionnary for API output
    dico_top3 = dict(zip(top3_pred_names,top3_pred_values))
    print(dico_top3)
    return dico_top3


def get_model(path_to_model):
    print('Begin model loading !')
    model = tf.keras.models.load_model(path_to_model)
    print('Model loaded from file !')
    return model


if __name__=="__main__":
    path_to_model = 'model/model_densenet169_v1'
    filepath = 'raw_data/data_10s/test/Hirundo-rustica-157282_tens.ogg'
    model = get_model(path_to_model)
    spectrogram = generate_mel_spectrogram_prediction(filepath)
    dico_pred = get_top_predictions_dict(spectrogram, model)
    print(dico_pred)