import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
import os


TARGET_SAMPLE_RATE = 16_000
TARGET_SPLIT_DURATION_SEC = 10 


def generate_tensor(file_path, label, 
                    split=True, output_rate=TARGET_SAMPLE_RATE):
    '''
    Objective : Generate tensor from file path and return all steps
    Input : file_path , label = integer between 0 and 49
    Output : tf.Tensor shape(x,), label, input_rate, output_rate
    '''
    
    # 1 - Convert audio file to AudioTensor
    audio_tensor = tfio.audio.AudioIOTensor(file_path, dtype='float32')
    
    # 2 - Convert AudioTensor to tf Tensor and get rate
    tensor = audio_tensor.to_tensor()
    input_rate = tf.cast(audio_tensor.rate, tf.int64)
    
    # 3 - Resample to output_rate
    output_rate = np.int64(output_rate)
    tensor = tfio.audio.resample(tensor, input_rate, output_rate, name=None)
    
    # 4 - TO DO >>> Pad if too short duration
    
    # 5 - Split if too long duration
    if split==True:
        split_index = output_rate * TARGET_SPLIT_DURATION_SEC
        tensor = tensor[:split_index]
        
    # 6 - Harmonize tensor shape >>> TO DO : test if necessary after step 1 ?
    if tensor.dtype == tf.int16:
        tensor = tf.cast(tensor, tf.float32)
    
    # 7 - Convert stereo to mono and remove last dimension
    tensor = tf.reduce_mean(tensor, 1)
    
    return tensor, label, input_rate, output_rate


def generate_spectrogram(file_path, label,
                         split=True, output_rate=TARGET_SAMPLE_RATE,  
                         transpose=True,
                         nfft=2048, window=256, stride=256):
    '''
    Objective : Generate spectrogram from an audio file path
    Input : file_path , label = integer between 0 and 49
    Ouput : Spectrogram tf.Tensor shape (x,y), label
    '''
    
    # 1 - Generate tensor from file path
    tensor, label, input_rate, output_rate = generate_tensor(file_path, label, split=split, output_rate=output_rate)

    # 2 - Generate spectrogram
    spectrogram = tfio.audio.spectrogram(tensor, nfft=nfft, window=window, stride=stride)
    
    # 3 - Transpose output if asked
    if transpose == True:
        spectrogram = tf.transpose(spectrogram, perm=[1, 0])
    
    # 4 - Expand dim to get channel dimension
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    
    return spectrogram, label


def generate_mel_spectrogram(file_path, label,
                             split=True, output_rate=TARGET_SAMPLE_RATE,
                             transpose=True,
                             nfft=2048, window=256, stride=256,
                             rate=TARGET_SAMPLE_RATE, mels=128, fmin=0, fmax=8000):
    '''
    Objective : Generate mel spectrogram from an audio file path
    Input : file_path , label = integer between 0 and 49
    Ouput : Spectrogram tf.Tensor shape (x,y), label
    '''
    
    # 1 - Generate tensor from file path
    tensor, label, input_rate, output_rate = generate_tensor(file_path, label, split=split, output_rate=output_rate)

    # 2 - Generate spectrogram
    spectrogram = tfio.audio.spectrogram(tensor, nfft=nfft, window=window, stride=stride)
    
    # 3 - Convert to mel spectrogram
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=rate, mels=mels, fmin=fmin,fmax=fmax)
    
    # 4 - Transpose output if asked
    if transpose == True:
        mel_spectrogram = tf.transpose(mel_spectrogram, perm=[1, 0])

    # 5 - Expand dim to get channel dimension
    mel_spectrogram = tf.expand_dims(mel_spectrogram, axis=-1)
    
    return mel_spectrogram, label


def generate_db_scale_mel_spectrogram(file_path, label,
                                      split=True, output_rate=TARGET_SAMPLE_RATE,
                                      transpose=True,
                                      nfft=2048, window=256, stride=256,
                                      rate=TARGET_SAMPLE_RATE, mels=128, fmin=0, fmax=8000,
                                      top_db=80):
    '''
    Objective : Generate db scale mel spectrogram from an audio file path
    Input : file_path , label = integer between 0 and 49
    Ouput : Spectrogram tf.Tensor shape (x,y), label
    '''
    
    # 1 - Generate tensor from file path
    tensor, label, input_rate, output_rate = generate_tensor(file_path, label, split=split, output_rate=output_rate)

    # 2 - Generate spectrogram
    spectrogram = tfio.audio.spectrogram(tensor, nfft=nfft, window=window, stride=stride)
    
    # 3 - Convert to mel spectrogram
    mel_spectrogram = tfio.audio.melscale(spectrogram, rate=rate, mels=mels, fmin=fmin,fmax=fmax)
    
    # 4 - Convert to db scale mel spectrogram
    db_scale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=top_db)
    
    # 5 - Transpose output if asked
    if transpose == True:
        db_scale_mel_spectrogram = tf.transpose(db_scale_mel_spectrogram, perm=[1, 0])
    
    # 6 - Expand dim to get channel dimension
    db_scale_mel_spectrogram = tf.expand_dims(db_scale_mel_spectrogram, axis=-1)
    
    return db_scale_mel_spectrogram, label


def one_hot_encode_target(spectrogram, label):
    depth = 50
    ohe_label = tf.one_hot(label, depth)
    return spectrogram, ohe_label


if __name__=="__main__":
    
    # Test generate_spectrogram
    file_path = "raw_data/data_10s/train/Troglodytes-troglodytes-463329_tens.ogg"
    label = 2
    spectro, label = generate_spectrogram(file_path, label)
    assert label == 2
    
    # Test generate dataset
    files = [
        'Glaucidium-passerinum-201176_tens.ogg',
        'Glaucidium-passerinum-408254_tens.ogg',
        'Glaucidium-passerinum-309791_tens.ogg'
        ]
    labels = [1, 2, 3]
    
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_PATH = os.path.join(ROOT_PATH, 'raw_data', 'data_10s', 'train')
    
    files = [os.path.join(DATA_PATH, file) for file in files]
    
    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(generate_spectrogram)
    dataset = dataset.batch(1)
    
    print(next(iter(dataset)))