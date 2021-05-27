import tensorflow as tf
from tensorflow.python.ops.signal import spectral_ops
import tensorflow_io as tfio
import numpy as np


TARGET_SAMPLE_RATE = 16_000
TARGET_SPLIT_DURATION_SEC = 10 # For optional call to function split tensor


def convert_audio_file_to_audio_tensor(filepath):
    '''
    Objective : import audio file as audio tensor with tensorflow IO
    Input : file path
    Output : AudioTensor
    '''
    return tfio.audio.AudioIOTensor(filepath, dtype='float32')


def convert_to_tensor(audio):
    '''
    Objective : convert AudiioIOTensor to tf Tensor and keep track of audio sample rate
    Input : AudioIOTensor
    Output : tf.Tensor and audio sample rate (Hz - int)
    '''
    return audio[:], audio.rate.numpy()


def resample_audio_tensor(tensor, input_rate, output_rate=TARGET_SAMPLE_RATE):
    '''
    Objective : resample audio file to 16_000 hz
    Input : tf.Tensor, original audio sample rate (Hz - int), output rate
    Output : tf.Tensor resampled in 16_000 hz
    '''
    
    return tfio.audio.resample(tensor, input_rate, output_rate, name=None)


# Optional
def split_tensor(tensor, audio_rate=TARGET_SAMPLE_RATE):
    '''
    Objective : split to keep only 160 k datapoints (10 seconds with audio sample rate 16k)
    Input : tf.Tensor, audio sample rate
    Output : tf.Tensor
    '''
    # Get split index adequate to audio rate
    split_index = audio_rate * TARGET_SPLIT_DURATION_SEC
    
    # Split if audio length > split_index
    if tensor.shape[0] > split_index :
        return tensor[:split_index]
    return tensor[:]

def harmonize_tensor_shape(tensor):
    '''
    Objective : Harmonize tensor shape and dtype of audio to shape (x,),dtype=float32
    Input : tf.Tensor shape (x,2) stereo or (x,1) mono
    Output : tf.Tensor shape(x,)
    '''
    
    # Convert to float32 dtype if necessary
    # if tensor.dtype == tf.int16 :
    #     tensor = tf.cast(tensor, tf.float32) / 32768.0
        
    if tensor.shape[1] == 2 :                               # If stereo, convert to mono 
        return tf.reduce_mean(tensor, 1)                    # and remove last dimension
    
    return tf.squeeze(tensor, axis=[-1])                    # If mono, remove last dimension



def generate_spectrogram(file_path, label='label',                      # audio file params
                         split=True, output_rate=TARGET_SAMPLE_RATE,    # preprocessing params  
                         transpose=True,                                # output params
                         nfft=2048, window=256, stride=256):            # spectrogram params
    '''
    Objective : Generate spectrogram from an audio file path
    Input : file_path , label = integer between 0 and 49
    Ouput : Spectrogram tf.Tensor shape (x,y), harmonizedtensor, label, input_rate, output_rate
    '''    
    
    audio_tensor = convert_audio_file_to_audio_tensor(file_path)            # Get Audio tensor from file
    tensor, input_rate = convert_to_tensor(audio_tensor)                    # Convert to tf.tensor
    if input_rate != output_rate:                                           # Resample to TARGET_SAMPLE_RATE
        tensor = resample_audio_tensor(tensor, input_rate, output_rate)
    if split==True:                                                         # Split to keep only first 10 sec.
        tensor = split_tensor(tensor)
    harmonizedtensor = harmonize_tensor_shape(tensor)                       # Harmonize to get mono and float32 dtype tensor

    spectrogram = tfio.audio.spectrogram(                                   # Generate spectrogram
        harmonizedtensor,
        nfft=nfft,
        window=window,
        stride=stride)
    
    if transpose == True:                                                   # Transpose output if asked
        spectrogram = tf.transpose(spectrogram, perm=[1, 0])
    
    return spectrogram, harmonizedtensor, label, input_rate, output_rate


def generate_mel_spectrogram(file_path, label='label',                                  # audio file params
                             split=True, output_rate=TARGET_SAMPLE_RATE,                # preprocessing params
                             transpose=True,                                            # output params
                             nfft=2048, window=256, stride=256,                         # spectrogram params
                             rate=TARGET_SAMPLE_RATE, mels=128, fmin=0, fmax=8000):     # mel spectrogram params
    '''
    Objective : Generate mel spectrogram from an audio file path
    Input : file_path , label = integer between 0 and 49
    Ouput : Spectrogram tf.Tensor shape (x,y), harmonizedtensor, label, input_rate, output_rate
    '''
    
    spectrogram, harmonizedtensor, label, input_rate, output_rate = generate_spectrogram(     # Generate non transposed spectrogram
        file_path, label=label, split=split, output_rate=output_rate,
        transpose=False,
        nfft=nfft, window=window, stride=stride)
    
    mel_spectrogram = tfio.audio.melscale(spectrogram,                                          # Convert to mel_spectrogram
                                          rate=rate, mels=mels, fmin=fmin,fmax=fmax)
    
    if transpose == True:                                                                       # Transpose output if asked
        mel_spectrogram = tf.transpose(mel_spectrogram, perm=[1, 0])

    return mel_spectrogram, harmonizedtensor, label, input_rate, output_rate


def generate_db_scale_mel_spectrogram(file_path, label='label', split=True,                     # audio file params
                                      transpose=True,                                           # output params
                                      nfft=2048, window=256, stride=256,                        # spectrogram params
                                      rate=TARGET_SAMPLE_RATE, mels=128, fmin=0, fmax=8000,     # mel spectrogram params
                                      top_db=80):                                               # db scale mel spectrogram params
    '''
    Objective : Generate db scale mel spectrogram from an audio file path
    Input : file_path , label = integer between 0 and 49
    Ouput : Spectrogram tf.Tensor shape (x,y), harmonizedtensor, label, input_rate, output_rate
    '''
    
    mel_spectrogram, harmonizedtensor, label, input_rate, output_rate = generate_mel_spectrogram(     # Generate non transposed mel spectrogram
        file_path, label=label, split=split,
        transpose=False,
        nfft=nfft, window=window, stride=stride,
        rate=rate, mels=mels, fmin=fmin, fmax=fmax)
    
    db_scale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram,                              # Convert to db scale mel spectrogram
                                                  top_db=top_db)
    
    if transpose == True:                                                                       # Transpose output if asked
        db_scale_mel_spectrogram = tf.transpose(db_scale_mel_spectrogram, perm=[1, 0])
    
    return db_scale_mel_spectrogram, harmonizedtensor, label, input_rate, output_rate


if __name__=="__main__":
    # Test fonctions unitaires
    file_path = "raw_data/data_10s/train/Troglodytes-troglodytes-463329_tens.ogg"
    audio_tensor = convert_audio_file_to_audio_tensor(file_path)
    assert audio_tensor.rate.numpy() == 16000
    tensor, audio_rate = convert_to_tensor(audio_tensor)
    assert audio_rate == 16000
    tensor = resample_audio_tensor(tensor, audio_rate)
    split = split_tensor(tensor)
    assert split.shape[0] == 441000
    harmonizedtensor = harmonize_tensor_shape(split)
    assert harmonizedtensor.shape[0] == 441000
    spectrogram, harmonizedtensor, label, input_rate, output_rate = generate_spectrogram(file_path)
    mel_spectrogram, harmonizedtensor, label, input_rate, output_rate = generate_mel_spectrogram(file_path)
    db_scale_mel_spectrogram, harmonizedtensor, label, input_rate, output_rate = generate_db_scale_mel_spectrogram(file_path)
    
    # Test full intégré
    label = 2
    spectro, harmonizedtensor, label2, input_rate, output_rate  = generate_spectrogram(file_path, label)
    assert label2 == 2
