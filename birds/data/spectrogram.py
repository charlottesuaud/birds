import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

TARGET_SAMPLE_RATE = 44_100

TARGET_SPLIT_DURATION_SEC = 10 # For optional call to function split tensor


def convert_audio_file_to_audio_tensor(filepath):
    '''
    Objective : import audio file as audio tensor with tensorflow IO
    Input : file path
    Output : AudioTensor
    '''
    return tfio.audio.AudioIOTensor(filepath)

def convert_to_tensor(audio):
    '''
    Objective : convert AudiioIOTensor to tf Tensor and keep track of audio sample rate
    Input : AudioIOTensor
    Output : tf.Tensor and audio sample rate (Hz - int)
    '''
    return audio[:], audio.rate.numpy()


# Optional if audio sampling rate != 44,1  kHz
def resample_audio_tensor(tensor, input_audio_rate):
    '''
    Objective : resample audio file to 44_100 hz
    Input : tf.Tensor and original audio sample rate (Hz - int)
    Output : tf.Tensor resampled in 44_100 hz
    '''
    return tfio.audio.resample(tensor, input_audio_rate, TARGET_SAMPLE_RATE, name=None)


# Optional
def split_tensor(tensor,audio_rate=TARGET_SAMPLE_RATE):
    '''
    Objective : split to keep only 441 k datapoints (10 seconds with audio sample rate 44k)
    Input : tf.Tensor, audio sample rate
    Output : tf.Tensor
    '''
    # Get split index adequate to audio rate
    split_index = audio_rate * TARGET_SPLIT_DURATION_SEC
    # Split if audio length > split_index
    if tensor.shape[0] > split_index :
        return tensor[:split_index]
    return tensor[:]

def harmonize_tensor_shape(audio):
    '''
    Objective : Harmonize tensor shape and dtype of audio to shape (x,),dtype=float32
    Input : tf.Tensor shape (x,2) stereo or (x,1) mono
    Output : tf.Tensor shape(x,)
    '''
    # Convert to float32 dtype if necessary
    if audio.dtype == tf.int16 :
        audio = tf.cast(audio, tf.float32) / 32768.0
    # Convert stero to mono if adequate :
    if audio.shape[1] == 2 :
        return tf.reduce_mean(audio, 1)
    # Remove last dimension if mono sound
    return tf.squeeze(audio, axis=[-1])


def generate_spectrogram(audio,nfft=2048,window=256,stride=256):
    '''
    Objective : Generate spectrogram
    Input : Audio tf.Tensor shape(x,)
    Ouput : Spectrogram tf.Tensor shape (x,y)
    '''
    spectrogram = tfio.audio.spectrogram(
        audio,
        nfft=nfft,
        window=window,
        stride=stride)
    return tf.transpose(spectrogram, perm=[1, 0]) 
    # On transpose pour avoir une shape similaire à celle de librosa en sortie


def full_spectro_generation(file_path, label, split=False,nfft=2048,window=256,stride=256):
    '''
    Objective : Generate spectrogram from an audio file path
    Input : file_path , label = integer between 0 and 49
    Ouput : Spectrogram tf.Tensor shape (x,y)
    '''    
    audio_tensor = convert_audio_file_to_audio_tensor(file_path)
    tensor, audio_rate = convert_to_tensor(audio_tensor)
    if audio_rate != TARGET_SAMPLE_RATE:
        tensor = resample_audio_tensor(tensor, audio_rate)
    if split==True:
        tensor = split_tensor(tensor)
    harmonizedtensor = harmonize_tensor_shape(tensor)
    spectrogram = generate_spectrogram(harmonizedtensor,nfft=nfft, window=window, stride=stride)
    return spectrogram, label


def generate_mel_spectrogram(spectrogram,rate=TARGET_SAMPLE_RATE, mels=128, fmin=0, fmax=8000):
    '''
    Objective : Convert to mel spectrogram
    Input : Spectrogram tf.Tensor shape (x,y)
    Ouput : Spectrogram tf.Tensor shape (x,y)
    '''
    mel_spectrogram = tfio.audio.melscale(
        spectrogram,
        rate=rate,
        mels=mels,
        fmin=fmin,
        fmax=fmax)
    return mel_spectrogram


def generate_db_scale_mel_spectrogram(mel_spectrogram, top_db=80):
    '''
    Objective : Convert to db scale spectrogram
    Input : Spectrogram tf.Tensor shape (x,y)
    Ouput : Spectrogram tf.Tensor shape (x,y)
    '''
    db_scale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram,
        top_db=top_db)
    return db_scale_mel_spectrogram


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
    spectrogram = generate_spectrogram(harmonizedtensor)
    mel_spectrogram = generate_mel_spectrogram(spectrogram)
    db_scale_mel_spectrogram = generate_db_scale_mel_spectrogram(mel_spectrogram)
    
    # Test full intégré
    label = 2
    spectro, label2  = full_spectro_generation(file_path, label, split=True)
    assert label2 == 2
