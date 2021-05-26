import tensorflow as tf
import tensorflow_io as tfio
import numpy as np

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


# Optional
def split_tensor(tensor, audio_rate):
    '''
    Objective : split to keep only first seconds of audio
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

def full_spectro_generation(file_path,nfft=2048,window=256,stride=256):
    '''
    Objective : Generate spectrogram from an audio file path
    Input : file_path
    Ouput : Spectrogram tf.Tensor shape (x,y)
    '''    
    audio_tensor = convert_audio_file_to_audio_tensor(file_path)
    tensor, audio_rate = convert_to_tensor(audio_tensor)
    split = split_tensor(tensor, audio_rate)
    harmonizedtensor = harmonize_tensor_shape(split)
    spectrogram = generate_spectrogram(harmonizedtensor,nfft=nfft, window=window, stride=stride)
    return spectrogram


def generate_mel_spectrogram(spectrogram,rate=44100, mels=128, fmin=0, fmax=8000):
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
    file_path = "raw_data/subdataset_cs/train/Aegolius-funereus-131493.mp3"
    audio_tensor = convert_audio_file_to_audio_tensor(file_path)
    assert audio_tensor.rate.numpy() == 44100
    tensor, audio_rate = convert_to_tensor(audio_tensor)
    assert audio_rate == 44100
    assert tensor.shape[1] == 2 # stereo file
    split = split_tensor(tensor, audio_rate)
    assert split.shape[0] == 441000
    harmonizedtensor = harmonize_tensor_shape(split)
    assert harmonizedtensor.shape[0] == 441000
    spectrogram = generate_spectrogram(harmonizedtensor)
    mel_spectrogram = generate_mel_spectrogram(spectrogram, rate=audio_rate)
    db_scale_mel_spectrogram = generate_db_scale_mel_spectrogram(mel_spectrogram)
    
    # Test full intégré
    spectro = full_spectro_generation(file_path)
