import tensorflow as tf
import pandas as pd
from birds.preproc import generate_spectrogram, generate_mel_spectrogram, generate_db_scale_mel_spectrogram, one_hot_encode_target

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
BUFFER_SIZE = 10

# 1 - Create full dataset

def create_dataset(directory,
                   spectro_type='mel',
                   batch_size=BATCH_SIZE,buffer_size=BUFFER_SIZE):
    '''
    Objective : Generate dataset from directory
    Inputs : directory :    audio files and y_train.csv file listing files paths and related targets
             spectro_type : choose between 'spectro' , 'mel', and 'db' to obtain spectrogram, mel spectrogram or db scaled mel spectrogram
                            default value returns mel_spectrogram.
             batch_size :   Number of files processed in a batch. Default value is 32
             buffer_size:   Shuffling parameter. Default value is 10.
    Output : PrefetchDataset
    '''

    # 1 - Reading y_train.csv
    df = pd.read_csv(directory + 'y_train.csv')
    file_paths = directory + df['Path'].values
    labels = df['Target'].values

    # 2 - Create dataset from y_train.csv
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels)) 

    # 3 - Generate spectrogram from path colum 
    if spectro_type == 'mel':
        ds_train = ds_train.map(generate_mel_spectrogram)
    elif spectro_type == 'spectro':
        ds_train = ds_train.map(generate_spectrogram)
    elif spectro_type == 'db':
        ds_train = ds_train.map(generate_db_scale_mel_spectrogram)
    else :
        print('Choose correct spectro type between mel spectro db')
    print(ds_train)

    # 4 - One hot encode target
    ds_train = ds_train.map(one_hot_encode_target)
    print(ds_train)

    # 5 - Generate ds_train
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(buffer_size,reshuffle_each_iteration=False)
    # The parameter reshuffle_each_iteration=False is important for train / val split afterwards.
    # It makes sure the original dataset is shuffled once and no more. Otherwise, the two resulting sets may have some overlaps.
    ds_train = ds_train.batch(batch_size, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.prefetch(AUTOTUNE)
    
    return ds_train


# 2 Splitting the dataset for training and testing.

def is_test(x, _):
    return x % 4 == 0
# x % 4 == 0 takes 1 sample out of for (75% train/val split ratio). Likewise, x % 5 == 0 would achieve 80% split ratio. 

def is_train(x, y):
    return not is_test(x, y)

recover = lambda x, y: y
# delaring lambda separately to avoid AutoGraph limitations in TF 2.0 and above.

def split_train_val_dataset(dataset):
    # Split the dataset for training.
    val_dataset = dataset.enumerate() \
                .filter(is_test) \
                .map(recover)
    print(val_dataset)

    # Split the dataset for testing/validation.
    train_dataset = dataset.enumerate() \
                .filter(is_train) \
                .map(recover)
    print(train_dataset)
    return train_dataset, val_dataset



if __name__=="__main__":
    directory = 'raw_data/data_10s/train/'
    ds_train = create_dataset(directory)
    print(ds_train)