import os
import tensorflow as tf
import pandas as pd
from birds.preproc import generate_spectrogram, generate_mel_spectrogram, generate_db_scale_mel_spectrogram, one_hot_encode_target

AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
BUFFER_SIZE = 10

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


# 1 - Create train and val dataframes listing files and targets

def create_df_train_df_val_from_directory(directory, sound_filetype = 'ogg',train_val_ratio = 0.8):
    '''
    Objective : Generate two data frames (train and val) with list of audio files and associated target number,
                taking into account potential class imbalances
    Inputs : directory :  file directory containing audio files
             sound_file_type : by default ogg
             train_val_ratio : ratio used to split between training and validation data
    Output : train and val dataframes
    '''
    # create dataframe with directory audio file list, target names derived from files name and target number
    data = pd.DataFrame(sorted([file for file in os.listdir(directory) if file.endswith(sound_filetype)])
                        ,columns=['Path'])
    data['Target_name'] = data['Path'].apply(lambda x : ' '.join(x.split(sep='-')[0:2]))
    target_list = list(pd.unique(data['Target_name']))
    # data['Target'] = data['Target_name'].apply(lambda x: target_list.index(x))
    data['Target'] = data['Target_name'].map(TARGET_DICT) # On récupère les numéros de classe originaux
    
    # create intermediate dataframe to calculate split indexes by target using train_val_ratio
    subdf_count_by_target = pd.pivot_table(data,index=['Target_name'],aggfunc={'Target' : 'count'})\
                                            .rename(columns={'Target':'target_size'})
    subdf_cummul_sum_by_target = data.groupby(by=['Target_name']).sum()\
                                            .rename(columns={'Target':'start_index'})
    split_index_df = subdf_count_by_target.merge(subdf_cummul_sum_by_target, left_index=True, right_index=True)
    split_index_df['split_index'] = split_index_df['start_index']+round(train_val_ratio*split_index_df['target_size'])
    print(split_index_df.head())
    
    # create train and val from first dataframe using indexes calculated in split_index_df
    df_train = data.iloc[0:0]
    for target in list(pd.unique(data['Target_name'])):
        start_index = int(split_index_df.loc[target].start_index)
        split_index = int(split_index_df.loc[target].split_index)
        df_train = df_train.append(data.iloc[start_index:split_index])

    df_val = pd.concat([data,df_train]).drop_duplicates(keep=False)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    
    return df_train, df_val

# 2 - create datasets

def create_train_val_datasets(directory,
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
    # 1 - Get val and train files
    df_train, df_val = create_df_train_df_val_from_directory(directory)
    file_paths_train = directory + df_train['Path'].values
    labels_train = df_train['Target'].values
    file_paths_val = directory + df_val['Path'].values
    labels_val = df_val['Target'].values   

    # 2 - Create datasets ds_train and ds_val
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths_train, labels_train))
    ds_val = tf.data.Dataset.from_tensor_slices((file_paths_val, labels_val)) 

    # 3 - Generate spectrogram from path colum 
    if spectro_type == 'mel':
        ds_train = ds_train.map(generate_mel_spectrogram)
        ds_val = ds_val.map(generate_mel_spectrogram)
    elif spectro_type == 'spectro':
        ds_train = ds_train.map(generate_spectrogram)
        ds_val = ds_val.map(generate_spectrogram)
        
    elif spectro_type == 'db':
        ds_train = ds_train.map(generate_db_scale_mel_spectrogram)
        ds_val = ds_val.map(generate_db_scale_mel_spectrogram)
    else :
        print('Choose correct spectro type between mel spectro db')
    print(ds_train)

    # 4 - One hot encode target
    ds_train = ds_train.map(one_hot_encode_target)
    ds_val = ds_val.map(one_hot_encode_target)
    print(ds_train)

    # 5 - Generate ds_train
    ds_train = ds_train.cache()
    ds_val = ds_val.cache()
    ds_train = ds_train.shuffle(buffer_size)
    ds_val = ds_val.shuffle(buffer_size)
    ds_train = ds_train.batch(batch_size, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.batch(batch_size, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.prefetch(AUTOTUNE)
    ds_val = ds_val.prefetch(AUTOTUNE)
    
    return ds_train, ds_val

if __name__=="__main__":
    directory = 'raw_data/data_10s/train/'
    ds_train, ds_val = create_train_val_datasets(directory)
    
    print(ds_train)