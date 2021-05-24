import pandas as pd

url='https://drive.google.com/file/d/1rk9luc1DICD492x-t9avH2sDpXmGBnKO/view?usp=sharing'
url2='https://drive.google.com/uc?id=' + url.split('/')[-2]
df = pd.read_csv(url2)


""""
Objective : Remove some recordings
Input : DataFrame
Output : DataFrame
"""

def filter_data_frame(df) :
  # Keep only recordings where the bird is alone
  df_alone = df[df['Other_species1'].isna()]
  return df_alone

""""
Objective : Retrieve paths from dataframe
Inputs : DataFrame
Output : Paths list of mp3 files on a filtered DataFrame
"""

def retrieve_paths(df):
    return [df['Path'].iloc[i][5:] for i in range(len(df['Path']))]


if __name__ =="__main__":
    df_alone = filter_data_frame(df)
    assert type(df_alone) == pd.core.frame.DataFrame
    paths = retrieve_paths(df_alone)
    assert type(paths) == list