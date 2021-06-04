# Projects : BIRDS
- Description: Bird song recognition model and API
- Data Source: https://www.kaggle.com/monogenea/birdsongs-from-europe .
50 european birds species from Xeno canto.
All files were used (after train / test split) and only first 10 sec used to build model.
- Type of analysis: Dataset analysis + classification model using Densenet transfer learning

Please document the project the better you can.

# Main libraries
Pandas : Data analysis
Tensorflow IO (for audio file input agnostic to file type)
Tensorflow : for preprocessing : audio to mel spectrogram + dataset test/val creation
	     for model training : fully integrated with preprocessing
	     
Model : CNN transfer learning from Densenet169 adaptated for our topic


# Methodology
Dataset investigation and preprocessing tested through Notebooks -> see Notebooks folder
Code packaged into py modules
Model training done in Google Colab
Mel spectrogram were used to train model to achieve best performance

Model exposed through fastAPI and deployed on Google Cloud Run using Docker
Front end in Heroku : see dedicated git : git@github.com:benoitdb/birds-frontend.git

Project done during le Wagon data batch 589.

Enjoy !
