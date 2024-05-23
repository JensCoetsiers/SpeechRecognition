# SpeechRecognition
This repo includes all the code needed to run different speech recognition models

## How to
### automate.py
This is the most important file in the repo, because this file imports functions from the seperate model files such as google_model.py,...
It is important that you run this file in the correct virtual environment for example:
If you want to use the deepgram model, start up a terminal with the deepgram virtual environment active and then run
```
python ./automate.py
```

### Model specific files
Files such as google_model.py, seamless_model.py are files that include specific functions for each model to work, every model takes in data in different ways and these files contain the specific implementation elements.

## Folders
### CGN code
This folder has the scripts written by https://github.com/wilrop/Import-CGN, these scripts preprocess and import the CGN data into csv files with the filepath and filename.

### metadata 
This folder contains CSV files with speaker metadata, including details such as region and age. Additionally, the text files provided are all the files corresponding to that region (CGN only).

### ref_transcripts
This folder includes all of the reference transcripts for CGN and Variants corpus.
These files have the following structure:
```
filename|transcript
```
