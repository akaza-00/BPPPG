# Blood Pressure Estimation using PPG Signal Morphological Features

This repository contains the codes corresponding to study continous blood pressure estimation using PPG features.


## Usage
 You can skip steps 1,2, and 3 and use the extracted features.csv.
 
 1- Download MIMIC II dataset (BP, ECG, PPG) from https://archive.physionet.org/mimic2/ and save as MATLAB matrices in "data" folder.
 
 2- Run featureExtractor.py
 
 3- Run CSVconcat.py to concat CSV feature files created by the previous step.
 
 4- Run BPEstimation.py to fit and test machine learning based models.

## Citing this work
Please use the following citation:
```
Hasanzadeh, Navid, Mohammad Mahdi Ahmadi, and Hoda Mohammadzade. "Blood pressure estimation using photoplethysmogram signal and its morphological features." IEEE Sensors Journal 20, no. 8 (2019): 4300-4310.
```
