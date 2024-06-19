
# Stress Detection with Machine Learning and Deep Learning using Multimodal Physiological Data

## Overview

This repository contains the implementation for the paper "Stress Detection with Machine Learning and Deep Learning using Multimodal Physiological Data." The aim of this project is to detect stress levels using various machine learning and deep learning techniques on multimodal physiological data collected from wearable sensors.

## Dataset

The dataset used in this study is the WESAD (Wearable Stress and Affect Detection) dataset, which includes data from 15 subjects. The data was collected using two devices:
- RespiBAN Professional (chest-worn) which measures ACC, RESP, ECG, EDA, EMG, and TEMP.
- Empatica E4 (wrist-worn) which measures ACC, BVP, EDA, and TEMP.

## Features

The features extracted from the dataset include statistical features like mean, standard deviation, minimum, maximum, and peak frequency for each signal. These features are used to train and test various machine learning and deep learning models.

## Models

The following models were used for classification tasks:
- **Machine Learning Models**
  - K-Nearest Neighbour (KNN)
  - Linear Discriminant Analysis (LDA)
  - Random Forest (RF)
  - Decision Tree (DT)
  - AdaBoost (AB)
  - Kernel Support Vector Machine (SVM)
- **Deep Learning Model**
  - Artificial Neural Network (ANN)

## Classification Tasks

Two types of classification tasks were performed:
1. **Three-class classification**: amusement vs. baseline vs. stress
2. **Binary classification**: stress vs. non-stress

## Requirements

To run the code in this repository, you need the following dependencies:
- Python 3.6 or higher
- numpy
- pandas
- scikit-learn
- keras
- tensorflow

You can install the dependencies using pip:
```bash
pip install numpy pandas scikit-learn keras tensorflow
```

## Results

The results of the models will be saved in the `results` directory. You can find detailed performance metrics and plots in this directory.

## Conclusion

This project demonstrates the feasibility of detecting stress levels using machine learning and deep learning techniques on multimodal physiological data. The deep learning model (ANN) achieved the highest accuracy for both three-class and binary classification tasks.

## References

If you use this code or dataset, please cite the following paper:
```
@inproceedings{bobade2020stress,
  title={Stress Detection with Machine Learning and Deep Learning using Multimodal Physiological Data},
  author={Bobade, Pramod and Vani, M},
  booktitle={2020 Second International Conference on Inventive Research in Computing Applications (ICIRCA)},
  pages={51--57},
  year={2020},
  organization={IEEE}
}
```

## License

This project is licensed under the MIT License.
