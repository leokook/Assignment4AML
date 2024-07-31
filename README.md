# Breast Cancer Data Analysis and Prediction Streamlit App

## Project Description
This project aims to analyze breast cancer data and build a predictive model to classify cancer as benign or malignant. The model is integrated into a Streamlit app for easy interaction and prediction.

## Project Structure
- **Prediction.ipynb**: Jupyter notebook containing data analysis, preprocessing, model training, and evaluation.
- **app.py**: Streamlit app code that loads the trained model and scaler to make predictions on new input data.
- **X_test.csv**: Test data features.
- **y_test.csv**: Test data labels.
- **best_model.joblib**: Serialized trained model.
- **scaler.joblib**: Serialized scaler for data preprocessing.
- **requirements.txt**: List of dependencies required to run the project.

## Requirements
To run this project, you need to have Python installed along with the following libraries:
-  numpy
-  pandas
-  scikit-learn
-  streamlit
-  matplotlib
-  seaborn
-  joblib
-  scikeras
-  tensorflow
-  keras

You can install all the dependencies using the following command:
```bash
pip install -r requirements.txt
