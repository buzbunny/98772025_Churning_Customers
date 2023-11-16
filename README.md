# Churn Prediction Model Deployment

This repository contains code for training a churn prediction model using customer data and deploying it as a Streamlit web app. The model predicts whether a customer is likely to churn based on various features.

## Model Training
The model training code is available in the `model_training.ipynb` notebook. It includes steps for:

- Data preprocessing: handling missing values, encoding categorical features, scaling numerical features.
- Feature importance analysis and exploratory data analysis (EDA).
- Model selection, training (using both Random Forest Classifier and a Neural Network), and optimization using GridSearchCV for hyperparameter tuning.
- Evaluating the model's performance using accuracy and AUC score.

## Model Deployment
The deployment code is available in the `Churn_web_app.py` file. It utilizes Streamlit to create a web app that takes user inputs and makes predictions using the trained model.

### Steps for Running the Web App
1. Ensure all required libraries are installed by running `pip install -r requirements.txt`.
2. Run the app using `streamlit run app_deployment.py`.
3. Input customer features through the sidebar and click the "Predict" button to see the churn prediction.

## Files in the Repository
- `model_training.ipynb`: Jupyter Notebook containing the model training code.
- `app_deployment.py`: Python file for deploying the trained model as a Streamlit web app.
- `best_model1.h5`: Saved trained neural network model.
- `scaler.pkl`: Pickled StandardScaler used for scaling input features.
- `README.md`: This documentation file.
- `requirements.txt`: List of required Python libraries.
- `summary_video.mp4`: Summary video demonstrating the deployment process.

### Access Summary Video
You can access the summary video of the deployment with the link below (https://drive.google.com/file/d/1j2PZJrtz5PjMvKoVfrf2YOUXLBmj6s2b/view?usp=share_link).

For any inquiries or issues, feel free to contact baidenhenry2@gmail.com.
