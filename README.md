## Vehicle Fraud Detection Tool: 
### Insurance fraud is a big problem for insurance companies as such, this project aims to detect insurance fraud using machine learning. The project consists of two parts: Jupyter notebook and a web application.
the Jupyter notebook is for data analysis and machine learning model building and the web application for deploying the model using Streamlit

## The running version of this app can be viewed online using this URL
https://auto-fraud-detection-cetm46-bi55iw.streamlit.app/

To run locally, follow the installation guide below


### Installation

To run the app, you need to have Python installed on your system preferably version 3.9.11 
Additionally, ensure you have the necessary libraries installed. 
You can install the required libraries using pip:


pip install pandas 
pip install seaborn
pip install 
pip install plotly-express 
pip install numpy
pip install -U scikit-learn
pip install joblib
Pip install streamlit


### Dataset

The dataset contains detailed records of vehicle insurance claims, capturing temporal aspects like the month and day of the week the claim was filed and processed. It includes vehicle specifics such as make and age, and policyholder demographics like gender, marital status, and age group. The dataset also records whether a police report was filed, if a witness was present, and the type of agent handling the claim. Other details include the number of supplements filed, recent address changes, the total number of insured cars, the initiation year of the policy, and the policy type (Liability or Collision). 

### Components


1. **Jupyter Notebook (`model.ipynb`):**
   - The notebook performs exploratory data analysis (EDA) on the dataset to understand its characteristics.
   - It visualizes data distributions, correlations, and trends using various plots such as bar charts, and heatmaps.
   - Data cleaning and preprocessing is also carried out having noticed that the dataset was highly imbalanced.
   - Six different algorithms namely: SVM, Logistics regression, K-Nearest neighbours, decision trees, random forest and gradient boosting were sampled. The best performing model was selected and saved as 'best_model.pkl'. 
   

2. **Streamlit Web App (`app.py`):**
   - The GUI-based app allows users to input policy holder details.
   - Based on the input, the app predicts the outcome
   - Users can visualize the predicted action and expected profit change.
   - Users input is also saved a CSV file (user_inputs.csv)

3. **(best_model.pkl):**
   - This saves the best algorithm and uses it for the prediction.

4 **(user_inputs.csv):**
   - This CSV file acts as a repository for user-provided information. The collected data can be used to create a new dataset, which can then be analyzed and utilized to further enhance the model's performance in the future.


### Usage

1. **Run the Jupyter Notebook:**
   - Open the Jupyter Notebook (`model.ipynb`) in a Jupyter environment.
   - Execute the cells in the notebook to perform EDA and train the model.
   - A file will be saved as best_model.pkl. This model saves the best algorithm and uses it for prediction.

2. **Run the Streamlit App:**
   - Ensure that you have the model and the best algorithm saved in a file is created as (`best_model.pkl`).This should be saved in the same directory as the app.

   - Run the Streamlit app by executing the following command:
     ```bash
     run streamlit app.py
     ```
   - The app window will appear, allowing you to select a module
   - The 'dashboard' module is where data visualization happens to provide domain experts with valuable data insights
   - Click the "Predict Insurance Claim" button to see the predicted outcome of the claim.
   

### Note

- Ensure that all necessary files (`app.py`, `model.ibynb`,`best_model.pkl` and  `fraud_oracle.csv`) are in the same directory.


