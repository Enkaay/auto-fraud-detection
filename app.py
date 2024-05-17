import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# Load the best model
best_pipeline = joblib.load('best_model.pkl')

# Load the dataset
data = pd.read_csv('fraud_oracle.csv')
target_variable = 'FraudFound_P'

# Define variable lists
categorical_vars = ['Month', 'DayOfWeek', 'Make', 'AccidentArea', 'DayOfWeekClaimed', 'MonthClaimed', 'Sex', 
                    'MaritalStatus', 'Fault', 'PolicyType', 'VehicleCategory', 'VehiclePrice', 'AgeOfPolicyHolder',
                    'Days_Policy_Accident', 'Days_Policy_Claim', 'PastNumberOfClaims', 'AgeOfVehicle', 
                    'PoliceReportFiled', 'WitnessPresent', 'AgentType', 'NumberOfSuppliments', 
                    'AddressChange_Claim', 'NumberOfCars', 'BasePolicy']
numerical_vars = ['WeekOfMonth', 'WeekOfMonthClaimed',  'RepNumber', 'DriverRating', 'Year', 'Deductible']

# Map fraud status to meaningful labels
label_map = {0: "No Fraud", 1: "Fraud"}
data['Fraud Status'] = data[target_variable].map(label_map)

# Streamlit page configurations
st.set_page_config(page_title="Vehicle Insurance Claim Prediction", layout="wide")

# Set up the sidebar
page = st.sidebar.radio("Menu", ['Dashboard', 'Prediction'])

def dashboard():
    # Dashboard Page
    st.title("Data Visualization Dashboard")

    # Select box for variable type
    data_type = st.selectbox("Select variable type for visualization:", ["Target Variable", "Categorical", "Numerical"])
    
    if data_type == "Target Variable":
        plot_data = data['Fraud Status'].value_counts().reset_index()
        plot_data.columns = ['Fraud Status', 'Count']
        colors_map = {'No Fraud':'blue', 'Fraud':'red'}
        fig = px.bar(plot_data, x='Fraud Status', y='Count', title="Distribution of Fraud Cases", 
                     labels={'Counts': 'Number of Cases'}, color='Fraud Status', 
                     color_discrete_map=colors_map)
        st.plotly_chart(fig)

    elif data_type == "Categorical":
        selected_cats = st.multiselect("Select Categorical Variables", categorical_vars, default=categorical_vars[:3])
        for var in selected_cats: 
            fig = px.histogram(data, x=var, color='Fraud Status', 
                               color_discrete_map={'No Fraud':'blue', 'Fraud':'red'},
                               barmode='group', labels={var: var, 'Fraud Status': "Fraud Status"},
                               title=f"{var} Distribution")
            st.plotly_chart(fig)
        
    elif data_type == "Numerical":
        selected_nums = st.multiselect("Select Numerical Variables", numerical_vars, default=numerical_vars[:3])
        for var in selected_nums:
            fig = px.histogram(data, x=var, color='Fraud Status', 
                               color_discrete_map={'No Fraud':'blue', 'Fraud':'red'},
                               barmode='group', labels={var: var, 'Fraud Status': "Fraud Status"},
                               title=f"{var} Distribution")
            st.plotly_chart(fig)

def prediction():
    st.subheader('Prediction')
    with st.form("input_form"):
        # Input form
        st.write("Please fill in the following information to predict the insurance claim.")

        areaOptions = ['Rural', 'Urban']
        AccidentArea = st.selectbox("Accident Area", [0, 1], index=1, format_func=lambda x: areaOptions[x])
        sexOptions = ['Female', 'Male']
        Sex = st.selectbox('Sex', [0, 1], index=0, format_func=lambda x: sexOptions[x])
        maritalStatusOptions = ['Divorced', 'Married', 'Single', 'Widow']
        MaritalStatus = st.selectbox("Marital Status", [0, 1, 2, 3], index=2, format_func=lambda x: maritalStatusOptions[x])
        faultOptions = ['Policy Holder', 'Third Party']
        Fault = st.selectbox("Fault", [0, 1], index=0, format_func=lambda x: faultOptions[x])
        # create an array of policy types, arrange them in alphabetical order
        policyTypeOptions = ['All Perils', 'Collision', 'Liability', 'Sedan - All Perils', 'Sedan - Collision', 'Sedan - Liability', 'Sport - All Perils', 'Sport - Collision', 'Sport - Liability', 'Utility - All Perils', 'Utility - Collision', 'Utility - Liability']
        PolicyType = st.selectbox("Policy Type", list(range(len(policyTypeOptions))), index=0, format_func=lambda x: policyTypeOptions[x])
        vehicleCategoryOptions = ['Sedan', 'Sport', 'Utility']
        VehicleCategory = st.selectbox("Vehicle Category", list(range(len(vehicleCategoryOptions))), index=0, format_func=lambda x: vehicleCategoryOptions[x])
        
        # create an array of vehicle prices, arrange them in alphabetical order
        vehiclePriceOptions = ['20000 to 29000', '30000 to 39000', '40000 to 59000', '60000 to 69000', 'less than 20000', 'more than 69000']
        VehiclePrice = st.selectbox("Price of the Vehicle", list(range(len(vehiclePriceOptions))), index=0, format_func=lambda x: vehiclePriceOptions[x])

        deductibleOptions = ['300 to 400', '400 to 600', '600 to 800', '800 to 1000', 'less than 300', 'more than 1000']
        Deductible = st.selectbox("Deductible", list(range(len(deductibleOptions))), index=0, format_func=lambda x: deductibleOptions[x])

        DriverRating = st.selectbox("Driver Rating", ['1', '2', '3', '4', '5'], index=0)
        
        daysPolicyAccidentOptions = ['1 to 7', '8 to 15', 'more than 30', 'none']
        Days_Policy_Accident = st.selectbox("Days Policy Accident", list(range(len(daysPolicyAccidentOptions))), index=0, format_func=lambda x: daysPolicyAccidentOptions[x])

        ageOptions = ['16 to 17', '18 to 20', '21 to 25', '26 to 30', '31 to 35', '36 to 45', '41 to 50', '51 to 65', 'over 65']
        AgeOfPolicyHolder = st.selectbox("Age Of Policy Holder", list(range(len(ageOptions))), index=0, format_func=lambda x: ageOptions[x])

        reportFiledOptions = ['No', 'Yes']
        PoliceReportFiled = st.selectbox("Police Report Filed", [0, 1], index=0, format_func=lambda x: reportFiledOptions[x])

        witnessPresentOptions = ['No', 'Yes']
        WitnessPresent = st.selectbox("Witness Present", [0, 1], index=0, format_func=lambda x: witnessPresentOptions[x])

        agentTypeOptions = ['External', 'Internal']
        AgentType = st.selectbox("Agent Type", [0, 1], index=0, format_func=lambda x: agentTypeOptions[x])

        supplimentsOptions = ['1 to 2', '3 to 5', 'more than 5', 'none']
        NumberOfSuppliments = st.selectbox("Number Of Suppliments", list(range(len(supplimentsOptions))), index=0, format_func=lambda x: supplimentsOptions[x])

        numberOfCarsOptions = ['1', '2', '3 to 4', '5 to 8', 'more than 8']
        NumberOfCars = st.selectbox("Number Of Cars", list(range(len(numberOfCarsOptions))), index=0, format_func=lambda x: numberOfCarsOptions[x])

        Year = st.number_input("Year", min_value=1990, max_value=2024, step=1, value=2022)

        basePolicyOptions = ['All Perils', 'Collision', 'Liability']
        BasePolicy = st.selectbox("Base Policy", [0, 1, 2], index=0, format_func=lambda x: basePolicyOptions[x])
       
        submitted = st.form_submit_button("Predict Insurance Claim")

        # Perform prediction if form is submitted
        if submitted:
            # Create a DataFrame with user inputs
            input_data = pd.DataFrame({
                'AccidentArea': [AccidentArea],
                'Sex': [Sex],
                'MaritalStatus': [MaritalStatus],
                'Fault': [Fault],
                'PolicyType': [PolicyType],
                'VehicleCategory': [VehicleCategory],
                'VehiclePrice': [VehiclePrice],
                'Deductible': [Deductible],
                'DriverRating': [DriverRating],
                'Days_Policy_Accident': [Days_Policy_Accident],
                'AgeOfPolicyHolder': [AgeOfPolicyHolder],
                'PoliceReportFiled': [PoliceReportFiled],
                'WitnessPresent': [WitnessPresent],
                'AgentType': [AgentType],                
                'NumberOfSuppliments': [NumberOfSuppliments],
                'NumberOfCars': [NumberOfCars],
                'Year': [Year],
                'BasePolicy': [BasePolicy],      
                  

            })

            # Perform prediction
            prediction = best_pipeline.predict(input_data)

            
            #define file path
            file_path = 'user_inputs.csv'
            # Check if the file exists
            file_exists = os.path.exists(file_path)

            
	    # Save the DataFrame to a CSV file
            input_data.to_csv(file_path, index=False, mode='a', header=not file_exists)


            # create dataFrame from the prediction
            #resultFrame = pd.DataFrame(prediction)
            #write the prediction result as a table to a file
            #resultFrame.to_csv('prediction_result2.csv', index=False)


            # Display prediction result
            if prediction[0] == 0:
                st.write("This claim is predicted to be a fraudlent insurance claim.")
            else:
                st.write("This claim is predicted to be a non-fraudlent insurance claim.")

# Page navigation
if page == "Dashboard":
    dashboard()
elif page == "Prediction":
    prediction()
