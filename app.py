import streamlit as st
import numpy as np
import pandas as pd
import svm
import matplotlib.pyplot as plt
import seaborn as sns

import pickle

model = pickle.load(open('loan_status_model.pkl', 'rb'))

st.title("📊 My Streamlit App")

option = st.sidebar.selectbox(
    "Choose an option",
    ("Loan Prediction","Data Visualizer")
)

# Change the upper title
# st.set_page_config(page_title="Loan Prediction App",page_icon="pic2.png")



def preprocess_features(features):
    # Fill missing values
    features.interpolate(method='linear', inplace=True)
    features['Gender'].fillna(features['Gender'].mode()[0], inplace=True)
    features['Married'].fillna(features['Married'].mode()[0], inplace=True)
    features['Dependents'].fillna(features['Dependents'].mode()[0], inplace=True)
    features['Self_Employed'].fillna(features['Self_Employed'].mode()[0], inplace=True)

    # Replace categorical values with numerical labels
    features.replace({'Married': {'No': 0, 'Yes': 1},
                      'Gender': {'Male': 1, 'Female': 0},
                      'Self_Employed': {'No': 0, 'Yes': 1},
                      'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
                      'Education': {'Graduate': 1, 'Not Graduate': 0}}, inplace=True)

    # Replace '3+' with 4 in the Dependents column
    features = features.replace(to_replace='3+', value=4)

    return features

if option=="Loan Prediction":
    # Set a title for the web app
    # CSS styling to center-align the title
    title_style = """
        <style>
            .title {
                text-align: center;
            }
        </style>
    """

    # Display the title with centered styling
    st.markdown(title_style, unsafe_allow_html=True)
    st.markdown("<h1 class='title'>Loan Status Prediction</h1>", unsafe_allow_html=True)
    image_path ="pic3.jpeg"
    st.image(image_path,use_column_width=True)


    # Add input fields for the user to enter the feature values
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["No", "Yes"])
    dependents = st.number_input("Number of Dependents (0-4)",min_value=0,max_value=4)
    education = st.selectbox("Education", ["Not Graduate", "Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])
    applicant_income = st.number_input("Applicant Income (INR) (1000-10000)", min_value=0, value=0)
    coapplicant_income = st.number_input("Co-applicant Income (INR) (1000-10000)", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands) (10-500)", min_value=0, value=0)
    loan_amount_term = st.number_input("Loan Amount Term (in months) (10-360)", min_value=0, value=0)
    credit_history = st.selectbox("Credit history of individual’s repayment of their debts (0 for No history 1 for Having History) ", [0, 1])
    property_area = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    # Create a dictionary with the user input features
    user_data = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    # Preprocess the user input features
    processed_data = preprocess_features(pd.DataFrame(user_data, index=[0]))

    # Make predictions using the loaded model
    prediction = model.predict(processed_data)

    if st.button("Predict Loan Status"):
        # Make predictions using the loaded model
        prediction = model.predict(processed_data)

        # Display the prediction result
        if prediction[0] == 1:
            st.success("Congratulations! Your loan is likely to be approved.")
        else:
            st.error("Sorry, your loan is likely to be rejected.")


elif option == "Data Visualizer":
    st.header("Data Visualization")
    # Add your code or content for data visualization here
    # st.write("This is where you can upload your data and create visualizations.")

    df = st.file_uploader(label= 'Upload your dataset: ')

    if df:
        df = pd.read_csv(df)
        
        feature_columns = df.columns.tolist()

        st.write("")
        st.write(df.head())
        
        object_type = df.select_dtypes(include = ['object']).columns.tolist()
        numeric_type = df.select_dtypes(include= ['int64','int32']).columns.tolist()
        
        st.write("---")
        with st.container():
            st.markdown("<h2 style='text-align: center'>Feature Types</h2>", unsafe_allow_html=True)
        
        # Nested container for the columns
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Categorical Feature")
                for col in object_type:
                    st.write(f"- {col}")

            with col2:
                st.subheader("Numerical Feature")
                for col in numeric_type:
                        st.write(f"- {col}")
                
        
        
        st.write("---")
        
        x_axis = st.selectbox("Select the X_axis", options = feature_columns + ["None"], index = None)
        y_axis = st.selectbox("Select the Y_axis", options = feature_columns + ["None"], index = None)
            
        plot_list = ["Line Plot", "Bar Chart", "Scatter Plot", "Distribution Plot", "Count Plot"]
        plot = st.selectbox("Select a Visualisation" , options= plot_list, index= None)
        
        
        if st.button("Generate Plot"):
            fig, ax = plt.subplots(figsize =(6,4))
            
            if plot == "Line PLot":
                sns.lineplot(x = df[x_axis], y = df[y_axis], ax = ax)
            elif plot == "Bar Chart":
                sns.barplot(x = df[x_axis], y = df[y_axis], ax = ax)
            elif plot == "Scatter Plot":
                sns.scatterplot(x = df[x_axis], y = df[y_axis], ax = ax)
            elif plot == "Distribution Plot":
                sns.histplot(x = df[x_axis],kde= True ,ax = ax)
            elif plot == "Count Plot":
                sns.countplot(x = df[x_axis],ax = ax)
                
            
            st.pyplot(fig)
            
        
    # st.write(df['Units Sold'].groupby(df['Product Category']).value_counts().reset_index())
# if __name__ == '__main__':
#     main()
