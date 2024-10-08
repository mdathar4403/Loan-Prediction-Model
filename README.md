# 📉 Loan Status Prediction 📊

## About the Project

A Loan Status Prediction system is a machine learning model or application that predicts whether a loan application will be approved or rejected based on various input features and historical data. The system aims to automate the loan approval process and assist in making informed decisions by assessing the creditworthiness of applicants.

The Loan Status Prediction system typically utilizes a dataset that includes information about loan applicants, such as personal details (gender, age, marital status), financial information (income, debt obligations), credit history (credit score, repayment history), employment details, and other relevant factors.

The system goes through several steps, including data preprocessing, feature engineering, model training, and prediction. 

The Dataset Contains 13 features
Let's discuss how each feature in the dataset could potentially impact the target feature, which is "Loan_Status" (indicating whether the loan was approved or not).

**1) Loan_ID:** A unique identifier for each loan application. It doesn't contribute to the decision-making process but can be useful for record-keeping.

**2) Gender:** Lending institutions might consider gender as a factor in loan approval, depending on historical data or institutional policies. For instance, if there's evidence of gender-based discrimination, it could affect loan approval.

**3) Married:** Married individuals may be perceived as more financially stable and responsible.Lenders might be more inclined to approve loans for married applicants.

**4) Dependents:** The number of dependents could influence loan approval, as more dependents might mean higher financial responsibilities. Lenders may assess the applicant's ability to repay the loan considering their family size.

**5) Education:** The level of education might be a proxy for the applicant's earning potential and financial stability. Graduates may be perceived as having better job prospects and, consequently, higher repayment capabilities.

**6) Self_Employed:** Self-employed individuals may face different income patterns compared to salaried individuals. Lenders might scrutinize the stability of self-employed applicants' income sources.

**7) ApplicantIncome:** Higher income generally indicates a better ability to repay a loan. However, extremely high or low incomes might be red flags. Lenders may set income thresholds for loan approval.

**8) CoapplicantIncome:** The income of the coapplicant can supplement the household income, affecting the overall repayment capacity. A higher coapplicant income may positively influence loan approval.

**9) LoanAmount:** The amount of the loan applied for is crucial. Lenders will assess whether the requested loan amount aligns with the applicant's income and financial situation.

**10) Loan_Amount_Term:** The term of the loan affects monthly repayment amounts. Shorter terms might indicate a quicker repayment ability, while longer terms might be associated with higher overall interest payments.

**11) Credit_History:** This is likely one of the most critical factors. A good credit history (1.0) is generally associated with a higher likelihood of loan approval. Lenders heavily rely on credit history to assess risk.

**12) Property_Area:** The location of the property can influence loan approval. Urban areas might have different risk profiles than rural areas, and lenders may have specific criteria for different regions.


## Steps involved in the Project

1. **Dataset**: The project utilizes a dataset containing information about loan applications. The dataset includes features such as gender, marital status, number of dependents, education, employment status, income details, loan amount, loan amount term, credit history, property area, and the loan status (whether approved or not). ```Click on the dataset link```[click here](https://github.com/mdathar4403/Loan-Prediction-Model)

2. **Data Preprocessing:** The dataset undergoes several preprocessing steps to handle missing values, convert categorical variables into numerical representations, and ensure consistency with the training data. Missing values are filled using appropriate methods, and categorical variables are encoded using label encoding or mapping.

3. **Exploratory Data Analysis (EDA):** EDA is performed using data visualization techniques. Plots such as count plots are used to analyze the distribution of loan approval based on education, marital status, etc., providing insights into the relationship between various features and the loan status.

4. **Splitting the Data:** The preprocessed dataset is divided into training and testing sets using the train_test_split function from sklearn.model_selection. This splitting allows us to train the SVM classifier on a portion of the data and evaluate its performance on unseen data.

5. **SVM Classifier:** The SVM (Support Vector Machine) classifier is chosen as the predictive model for loan status prediction. The classifier is initialized with a linear kernel, indicating a linear decision boundary between the loan approval classes.

6. **Training and Evaluation:** The SVM classifier is trained using the training data. The fit method is used to train the classifier on the features and the corresponding loan approval labels. The accuracy of the model is evaluated using the training data itself by comparing the predicted labels with the actual labels. The same process is applied to the testing data to evaluate the model's performance on unseen data.

7. **Saving the Model:** The trained SVM classifier is saved using the pickle library. The model is serialized and saved to a file (e.g., loan_status_model.pkl) to be used for future predictions without the need for retraining.

8. **Streamlit App:** A web application is developed using Streamlit to create an interactive interface for users to input loan application details. The input fields include features such as gender, marital status, income, loan amount, etc. After entering the details and clicking the "Predict Loan Status" button, the trained SVM model is loaded, and the loan status is predicted based on the provided information.

## ```Click to check the loan status``` [click here👉](https://loan-prediction-model-mdathar4403.streamlit.app/)

## A brief about Support Vector Machine Model

Support Vector Machine,(SVM), falls under the “supervised machine learning algorithms” category. It can be used for classification, as well as for regression. In this model, we plot each data item as a unique point in an n-dimension,(where n is actually, the number of features that we have), with the value of each of the features being the value of that particular coordinate. Then, we perform the process of classification by finding the hyper-plane that differentiates the two classes.

![1](https://github.com/dhrupad17/Loan-Status-Prediction/assets/91726340/2e544eb1-5c8d-4239-a81b-b6001ef8185e)

SVM is preferred over other algorithms when :

1)The data is not regularly distributed.

2)SVM is generally known to not suffer the condition of overfitting.

3)Performance of SVM, and its generalization is better on the dataset.

4)And, lastly, SVM is known to have the best results for classification types of problems.


