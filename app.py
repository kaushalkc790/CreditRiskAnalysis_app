
import streamlit as st

# App title and description
st.set_page_config(page_title="Credit Risk Analysis", layout="wide")

st.title("🏦 Credit Risk Analysis System")
st.markdown("""
Predict the likelihood of loan default based on applicant characteristics.
""")

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model3.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Ensure category names are clean
term_categories = ['Short Term', 'Medium Term', 'Long Term']
term_categories = [x.strip() for x in term_categories] 

# Initialize label encoders (you'll need to replace these with your actual categories)
label_encoders = {
    'Gender': LabelEncoder().fit(["Male", "Female", "Joint", "Sex Not Available"]),
    'loan_type': LabelEncoder().fit(["type1", "type2", "type3"]),
    'loan_purpose': LabelEncoder().fit(["p1", "p2", "p3"]),
    'credit_type': LabelEncoder().fit(["CIB", "CRIF", "EXP"]),
    'occupancy_type': LabelEncoder().fit(['pr', 'ir', 'sr']),
    'age_group': LabelEncoder().fit(['<25', '25-34', '35-44','45-54','55-64','65-74','>74']),
    'Term_Category': LabelEncoder().fit(term_categories),
}




# Sidebar for user inputs
with st.sidebar:
    st.header("🔍 Applicant Information")
    
    # Numerical inputs with validation
    loan_amount = st.number_input("Loan Amount ($)", min_value=10000, value=50000, step=1000)
    property_value = st.number_input("Property Value ($)", min_value=10000, value=250000, step=1000)
    income = st.number_input("Monthly Income ($)", min_value=1000, value=15000, step=100)
    LTV = st.number_input("Loan-to-Value Ratio (LTV )", min_value=25.0, max_value=115.0, value=70.0) 
    dtir = st.number_input("Debt-to-Income Ratio (DTI )", min_value=6.0, max_value=68.0, value=40.0)
    DSCR =  st.number_input("Debt Service Coverage Ratio (DSCR )", min_value=1.0, max_value=15.0, value=14.0)
    ICR = st.number_input("Interest Coverage Ratio (ICR )", min_value=1.0, max_value=25.0, value=10.0)
    Credit_Score = st.number_input("Credit Score", min_value=500, max_value=900, value=700)
    rate_of_interest = st.number_input("Yearly Interest Rate (%)", min_value=0.0, max_value=10.0, value=5.0) 
    
    # Categorical inputs
    Gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
    loan_type = st.selectbox("Loan Type", label_encoders['loan_type'].classes_)
    loan_purpose = st.selectbox("Loan Purpose", label_encoders['loan_purpose'].classes_)
    credit_type = st.selectbox("Credit Type", label_encoders['credit_type'].classes_)
    occupancy_type = st.selectbox("Occupancy Type", label_encoders['occupancy_type'].classes_)
    age_group = st.selectbox("Age Group", label_encoders['age_group'].classes_)
    Term_Category = st.selectbox("Term Category", label_encoders['Term_Category'].classes_)

def preprocess_input(data):
    """Convert categorical variables using label encoding"""
    data['Gender'] = label_encoders['Gender'].transform([data['Gender']])[0]
    data['loan_type'] = label_encoders['loan_type'].transform([data['loan_type']])[0]
    data['loan_purpose'] = label_encoders['loan_purpose'].transform([data['loan_purpose']])[0]
    data['occupancy_type'] = label_encoders['occupancy_type'].transform([data['occupancy_type']])[0]
    data['credit_type'] = label_encoders['credit_type'].transform([data['credit_type']])[0] 
    data['age_group'] = label_encoders['age_group'].transform([data['age_group']])[0]
    data['Term_Category'] = label_encoders['Term_Category'].transform([data['Term_Category']])[0]
    return data

def main():
    # Create input dictionary
    input_data = {
        'Gender': Gender,
        'loan_type': loan_type,
        'loan_purpose': loan_purpose,
        'loan_amount': loan_amount,
        'rate_of_interest': rate_of_interest,
        'property_value': property_value,
        'occupancy_type': occupancy_type,
        'income': income,
        'credit_type': credit_type,
        'Credit_Score': Credit_Score,
        'age_group': age_group,
        'LTV': LTV,
        'dtir': dtir,
        'ICR':ICR,
        'DSCR':DSCR,
        'Term_Category': Term_Category
    }
    
    # Display input summary
    with st.expander("📋 Applicant Details", expanded=True):
        st.json({k: v for k, v in input_data.items() if not isinstance(v, str)})
        st.json({k: v for k, v in input_data.items() if isinstance(v, str)})
    
    if st.button("🔮 Predict Default Risk"):
        if model is None:
            st.error("Model not loaded. Cannot make predictions.")
            return
            
        try:
            # Preprocess input
            processed_data = preprocess_input(input_data.copy())
            input_df = pd.DataFrame([processed_data])
            
            # Make prediction
            prediction = model.predict(input_df)
            proba = model.predict_proba(input_df)[0]
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Prediction", 
                         "High Risk 🚨" if prediction[0] == 1 else "Low Risk ✅",
                         f"{proba[1]*100:.1f}% risk" if prediction[0] == 1 else f"{proba[0]*100:.1f}% safe")
            
            with col2:
                proba_df = pd.DataFrame({
                    'Risk Level': ['Low Risk', 'High Risk'],
                    'Probability': proba
                })
                st.bar_chart(proba_df.set_index('Risk Level'))
            
            # Add explanation
            st.info("""
            **Interpretation:**  
            - **Low Risk (✅):** Probability < 50%  
            - **High Risk (🚨):** Probability ≥ 50%
            """)
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

if __name__ == '__main__':
    main()


