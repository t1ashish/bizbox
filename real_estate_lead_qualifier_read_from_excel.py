import streamlit as st
from transformers import pipeline
import pandas as pd
import os
from datetime import datetime

# Initialize NLP model for intent classification
@st.cache_resource
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_classifier()

def qualify_lead(response, budget, location, timeframe):
    """Qualify a lead based on response, budget, location, and timeframe."""
    # Intent classification
    candidate_labels = ["high_intent", "medium_intent", "low_intent"]
    try:
        result = classifier(response, candidate_labels, multi_label=False)
        intent_score = result["scores"][result["labels"].index("high_intent")]
    except Exception as e:
        st.warning(f"Error classifying inquiry: {e}. Assuming low intent.")
        intent_score = 0.3
    
    # Budget scoring (tailored to 32259: $400,000–$600,000)
    try:
        budget = float(budget)
        if budget >= 600000:
            budget_score = 1.0
        elif budget >= 400000:
            budget_score = 0.8
        elif budget >= 200000:
            budget_score = 0.5
        else:
            budget_score = 0.2
    except (ValueError, TypeError):
        st.warning(f"Invalid budget: {budget}. Assuming $0.")
        budget_score = 0.2
    
    # Location scoring (prioritize 32259)
    location = str(location).lower() if location else ""
    if "32259" in location or "saint johns" in location or "st johns" in location:
        location_score = 1.0
    elif "florida" in location or "fl" in location:
        location_score = 0.7
    else:
        location_score = 0.3
    
    # Timeframe scoring
    timeframe = str(timeframe).lower() if timeframe else ""
    if "immediately" in timeframe or "asap" in timeframe or "now" in timeframe:
        timeframe_score = 1.0
    elif "month" in timeframe or "soon" in timeframe:
        timeframe_score = 0.7
    else:
        timeframe_score = 0.4
    
    # Weighted score
    final_score = (0.4 * intent_score + 0.3 * budget_score + 0.2 * location_score + 0.1 * timeframe_score)
    
    if final_score >= 0.7:
        return "High Intent", final_score
    elif final_score >= 0.4:
        return "Medium Intent", final_score
    else:
        return "Low Intent", final_score

def save_lead_to_csv(name, email, response, budget, location, timeframe, intent, score):
    """Save lead details to a CSV file."""
    data = {
        'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        'Name': [name],
        'Email': [email],
        'Inquiry': [response],
        'Budget': [budget],
        'Location': [location],
        'Timeframe': [timeframe],
        'Intent': [intent],
        'Score': [score]
    }
    df = pd.DataFrame(data)
    csv_file = 'leads.csv'
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, index=False)

def process_excel_file(file_path=None, uploaded_file=None):
    """Process leads from an Excel file (file path or uploaded file)."""
    try:
        if uploaded_file:
            df = pd.read_excel(uploaded_file, sheet_name='Leads')
        elif file_path and os.path.exists(file_path):
            df = pd.read_excel(file_path, sheet_name='Leads')
        else:
            raise FileNotFoundError("No Excel file provided or found.")
        
        required_columns = ['Name', 'Email', 'Inquiry', 'Budget', 'Location', 'Timeframe']
        if not all(col in df.columns for col in required_columns):
            st.error("Excel file must contain columns: Name, Email, Inquiry, Budget, Location, Timeframe")
            return None
        
        results = []
        for _, row in df.iterrows():
            intent, score = qualify_lead(
                row['Inquiry'],
                row['Budget'],
                row['Location'],
                row['Timeframe']
            )
            save_lead_to_csv(
                row['Name'],
                row['Email'],
                row['Inquiry'],
                row['Budget'],
                row['Location'],
                row['Timeframe'],
                intent,
                score
            )
            results.append({
                'Name': row['Name'],
                'Email': row['Email'],
                'Inquiry': row['Inquiry'],
                'Budget': row['Budget'],
                'Location': row['Location'],
                'Timeframe': row['Timeframe'],
                'Intent': intent,
                'Score': score
            })
        return pd.DataFrame(results)
    except Exception as e:
        st.error(f"Error processing Excel file: {e}")
        return None

def main():
    st.title("Real Estate Lead Qualifier (Saint Johns, FL 32259)")
    st.write("Automatically processes 'leads.xlsx' if present, or upload an Excel file (.xlsx/.xls) with leads, or enter details manually to qualify potential buyers for properties in the $400,000–$600,000 range.")
    
    # Auto-detect leads.xlsx
    default_file_path = 'leads.xlsx'
    st.subheader("Auto-Detected Leads (leads.xlsx)")
    if os.path.exists(default_file_path):
        st.write(f"Found '{default_file_path}'. Processing...")
        results_df = process_excel_file(file_path=default_file_path)
        if results_df is not None:
            st.success(f"Processed leads from '{default_file_path}' and saved to 'leads.csv'.")
            st.subheader("Processed Leads")
            st.dataframe(results_df)
    else:
        st.info(f"No '{default_file_path}' found in the current directory.")
    
    # Excel file upload
    st.subheader("Upload Leads from Excel")
    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])
    if uploaded_file:
        results_df = process_excel_file(uploaded_file=uploaded_file)
        if results_df is not None:
            st.success("Leads processed and saved to 'leads.csv'.")
            st.subheader("Processed Leads")
            st.dataframe(results_df)
    
    # Manual input form
    st.subheader("Manual Lead Entry")
    with st.form(key='lead_form'):
        name = st.text_input("Lead Name")
        email = st.text_input("Lead Email")
        response = st.text_area("Lead Inquiry (e.g., 'Looking for a 3-bedroom home in Saint Johns')")
        budget = st.number_input("Budget ($)", min_value=0, value=400000, step=10000)
        location = st.text_input("Preferred Location (e.g., Saint Johns, FL 32259)")
        timeframe = st.text_input("Timeframe (e.g., 'Within 3 months')")
        submit_button = st.form_submit_button(label="Qualify Lead")
    
    if submit_button:
        if name and email and response and location and timeframe:
            intent, score = qualify_lead(response, budget, location, timeframe)
            st.success(f"Lead Intent: **{intent}** (Score: {score:.2f})")
            save_lead_to_csv(name, email, response, budget, location, timeframe, intent, score)
            st.write("Lead details saved to 'leads.csv' for CRM integration.")
            
            # Display saved leads
            if os.path.exists('leads.csv'):
                leads_df = pd.read_csv('leads.csv')
                st.subheader("Recent Leads")
                st.dataframe(leads_df.tail(5))
        else:
            st.error("Please fill in all fields for manual entry.")

if __name__ == "__main__":
    main()