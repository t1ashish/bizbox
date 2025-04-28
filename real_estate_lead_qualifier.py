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
    result = classifier(response, candidate_labels, multi_label=False)
    intent_score = result["scores"][result["labels"].index("high_intent")]
    
    # Budget scoring (tailored to 32259: $400,000–$600,000)
    if budget >= 600000:
        budget_score = 1.0
    elif budget >= 400000:
        budget_score = 0.8
    elif budget >= 200000:
        budget_score = 0.5
    else:
        budget_score = 0.2
    
    # Location scoring (prioritize 32259)
    location = location.lower()
    if "32259" in location or "saint johns" in location or "st johns" in location:
        location_score = 1.0
    elif "florida" in location or "fl" in location:
        location_score = 0.7
    else:
        location_score = 0.3
    
    # Timeframe scoring
    timeframe = timeframe.lower()
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

def main():
    st.title("Real Estate Lead Qualifier (Saint Johns, FL 32259)")
    st.write("Enter lead details to qualify potential buyers for properties in the $400,000–$600,000 range.")
    
    # Input form
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
            st.error("Please fill in all fields.")

if __name__ == "__main__":
    main()