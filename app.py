import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load data
training = pd.read_csv(r'F:\Internship\healthcare-chatbot\healthcare-chatbot\Data\Training.csv')
testing = pd.read_csv(r'F:\Internship\healthcare-chatbot\healthcare-chatbot\Data\Testing.csv')

# Preprocessing
cols = training.columns[:-1]
x = training[cols]
y = training['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier().fit(x_train, y_train)

# Dictionaries for severity, description, and precautions
severity_dict = {}
description_dict = {}
precaution_dict = {}

def get_severity_dict():
    with open(r'F:\Internship\healthcare-chatbot\healthcare-chatbot\MasterData\Symptom_severity.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                severity_dict[row[0]] = int(row[1])

def get_description_dict():
    with open(r'F:\Internship\healthcare-chatbot\healthcare-chatbot\MasterData\symptom_Description.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                description_dict[row[0]] = row[1]

def get_precaution_dict():
    with open(r'F:\Internship\healthcare-chatbot\healthcare-chatbot\MasterData\symptom_precaution.csv') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                precaution_dict[row[0]] = row[1:]

get_severity_dict()
get_description_dict()
get_precaution_dict()

# Chatbot response function
def chatbot_response(user_input):
    symptom_list = user_input.split(",")
    symptom_list = [s.strip() for s in symptom_list]

    input_vector = np.zeros(len(cols))
    for symptom in symptom_list:
        if symptom in cols:
            input_vector[cols.get_loc(symptom)] = 1

    prediction = clf.predict([input_vector])[0]
    description = description_dict.get(prediction, "No description available.")
    precautions = precaution_dict.get(prediction, ["No precautions available."])
    
    response = f"You may have **{prediction}**.\n\nDescription: {description}\n\nPrecautions:\n"
    for i, precaution in enumerate(precautions, 1):
        response += f"{i}. {precaution}\n"
    return response

# Define chatbot interaction function
def chatbot_interaction():
    st.title("HealthCare ChatBot")
    st.write("-----------------------------------HealthCare ChatBot-----------------------------------")

    name = st.text_input("Your Name:")
    if name:
        st.write(f"Hello, {name}!")

    symptom = st.text_input("Enter the symptom you are experiencing:")
    if symptom:
        st.write(f"Searches related to: {symptom}")
        related_symptoms = [key for key in severity_dict.keys() if symptom.lower() in key.lower()]
        if related_symptoms:
            st.write("Related symptoms found:")
            for idx, item in enumerate(related_symptoms):
                st.write(f"{idx}) {item}")

            selected_idx = st.number_input("Select the symptom index:", min_value=0, max_value=len(related_symptoms) - 1, step=1)
            selected_symptom = related_symptoms[int(selected_idx)]

            num_days = st.number_input(f"How many days have you been experiencing {selected_symptom}?", min_value=1, step=1)
            st.write("Answer the following questions with 'yes' or 'no':")
            additional_symptoms = []
            for symptom in description_dict.keys():
                response = st.radio(f"Do you have {symptom}?", ["yes", "no"], key=symptom)
                if response == "yes":
                    additional_symptoms.append(symptom)

            st.write("Analyzing your inputs...")
            st.write(f"You may have {selected_symptom}.")
            if selected_symptom in description_dict:
                st.write(f"Description: {description_dict[selected_symptom]}")
            if selected_symptom in precaution_dict:
                st.write("Take the following precautions:")
                for i, precaution in enumerate(precaution_dict[selected_symptom]):
                    st.write(f"{i + 1}) {precaution}")
            
            severity_score = sum([severity_dict.get(symptom, 0) for symptom in additional_symptoms])
            advice = "Consult a doctor immediately." if (severity_score * num_days) / (len(additional_symptoms) + 1) > 13 else "Take precautions."
            st.write(f"**Advice:** {advice}")
        else:
            st.write("No related symptoms found. Please try again with a different symptom.")

# Streamlit App
st.title("Healthcare Web App")

# Add custom CSS for a visually appealing layout
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Arial', sans-serif;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 10px;
        transition-duration: 0.4s;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton button:hover {
        background-color: white;
        color: black;
        border: 2px solid #4CAF50;
    }
    .stTextInput input {
        padding: 10px;
        border: 2px solid #ccc;
        border-radius: 4px;
        transition: border-color 0.3s;
    }
    .stTextInput input:focus {
        border-color: #4CAF50;
    }
    .stSidebar {
        background-color: #2E3B55;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation for improved interaction
menu = st.sidebar.selectbox("Navigation", ["Chatbot", "Chatbot Interaction", "Disease Prediction", "About"])

if menu == "Chatbot":
    st.header("Healthcare Chatbot")

    # Chatbot Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "bot", "text": "Hello! I am your healthcare chatbot. How can I assist you today?"}]
    
    for message in st.session_state.messages:
        if message["role"] == "bot":
            st.markdown(f"**Chatbot:** {message['text']}")
        else:
            st.markdown(f"**You:** {message['text']}")
    
    user_input = st.text_input("You:", key="input")
    if st.button("Send"):
        if user_input:
            st.session_state.messages.append({"role": "user", "text": user_input})
            bot_reply = chatbot_response(user_input)
            st.session_state.messages.append({"role": "bot", "text": bot_reply})
            st.rerun()

elif menu == "Chatbot Interaction":
    chatbot_interaction()

elif menu == "Disease Prediction":
    st.header("Disease Prediction")

    symptoms = st.text_input("Enter your symptoms (comma-separated):")
    if st.button("Predict", key="predict"):
        if symptoms:
            symptom_list = symptoms.split(",")
            symptom_list = [s.strip() for s in symptom_list]
            
            input_vector = np.zeros(len(cols))
            for symptom in symptom_list:
                if symptom in cols:
                    input_vector[cols.get_loc(symptom)] = 1
            
            prediction = clf.predict([input_vector])[0]
            st.write(f"You may have **{prediction}**.")
            
            description = description_dict.get(prediction, "No description available.")
            st.write("Description:", description)
            
            precautions = precaution_dict.get(prediction, ["No precautions available."])
            st.write("Precautions:")
            for i, precaution in enumerate(precautions, 1):
                st.write(f"{i}. {precaution}")
            
            severity_score = sum([severity_dict.get(symptom, 0) for symptom in symptom_list])
            num_days = st.number_input("How many days have you been experiencing these symptoms?", min_value=1, step=1)
            advice = "Consult a doctor immediately." if (severity_score * num_days) / (len(symptom_list) + 1) > 13 else "Take precautions."
            st.write(f"**Advice:** {advice}")
        else:
            st.write("Please enter symptoms to get a prediction.")

elif menu == "About":
    st.header("About")
    st.write("This is a healthcare web app with a chatbot feature designed to predict diseases based on symptoms and provide precautions.")
