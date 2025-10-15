import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import re
import streamlit as st

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Predefined lists
indian_names = ["Narendra", "Modi", "Sachin", "Tendulkar", "Ramesh", "Kumar"]
indian_cities = ["Mumbai", "Bengaluru", "Delhi", "Chennai", "Kolkata"]

# Regex patterns
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
phone_pattern = r'(\+?\d{1,3}[\s-]?)?\d{10}'

def extract_entities_regex(sentence):
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)

    entities = {"PERSON": [], "GPE": [], "EMAIL": [], "PHONE": []}

    # --- PERSON detection ---
    # Look for proper nouns (NNP) sequences
    person_tokens = []
    for word, pos in tags:
        if pos == "NNP":
            person_tokens.append(word)
        else:
            if person_tokens:
                name = " ".join(person_tokens)
                if name not in entities["PERSON"]:
                    entities["PERSON"].append(name)
                person_tokens = []
    if person_tokens:
        name = " ".join(person_tokens)
        if name not in entities["PERSON"]:
            entities["PERSON"].append(name)

    # --- Add Indian names manually ---
    for token in tokens:
        if token in indian_names and token not in entities["PERSON"]:
            entities["PERSON"].append(token)

    # --- GPE detection (cities) ---
    for token in tokens:
        if token in indian_cities and token not in entities["GPE"]:
            entities["GPE"].append(token)

    # --- Emails & Phones using regex ---
    emails = re.findall(email_pattern, sentence)
    phones = re.findall(phone_pattern, sentence)
    entities["EMAIL"] = emails
    entities["PHONE"] = phones

    return entities

# --- Streamlit App ---
st.title("📄 Regex + POS NER App")
st.write("Extract PERSON, GPE, EMAIL, and PHONE entities from a sentence.")

sentence = st.text_area("Enter a sentence:")
if st.button("Extract Entities"):
    if sentence.strip():
        result = extract_entities_regex(sentence)
        st.success("Entities extracted successfully!")
        st.json(result)
    else:
        st.warning("Please enter a sentence.")

