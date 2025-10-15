# ner_app.py

import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree
import streamlit as st

# Download necessary data once
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Predefined name and city lists
indian_names = ["Narendra", "Modi", "Sachin", "Tendulkar", "Ramesh", "Kumar"]
indian_cities = ["Mumbai", "Bengaluru", "Delhi", "Chennai", "Kolkata"]

# Entity extraction function
def extract_entities(sentence):
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)
    tree = ne_chunk(tags)
    entities = {"PERSON": [], "GPE": []}

    # Extract entities from tree
    for subtree in tree:
        if isinstance(subtree, Tree):
            label = subtree.label()
            name = " ".join([token for token, pos in subtree.leaves()])
            if label in entities:
                entities[label].append(name)

    # Rule-based Indian name/city detection
    for token in tokens:
        if token in indian_names and token not in entities["PERSON"]:
            entities["PERSON"].append(token)
        if token in indian_cities and token not in entities["GPE"]:
            entities["GPE"].append(token)

    return entities

# Streamlit UI
st.title("🧠 Named Entity Recognition (NER) System")
st.markdown("**Built using NLTK and Rule-based Methods — No spaCy Required!**")

# Input box
sentence = st.text_area("Enter a sentence:", "Narendra Modi is the Prime Minister of India.")

# Process button
if st.button("Extract Entities"):
    result = extract_entities(sentence)
    st.subheader("🔍 Extracted Entities")
    st.json(result)

# Footer
st.markdown("---")
st.caption("Developed using NLTK • Hybrid (Statistical + Rule-Based) NER System")
