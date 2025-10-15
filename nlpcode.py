import streamlit as st
import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

# ✅ Ensure required NLTK resources are downloaded every time the app starts
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

st.title("🧠 Named Entity Recognition (NER) App")
st.write("Enter a sentence and extract entities like PERSON and GPE (locations).")

indian_names = ["Narendra", "Modi", "Sachin", "Tendulkar", "Ramesh", "Kumar"]
indian_cities = ["Mumbai", "Bengaluru", "Delhi", "Chennai", "Kolkata"]

def extract_entities(sentence):
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)
    tree = ne_chunk(tags)

    entities = {"PERSON": [], "GPE": []}

    for subtree in tree:
        if isinstance(subtree, Tree):
            label = subtree.label()
            name = " ".join([token for token, pos in subtree.leaves()])
            if label in entities:
                entities[label].append(name)

    for token in tokens:
        if token in indian_names and token not in entities["PERSON"]:
            entities["PERSON"].append(token)
        if token in indian_cities and token not in entities["GPE"]:
            entities["GPE"].append(token)

    return entities

sentence = st.text_area("Enter a sentence:")
if st.button("Extract Entities"):
    if sentence.strip():
        result = extract_entities(sentence)
        st.success("Entities extracted successfully!")
        st.write(result)
    else:
        st.warning("Please enter a sentence.")

