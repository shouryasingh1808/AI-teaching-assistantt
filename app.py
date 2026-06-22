import json
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load embeddings
with open("embedding.json", "r") as f:
    chunks = json.load(f)

df = pd.DataFrame(chunks)
model = SentenceTransformer("all-MiniLM-L6-v2")
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("🎓 RAG Teaching Assistant")
st.write("Ask me anything about the Python course!")

query = st.text_input("Your question:")

if st.button("Ask"):
    query_embedding = model.encode(query)
    df["score"] = df["embedding"].apply(lambda x: np.dot(query_embedding, np.array(x)))
    top_chunks = df.nlargest(3, "score")
    context = "\n".join(top_chunks["text"].tolist())

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are a helpful teaching assistant. Answer based on the context provided."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )

    st.write(response.choices[0].message.content)