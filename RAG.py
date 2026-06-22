import json
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from groq import Groq

with open("embedding.json" , "r") as f:
    chunks = json.load(f)

df = pd.DataFrame(chunks)

model = SentenceTransformer("all-MiniLM-L6-v2")

import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Your question
query = input("Ask a question -- ")
query_embeddings = model.encode(query)


# find most similar chunk
df["score"] = df["embedding"].apply(lambda x: np.dot(query_embeddings , np.array(x)))
# Top three results --
top_chunks = df.nlargest(3,"score")

# Combine chunks
context = "\n".join(top_chunks["text"].to_list())
response = client.chat.completions.create(
    model = "llama-3.1-8b-instant",
    messages= [
        {"role" : "system" , "content" : "you are a helpful teaching assistant. Answer based on the context provided."},
        {"role" : "user" , "content" : f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)

print(response.choices[0].message.content)