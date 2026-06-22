import json
from sentence_transformers import SentenceTransformer

# Load dataset
with open("combined.json" , "r") as f:
    chunks = json.load(f)

# Model --
model = SentenceTransformer("all-MiniLM-L6-v2")

for chunk in chunks :
    chunk['embedding'] = model.encode(chunk["text"]).tolist()

# Save 
with open("embedding.json" , "w") as f:
    json.dump(chunks, f , indent=4)

print("Done json file saved!!")