import os
import numpy as np
from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(
    api_key= os.getenv("GEMINI_API_KEY")
)

def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return np.array(response.embeddings[0].values)

def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )

def generate_answer(query , retrieved_chunks):

    context = "\n\n".join(
        item["text"]
        for item in retrieved_chunks
    )

    prompt = f"""
    You are a helpful assistant.

    Answer the user's question using ONLY the
    information provided in the context.

    If the answer cannot be found in the context,
    say "I don't know based on the provided document."

    Context:
    ----------------
    {context}
    ----------------

    Question:
    {query}

    Answer:
    """

    response = client.models.generate_content(
        model = "gemin-2.5-flash",
        contents=prompt
    )

    return response.text