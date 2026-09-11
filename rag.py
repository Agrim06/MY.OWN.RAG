import os
import numpy as np
from google import genai
from dotenv import load_dotenv
import pymupdf as fitz


load_dotenv()

client = genai.Client(
    api_key= os.getenv("GEMINI_API_KEY")
)


def load_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text() + "\n"
    
    return text

def chunking(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    
    return chunks


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return np.array(response.embeddings[0].values)


def create_vector_store(chunks):
    vector_store = []

    for i , chunk in enumerate(chunks):
        print(f"Creating embedding {i + 1}/{len(chunks)}")

        embedding = get_embedding(chunk)

        vector_store.append({
            "text": chunk,
            "embedding": embedding
        })

    return vector_store


def retrieve(query, vector_store, k=3):

    query_embedding = get_embedding(query)

    results = []

    for item in vector_store:
        score = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        results.append({
            "text": item["text"],
            "score": score
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:k]


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
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


def main():
    pdf_path = "docs/doc.pdf"

    print(">>>>>>Loading PDF>>>>>")

    text = load_pdf(pdf_path)
    print(
        f"PDF loaded."
        f" Characters: {len(text)}"
    )

    chunks = chunking(text)

    print(f"Created {len(chunks)} chunks")

    print("\nCreating embeddings...")
    vector_store = create_vector_store(
        chunks
    )

    print("\nVector store created!")

    while True:

        query = input(
            "\nAsk a question "
            "(type 'exit' to quit): "
        )


        if query.lower() == "exit":

            break

        print(
            "\nRetrieving relevant chunks..."
        )

        retrieved_chunks = retrieve(
            query,
            vector_store,
            k=3
        )

        print("\nRetrieved chunks:")

        for i, item in enumerate(
            retrieved_chunks
        ):

            print(
                f"\n--- Chunk {i + 1} "
                f"| Score: "
                f"{item['score']:.4f} ---"
            )

            print(
                item["text"][:500]
            )

        print(
            "\nGenerating answer..."
        )

        answer = generate_answer(

            query,

            retrieved_chunks

        )


        print("\n====================")

        print("ANSWER")

        print("====================")

        print(answer)


if __name__ == "__main__":

    main()


    