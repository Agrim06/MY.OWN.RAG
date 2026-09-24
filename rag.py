from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
import chromadb

from dotenv import load_dotenv
import os
import numpy as np


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
client = chromadb.Client()

embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        output_dimensionality=768
    )

def load_pdf(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    return docs
    

def chunking(docs, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = text_splitter.split_documents(docs)
    
    return chunks


def get_embedding(docs):
    return embeddings.embed_query(docs)


def create_vector_store(chunks):
    
    vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    collection_name="RAG_DB",
    persist_directory="./chroma_db",
)
    return vector_store


def retrieve(query, vector_store, k=3):
    return vector_store.similarity_search(query, k=k)


def cosine_similarity(vector_a, vector_b):
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )

def generate_answer(query , retrieved_chunks):

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_chunks
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

    response = model.invoke(prompt)

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
    vector_store = create_vector_store(chunks)

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

        for i, item in enumerate(retrieved_chunks):

            print(
                f"\n--- Chunk {i + 1} ---"
            )

            print(
                item.page_content
            )

        print(
            "\nGenerating answer..."
        )

        answer = generate_answer(query, retrieved_chunks)


        print("\n====================")

        print("ANSWER")

        print("====================")

        print(answer)


if __name__ == "__main__":

    main()


    