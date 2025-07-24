# src/rag_pipeline.py

import os
import pickle
from typing import List
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load vector store
def load_vector_store(path: str = "../vector_store/index.pkl") -> FAISS:
    with open(path, "rb") as f:
        return pickle.load(f)

# Retrieve top-k context chunks
def retrieve_context(query: str, vector_store: FAISS, k: int = 5) -> List[str]:
    docs = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]

# Build the prompt template
def get_prompt_template() -> PromptTemplate:
    template = """You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}
Answer:"""
    return PromptTemplate.from_template(template)

# Generator: uses prompt, context, question to generate answer
def generate_answer(question: str, context_chunks: List[str], llm) -> str:
    context = "\n\n".join(context_chunks)
    prompt = get_prompt_template()
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=None,  # we are manually providing context
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return chain.combine_documents_chain.run(
        {"context": context, "question": question}
    )

# Setup LLM (can replace with Mistral, Llama, etc.)
def load_llm(model_name: str = "mistralai/Mistral-7B-Instruct-v0.1"):
    from transformers import pipeline
    pipe = pipeline("text-generation", model=model_name, max_new_tokens=256)
    return HuggingFacePipeline(pipeline=pipe)
