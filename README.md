# 🧠 Intelligent Complaint Analysis for Financial Services

### RAG-Powered Chatbot to Turn Feedback into Product Insights

---

## 📍 Overview

CrediTrust Financial serves over 500,000 customers across East Africa with financial products like credit cards, loans, BNPL, savings, and transfers. With thousands of monthly complaints coming through, teams are overwhelmed and insights are buried in text.

This project builds an **internal complaint analysis assistant** using **Retrieval-Augmented Generation (RAG)** to allow product, support, and compliance teams to query feedback in plain English and get useful, context-grounded answers.

---

## ⚙️ What This Repo Does

* ✅ Cleans and filters real-world financial complaint data
* ✅ Chunks and embeds narratives using `sentence-transformers`
* ✅ Stores embeddings in a FAISS/ChromaDB vector store
* ✅ Retrieves relevant complaints using semantic search
* ✅ Generates intelligent answers using LLMs (via LangChain/Hugging Face)

---

## 🗂️ Project Structure

```bash
.
├── data/                         # Cleaned dataset (filtered_complaints.csv)
│
├── notebooks/                    # Development notebooks
│   ├── EDA.ipynb                 # Exploratory analysis
│   ├── chunking.ipynb            # Text splitting tests
│   ├── build_index_modular.ipynb # Step-by-step vector indexing
│   └── embed-index.ipynb         # Embedding + storage
│
├── scripts/                     # Modular Python logic
│   ├── preprocessing.py         # Data cleaning & filtering
│   ├── EDA.py                   # Visualization logic
│   ├── chunking.py              # Text splitting utilities
│   └── build_vector_store.py    # Embedding + FAISS/Chroma indexing
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🧪 How to Use

1. **Clone the repo**

   ```bash
   git clone https://github.com/helina1abebe/Intelligent-Complaint-Analysis.git
   cd Intelligent-Complaint-Analysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Preprocess and clean the data**

   ```bash
   python scripts/preprocessing.py
   ```

4. **Build the vector store**

   ```bash
   python scripts/build_vector_store.py
   ```

5. **(Coming soon)** Run the RAG chatbot and ask questions

---

## 💡 Example Use Case

**Q:** *“What are the most common BNPL complaints?”*
**A:** *“Unexpected charges, unclear repayment terms, and automatic deductions are frequently mentioned in customer feedback.”*

---

## 📌 Tech Stack

* Python 3.10+
* LangChain
* Hugging Face Transformers
* Sentence-Transformers
* FAISS or ChromaDB
* Pandas, Seaborn, Matplotlib

---


## 📘 CFPB Dataset

Source: [Consumer Finance Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/)

Fields used:

* `Product`
* `Consumer complaint narrative`
* `Company`
* `Date received`

---

## 🧠 Next Steps

* [ ] Add the RAG retrieval + generation logic
* [ ] Evaluate with 5–10 benchmark questions
* [ ] Optional: Add a Gradio/Streamlit UI

