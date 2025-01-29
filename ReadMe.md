# Q&A Chatbot

## 📌 Overview
Q&A chatbot is an AI-powered chatbot that answers your questions quickly by retrieving question and answer pairs from the uploaded **CSV file**.  
The implementation leverages **Retrieval-Augmented Generation (RAG)** to:
1. **Retrieve relevant Q&A pairs from the uploaded CSV**.
2. **Pass the retrieved context to an LLM (Large Language Model)** for accurate responses.

## 🚀 Features
✅ Upload a CSV file containing questions and answers.  
✅ Store and retrieve data using **FAISS Vector Database**.  
✅ Use **Google Generative AI (`gemini-2.0-flash-exp`)** for intelligent responses.  
✅ **Streamlit UI** for an interactive chatbot experience.  
✅ **Efficient session handling** to avoid unnecessary vector database recreation.

---

## 🛠️ Installation & Setup

### 1. **Clone the Repository**
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. **Create a Virtual Environment (Optional but Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Set Up API Keys**
- Create a .env file in the project directory.
- Add your Google Gemini API Key:
```bash
GEMINI_API_KEY=your_google_gemini_api_key
```

### 5. **Run the Chatbot**
```bash
streamlit run main.py
```

---

## ⚙️ Implementation Details

### **1️⃣ CSV File Upload**
- Users upload a **CSV file** containing Q&A pairs.
- The file is **saved locally** in an `uploaded_files/` directory.
- If a file with the same name is uploaded again, it is **not reprocessed** to save resources.

### **2️⃣ Vector Database Creation (FAISS)**
- The CSV is converted into **text embeddings** using `HuggingFaceEmbeddings`.
- A **FAISS vector store** is created to efficiently **retrieve similar Q&A pairs**.

### **3️⃣ Query Processing & RAG Flow**
1. The user enters a question in the chatbot.
2. The **retriever searches for similar Q&A pairs** in the FAISS database.
3. The retrieved results are **passed to the LLM (`gemini-2.0-flash-exp`)**.
4. The LLM generates a response **only based on the retrieved context**.
5. The chatbot returns the **final answer**.


---

## 🏗️ Tech Stack
- **Python** 🐍  
- **Streamlit** 🎛️ (Frontend UI)  
- **FAISS** 🔍 (Vector Search)  
- **HuggingFace Embeddings** 🧠 (Text Embeddings)  
- **Google Generative AI (Gemini 2.0)** 🤖 (LLM)  
- **LangChain** 🏗️ (RAG Pipeline)  