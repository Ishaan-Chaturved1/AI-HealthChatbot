# 🧠 AI Healthcare Chatbot (ML + RAG + Conversational Memory)

💡 Ever wondered if Artificial Intelligence can act like a doctor?
This project demonstrates an **AI-powered Healthcare Chatbot** that predicts diseases from symptoms using **Machine Learning**, and enhances responses using **Retrieval-Augmented Generation (RAG)** with conversational memory.

---

## 🚑 Project Overview

This is a **complete End-to-End AI Healthcare System** that combines:

* 🤖 Machine Learning (Disease Prediction)
* 📄 RAG (Document-based medical knowledge)
* 💬 Conversational Memory (Context-aware chat)

The chatbot can:

* Understand user symptoms using NLP
* Ask intelligent follow-up questions
* Predict possible diseases with confidence score
* Retrieve relevant medical knowledge from documents (PDFs)
* Maintain conversation context across interactions
* Provide precautions, health tips, and a final motivational message

---

## 🔄 How It Works

### 1️⃣ Symptom Extraction (NLP)

* User input is processed using Natural Language Processing
* Symptoms are identified and normalized

### 2️⃣ Machine Learning Model

* Model used: **Random Forest Classifier**
* Trained on symptom-disease dataset
* Outputs:

  * Predicted disease
  * Confidence score

### 3️⃣ RAG Pipeline (Retrieval-Augmented Generation)

* Medical PDFs are ingested and processed
* Text is split into chunks and converted into embeddings
* Stored in a **FAISS Vector Database**
* Relevant medical context is retrieved based on user query

### 4️⃣ Conversational Memory

* Chat history is stored and reused
* Enables:

  * Context-aware responses
  * Follow-up question handling
  * More natural interaction

### 5️⃣ Response Generation

* Combines:

  * ML prediction
  * Retrieved medical context
  * Chat history
* Generates a final answer using LLM

---

## 🧠 Architecture

User Input
→ NLP Processing
→ ML Prediction
→ RAG Retrieval (FAISS)
→ Conversation Memory
→ LLM Response
→ Final Output

---

## ✨ Features

✅ Disease prediction using Machine Learning
✅ Context-aware chatbot with memory
✅ RAG-based knowledge retrieval from PDFs
✅ Follow-up questioning system
✅ Confidence score for predictions
✅ Health precautions & advice
✅ End motivational quote 💬

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
* Natural Language Processing (NLP)
* LangChain
* FAISS (Vector Database)
* OpenAI / LLM APIs

---

## 📂 Project Structure

```
project/
│
├── Data/
│   ├── pdf/              # Medical documents (ignored in Git)
│
├── faiss_index/          # Vector DB (ignored)
├── app.py                # FastAPI / backend
├── rag.ipynb             # Experimentation
├── requirements.txt
└── README.md
```

---

## 🚀 Key Learnings

* How to build an **end-to-end ML system**
* How to integrate **RAG with LLMs**
* Importance of **chunking & embeddings**
* Managing **conversation history in chatbots**
* Deploying AI applications

---

## ⚠️ Disclaimer

This chatbot is for **educational purposes only** and should not be used as a substitute for professional medical advice.

---

## 🔔 Future Improvements

* Add voice input/output
* Improve medical dataset accuracy
* Deploy with scalable vector database (Pinecone)
* Add authentication & user profiles

---

## 📢 Share & Support

If you found this useful:

* ⭐ Star the repo
* 📢 Share with others interested in AI & Healthcare
* 🔔 Follow for more AI/ML projects

---
