# 🎥 YouTube Video Chatbot

An AI-powered YouTube Video Chatbot built with Streamlit, LangChain, Groq, and Hugging Face that allows users to:

- Extract YouTube video transcripts
- Ask questions about videos
- Automatic Generate AI-suggested questions related to video
- Summarize video content
- Chat with video using chat prompt

---

# 🚀 Features

- 🎥 YouTube transcript extraction
- 🤖 AI-powered question answering
- 🧠 RAG (Retrieval-Augmented Generation)
- 💬 Natural language responses
- 🌍 Multi-language transcript support
- ⚡ Fast inference using Groq LLM
- 🔍 Vector search using FAISS
- 💡 AI-generated suggested questions
- 🎨 Clean Streamlit UI

---

# 🛠️ Tech Stack

- Python
- Streamlit
- LangChain Classic
- Groq API
- HuggingFace Embeddings
- FAISS Vector Store
- YouTube Transcript API

---

# 🌐 Supported Languages

The chatbot supports transcripts in multiple languages including:

* English
* Hindi
* Spanish
* French
* German
* Japanese
* Korean
* Chinese
* Arabic
* Russian
* Turkish
* And more...

---

# 🧠 How It Works

* User enters YouTube video URL
* Transcript is fetched using YouTube Transcript API
* Transcript is split into chunks
* Embeddings are created using HuggingFace
* FAISS vector database stores embeddings
* User questions are matched with relevant transcript chunks
* Groq LLM generates context-aware answers

---

## 🎯 Ask Questions

Users can ask questions like:

* Summarize the video
* Explain key concepts
* Generate quiz questions
* Generate interview questions
* Ask custom questions

---
# 📂 Project Structure

```bash
youtube-video-chatbot/
│
├── app.py
├── requirements.txt
├── .env
└── README.md
---

![App UI](https://github.com/shanirajsingh/Youtube_Video_Chatbot-Using_Langchain_RAG/blob/main/YT_chatbot_pic1.png?raw=true)
