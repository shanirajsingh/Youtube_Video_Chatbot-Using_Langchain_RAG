import warnings

warnings.filterwarnings("ignore")

import os
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled
)

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_community.vectorstores import (
    FAISS
)

from langchain_groq import ChatGroq

# IMPORTANT FIX
from langchain.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain.chains.combine_documents import (
    create_retrieval_chain
)

from langchain_core.prompts import (
    ChatPromptTemplate
)

from langchain_core.documents import (
    Document
)


# =========================
# Load Environment Variables
# =========================

# load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

load_dotenv()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    st.secrets.get("GROQ_API_KEY", "")
)

# =========================
# Streamlit Page Config
# =========================

st.set_page_config(
    page_title="YouTube Chatbot",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube Video Chatbot")
st.write("Ask questions from any YouTube video")


# =========================
# Extract Video ID
# =========================

def extract_video_id(url):
    parsed_url = urlparse(url)

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]

    if parsed_url.hostname in (
        "www.youtube.com",
        "youtube.com"
    ):
        return parse_qs(parsed_url.query).get("v", [None])[0]

    return None


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("Video Input")

    youtube_url = st.text_input(
        "Enter YouTube Video URL"
    )
    
    process_btn = st.button("Process Video",type="primary")
    


# =========================
# Session State
# =========================

if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None


# =========================
# Process Video
# =========================

if process_btn:

    st.session_state.retrieval_chain = None
    st.session_state.generated_questions = []

    if not youtube_url:
        st.warning("Please enter YouTube URL")
        st.stop()

    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Invalid YouTube URL")
        st.stop()

    try:

        # =========================
        # Fetch Transcript
        # =========================

        SUPPORTED_LANGUAGES = [
                                "en",
                                "hi",
                                "en-IN",
                                "hi-IN",
                                "es",
                                "fr",
                                "de",
                                "it",
                                "pt",
                                "ru",
                                "ja",
                                "ko",
                                "zh-Hans",
                                "zh-Hant",
                                "ar",
                                "tr",
                                "nl",
                                "pl",
                                "sv",
                                "id",
                                "th",
                                "vi"
                            ]

        with st.spinner("Fetching transcript..."):

            transcript = YouTubeTranscriptApi().fetch(
                video_id,
                languages=SUPPORTED_LANGUAGES
                # languages=[
                #     "en",
                #     "hi",
                #     "en-IN",
                #     "hi-IN"
                # ]
            )

            transcript_text = " ".join(
                [item.text for item in transcript]
            )

        # =========================
        # Convert to Documents
        # =========================

        docs = [
            Document(
                page_content=transcript_text
            )
        ]

        # =========================
        # Text Splitting
        # =========================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        split_docs = text_splitter.split_documents(docs)

        # =========================
        # Embeddings
        # =========================

        # embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2"
        # )

        @st.cache_resource
        def load_embeddings():
            return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        embeddings = load_embeddings()

        # =========================
        # Vector Store
        # =========================

        with st.spinner("Creating vector store..."):

            vector_store = FAISS.from_documents(
                split_docs,
                embeddings
            )

        retriever = vector_store.as_retriever()

        # =========================
        # LLM
        # =========================

        # llm = ChatGroq(
        #     groq_api_key=GROQ_API_KEY,
        #     model_name="llama-3.1-8b-instant"
        # )

        @st.cache_resource
        def load_llm():
            return ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name="llama-3.1-8b-instant"
            )

        llm = load_llm()

        # =========================
        # Prompt
        # =========================

        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful AI assistant.

            Answer the user's question ONLY using the provided
            video transcript context.

            Rules:
            - Use proper grammar and natural language.
            - Answer in the same language as the user's question.
            - Use natural vocabulary and sentence structure.
            - Always answer in complete sentences.
            - Give short but meaningful explanations.
            - Do not give single-word or incomplete answers.
            - Do not hallucinate or make up information.
            - If the answer is not present in the transcript, reply exactly:
            "Sorry, I don't know the answer to that question based on the video transcript."

            <context>
            {context}
            </context>

            Question: {input}
            """
        )

        # =========================
        # Document Chain
        # =========================

        document_chain = create_stuff_documents_chain(
            llm,
            prompt
        )

        # =========================
        # Retrieval Chain
        # =========================


        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )
        
        retrieval_chain = create_retrieval_chain(
            retriever,
            document_chain
        )

        st.session_state.retrieval_chain = retrieval_chain

        # =========================
        # Generate Suggested Questions
        # =========================

        with st.spinner("Generating suggested questions..."):

            question_prompt =   """
                                    Generate 5 short natural questions from the video transcript.

                                    Rules:
                                    - Use the same language as the transcript.
                                    - Use proper grammar and natural wording.
                                    - Keep each question short (6-10 words).
                                    - Generate only questions answerable from the transcript.
                                    - Do not generate unrelated or assumed questions.

                                    Output Rules:
                                    - Output ONLY questions.
                                    - No answers.
                                    - No introductions.
                                    - No explanations.
                                    - No headings.
                                    - No numbering.
                                    - No bullet points.
                                    - No extra text.
                                    - Do not use phrases like:
                                    "Here are 5 questions"
                                    "Question Asked"

                                    Content Rules:
                                    - If the transcript mainly contains repeated lines, rhyming patterns, chorus-like structure, singing phrases, musical expressions, or lyrical content, treat it as a SONG and generate questions related to:
                                    - meaning
                                    - emotions
                                    - lyrics
                                    - theme
                                    - artist
                                    - overall message

                                    - If the transcript mainly contains explanations, conversations, teaching, storytelling, interviews, podcasts, tutorials, or informational speech, treat it as SPEECH and generate context-aware informational questions.

                                    Context:
                                    {context}
                                """

            question_response = retrieval_chain.invoke(
                {
                    "input": question_prompt
                }
            )

            generated_questions = (
                question_response["answer"]
                .split("\n")
            )

            cleaned_questions = []

            for question in generated_questions:

                question = (
                    question
                    .replace("-", "")
                    .strip()
                )

                if question:
                    cleaned_questions.append(
                        question
                    )

            st.session_state.generated_questions = (
                cleaned_questions
            )

        st.success("Video processed successfully!")

    # except TranscriptsDisabled:
    #     st.error(
    #         "Transcript is disabled for this video"
    #     )
    
    except TranscriptsDisabled:
        st.session_state.youtube_input = ""
        st.session_state.retrieval_chain = None

        st.error(
            "Transcript is disabled for this video"
    )

    # except Exception as e:
    #     st.session_state.youtube_input = ""
    #     st.session_state.retrieval_chain = None
    #     st.error(f"Error: {str(e)}")
    except Exception:

        st.session_state.youtube_input = ""
        st.session_state.retrieval_chain = None

        st.toast(
            "Failed to process video",
            icon="❌"
        )

        st.error(
            "Something went wrong, Please try another YouTube video."
    )

# =========================
# Chat Section
# =========================

if st.session_state.retrieval_chain:

    st.divider()

    st.subheader("🎯 Ask Questions About Video")

    # =========================
    # Common Questions
    # =========================

    common_questions = [
        "Select a common question",
        "Summarize the video",
        "Give key points from the video",
        "Explain the video like I'm a beginner",
        "What are the important topics discussed?",
        "Generate interview questions from this video",
        "What are the main conclusions?",
        "Give action items from the video",
        "Generate quiz questions from this video"
    ]

    user_question = None

    # =========================
    # Suggested Questions
    # =========================

    if "generated_questions" in st.session_state:

        st.subheader(
            "💡 AI Suggested Questions"
        )

        cols = st.columns(2)

        for idx, question in enumerate(
            st.session_state.generated_questions
        ):

            with cols[idx % 2]:

                if st.button(
                    question,
                    key=f"question_{idx}"
                ):

                    user_question = question

    # =========================
    # Question Form
    # =========================

    with st.form(
        "question_form",
        clear_on_submit=True
    ):

        selected_question = st.selectbox(
            "Choose a common question",
            common_questions
        )

        custom_question = st.text_input(
            "Or Ask Your Own Question"
        ).strip()

        submit_button = st.form_submit_button(
            "Ask Question"
        )

        if submit_button:

            if (
                selected_question
                != "Select a common question"
            ):

                user_question = selected_question

            if custom_question:
                user_question = custom_question

    # =========================
    # Generate Response
    # =========================

    if user_question:

        with st.spinner(
            "Generating answer..."
        ):

            response = (
                st.session_state
                .retrieval_chain
                .invoke(
                    {
                        "input": user_question
                    }
                )
            )

        # =========================
        # Show Asked Question
        # =========================

        st.subheader("🧠 Answer")

        answer = response["answer"]
        # print(f"Generated Answer: {answer}")
        if not answer:
            answer = "Sorry, I could not generate an answer."

        st.write(
            answer
        )

        st.caption(
                "LLMs can make mistakes. Please verify important information."
        )
