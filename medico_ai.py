

import os
import streamlit as st
import time
import streamlit.components.v1 as components
from dotenv import load_dotenv
load_dotenv()
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# --- Configuration ---
DB_FAISS_PATH = "/Users/anukulchandra/Medico AI/vectorstore/faiss_db" 
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "HuggingFaceH4/zephyr-7b-beta"

# --- RAG PROMPT TEMPLATE ---
rag_prompt_template = """<|system|>
You are Medico AI, a professional medical assistant. 

INSTRUCTIONS:
1. ANSWERING: Answer based ONLY on the Context provided below.
2. SEMANTICS: If the user asks for a "cure" and the context contains "treatments" or "therapies", provide those as the answer.
3. FORMAT: 
   - Start directly with the answer.
   - Use bullet points for lists.
4. MISSING DATA: If the answer is truly missing, say: "I cannot find specific details in the provided documents."
5. CLARIFICATION: If the question is unclear, ask for clarification instead of guessing.


Context:
{context}
</s>
<|user|>
{question}
</s>
<|assistant|>
"""

# --- Cached Resources ---
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    try:
        if not os.path.exists(DB_FAISS_PATH):
            st.error(f"Error: Database folder '{DB_FAISS_PATH}' not found.")
            return None
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

@st.cache_resource
def get_inference_client():
    if not HF_TOKEN:
        st.error("HF_TOKEN environment variable is missing!")
        return None
    return InferenceClient(model=REPO_ID, token=HF_TOKEN)

# --- Greeting Logic ---
def get_static_response(query):
    query = query.lower().strip()
    greetings = ["hi", "hello", "hey", "hlw", "salam", "namaskar", "good morning", "good evening"]
    farewells = ["bye", "good night", "tata", "see you"]
    normal_msgs = ["how are you", "what's up", "how's it going", "how are you doing"]
    thanks_msgs = ["thank you", "thanks", "thx", "thank you so much", "thanks a lot"]
    ok_msgs = ["ok", "okay", "fine", "alright", "got it", "understood","No problem","Sure","Sounds good","No thanks","That's helpful","Appreciate it","Great","Perfect","Will do","Thanks for the info","Noted","I see","Alrighty","Cool","Good to know"]
    
    if query in greetings or any(query.startswith(g) and len(query) < 15 for g in greetings):
        return "Hello! I am **Medico AI** 🩺. I can answer medical questions based on your documents. How can I help?"
    
    if any(bye in query for bye in farewells):
        return "Goodbye! Stay healthy. 👋"
    if any(normal in query for normal in normal_msgs):
        return "I'm doing well, thank you for asking! How can I assist you today?"
    if any(thanks in query for thanks in thanks_msgs):
        return "You're welcome! Is there anything else I can help you with?"
    if any(ok in query for ok in ok_msgs):
        return "Great! Let me know if you have any medical questions or need assistance with your documents."
    
    return None

# --- Streaming Logic ---
def generate_response_stream(client, prompt_text):
    messages = [{"role": "user", "content": prompt_text}]
    stream = client.chat_completion(
        messages=messages, max_tokens=600, temperature=0.3, stream=True
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def auto_scroll():
    # এই জাভাস্ক্রিপ্ট কোডটি চ্যাট উইন্ডোকে জোর করে নিচে নামিয়ে দেবে
    js = """
    <script>
        var body = window.parent.document.querySelector(".main");
        body.scrollTop = body.scrollHeight;
    </script>
    """
    components.html(js, height=0, width=0)

# --- MAIN APPLICATION ---
def main():
    st.set_page_config(page_title="Medico AI", page_icon="🩺", layout="wide")

    
    st.markdown("""
    <style>
        
        .stMarkdown p, div[data-testid="stChatMessage"] p {
            color: #000000 !important; /* একদম কালো রং */
            font-weight: 500 !important; /* লেখা একটু মোটা */
        }
        
        ul, ol, li {
            color: #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar 
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3774/3774299.png", width=100)
        st.title("Medico AI 🩺")
        st.markdown("---")
        
        st.markdown("**System Status:** \n🟢 Online & Ready")
        st.info("💡 **Note:** I only answer based on the provided PDF documents.")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- Header ---
    st.markdown("""
    <h1 style='text-align: center; color: #2E86C1;'>🩺 Medico AI Assistant</h1>
    <p style='text-align: center; color: #444444;'>Your Strict Medical Documentation Partner</p>
    """, unsafe_allow_html=True)

    db = get_vectorstore()
    client = get_inference_client()

    if db is None or client is None:
        return

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # --- INPUT ---
    if prompt := st.chat_input("Ask a medical question..."):
        
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        with st.chat_message('assistant'):
            # --- STATUS (Thinking...) ---
            status_box = st.empty()
            status_box.markdown("⏳ **Thinking...**")
            
            try:
                static_reply = get_static_response(prompt)
                
                if static_reply:
                    status_box.empty()
                    def stream_static():
                        for word in static_reply.split(" "):
                            yield word + " "
                            time.sleep(0.05)
                    response = st.write_stream(stream_static)
                    source_docs = []
                
                else:
                    # Searching Status
                    status_box.markdown("🔍 **Searching medical records...**")
                    # time.sleep(0.5) # Speed er jonno comment kora thakuk
                    
                    retriever = db.as_retriever(search_kwargs={'k': 6})
                    source_docs = retriever.invoke(prompt)
                    
                    # Answering Status
                    status_box.markdown("💡 **Formulating answer...**")
                    
                    context = "\n\n".join([doc.page_content for doc in source_docs])
                    prompt_text = rag_prompt_template.format(context=context, question=prompt)
                    
                    # Stream Answer
                    status_box.empty()
                    response = st.write_stream(generate_response_stream(client, prompt_text))
                    
                    # Sources
                    if source_docs:
                        with st.expander("📚 Reference Context"):
                            for i, doc in enumerate(source_docs, 1):
                                st.caption(f"**Source {i}:** {doc.page_content[:300]}...")

            except Exception as e:
                status_box.error(f"Error: {e}")
                response = "Sorry, I encountered an error."

        st.session_state.messages.append({'role': 'assistant', 'content': response})

if __name__ == "__main__":
    main()