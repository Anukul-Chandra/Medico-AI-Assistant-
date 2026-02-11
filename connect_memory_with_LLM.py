
import os
import sys 
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient

# Step 1: Setup LLM
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN is not set in the environment variables.")

huggingface_repo_id = "HuggingFaceH4/zephyr-7b-beta"
#huggingface_repo_id = "mistralai/Mistral-7B-Instruct-v0.2" 


# Use InferenceClient directly
client = InferenceClient(model=huggingface_repo_id, token=HF_TOKEN)

# Step 2: Connect Memory
DB_PATH = "vectorstore/faiss_db"

custom_prompt_template = """You are a helpful medical assistant. You must answer the user's question based ONLY on the provided context below.

IMPORTANT RULES:
1. Use ONLY the context provided to answer.
2. If the answer is NOT in the context, simply say: "I cannot find the answer in the provided documents."
3. Do NOT make up any information or use outside knowledge.
4. If the answer is long, use bullet points.
5. Keep the answer concise and professional.
6. If the question is not related to the medical context, politely decline to answer and say: "Sorry, I couldn’t find relevant information for your question. I can only assist with queries related to the provided medical documents.."

Context: {context}

Question: {question}

Answer:"""

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

try:
    if not os.path.exists(DB_PATH):
        print(f"Error: Database folder '{DB_PATH}' not found. Run memory_for_llm.py first.")
        sys.exit(1)
        
    db = FAISS.load_local(DB_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading database: {e}")
    sys.exit(1)

# Step 3: Manual QA Function
retriever = db.as_retriever(search_kwargs={"k": 3})

def answer_question(question):
    """Manual RAG implementation"""
    # Get relevant documents
    docs = retriever.invoke(question)
    
    # Combine context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create the prompt
    prompt = custom_prompt_template.format(context=context, question=question)
    
    # Use chat_completion instead of text_generation
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = client.chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.5
    )
    
    # Extract the answer from the response
    answer = response.choices[0].message.content
    
    return answer, docs

# Step 4: Run
if __name__ == "__main__":
    print("Bot is ready! (Type 'exit' to stop)")
    while True:
        user_query = input("\nWrite Query Here: ")
        if user_query.lower() == "exit":
            break
            
        if user_query:
            print("Thinking...")
            try:
                answer, source_docs = answer_question(user_query)
                print(f"\n=== Answer ===\n{answer}")
                
                # Optional: Show sources
                print("\n--- Source Documents ---")
                for i, doc in enumerate(source_docs, 1):
                    print(f"{i}. {doc.page_content[:150]}...")
            except Exception as e:
                print(f"Error: {e}")