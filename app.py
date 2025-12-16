import sys
import os
import random  # <--- Random selection ke liye zaroori
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
DATA_PATH = "data/"
MODEL_FILE = "model_0.2.Q2_K.gguf"

print("🔄 Initializing System...")

# Folder Check
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
    print(f"⚠️ Warning: '{DATA_PATH}' folder not found. Created it. Please paste 'profile.txt' inside.")

# ==========================================
# 2. DATA LOADING & PROCESSING
# ==========================================
print("📂 Loading Data...")
try:
    loader = DirectoryLoader(DATA_PATH, glob='*.txt', loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print("❌ Error: Data folder is empty! Please add 'profile.txt'.")
        sys.exit()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"✅ {len(texts)} chunks processed.")

except Exception as e:
    print(f"❌ Data Error: {e}")
    sys.exit()

# ==========================================
# 3. DATABASE CREATION
# ==========================================
print("🧠 Building Memory (Embeddings)...")
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                   model_kwargs={'device': 'cpu'})

db = FAISS.from_documents(texts, embeddings)
print("✅ Database Ready!")

# ==========================================
# 4. LLM MODEL SETUP
# ==========================================
print("🤖 Loading Model... (Please wait)")
try:
    llm = CTransformers(
        model=MODEL_FILE,
        model_type="mistral",
        config={
            'max_new_tokens': 150,
            'temperature': 0.4,
            'repetition_penalty': 1.1,
            'context_length': 2048
        }
    )
except Exception as e:
    print(f"❌ Model Error: '{MODEL_FILE}' not found.")
    sys.exit()

# ==========================================
# 5. PROMPT TEMPLATE (English Focused)
# ==========================================
qa_template = """[INST] You are a professional AI Assistant for Prakash Bokarvadiya.

RULES:
1. Answer strictly based on the Context provided below.
2. Answer ONLY in English.
3. Keep answers professional, concise (2-3 sentences), and friendly.
4. Do NOT start with "Note:" or "Based on the context".

Context: {context}
User Question: {question}

Answer:
[/INST]
"""

PROMPT = PromptTemplate(template=qa_template, input_variables=['context', 'question'])

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 2}),
    chain_type_kwargs={'prompt': PROMPT},
    return_source_documents=False
)

# ==========================================
# 6. MAIN CHAT LOOP (English & Expanded)
# ==========================================
print("\n" + "="*50)
print("🚀 Prakash's Chatbot is Online!")
print("Type 'bye' to exit.")
print("="*50)

# List of varied English greetings for the BOT's response
bot_greetings = [
    # General/Friendly
    "Hello! 👋 I am Prakash's AI Assistant. How can I help you today?",
    "Hi there! I'm ready to answer questions about Prakash's projects and skills.",
    "Greetings! 🙏 You can ask me about Prakash's Education or Experience.",
    "Hello! I am here to assist you. What would you like to know?",
    "Hey! 👋 I'm listening. Feel free to ask about my Data Science background.",
    "Welcome! I can provide details on Prakash's Python and ML skills.",
    "Hi! I represent Prakash Bokarvadiya. How can I be of service?",
    
    # Professional/Direct
    "Good day! I'm here to provide professional insights into Prakash Bokarvadiya's profile.",
    "Thank you for reaching out. Which aspect of Prakash's professional background interests you?",
    "Greetings. I am an AI designed to articulate Prakash's experience and qualifications. How can I assist?",
    "Welcome to Prakash's professional portfolio interface. Feel free to ask your specific queries.",

    # Skill/Project Focused
    "Hello! I specialize in answering questions about Prakash's Machine Learning and Deep Learning projects.",
    "Hi! Let's talk data. Ready to explore Prakash's proficiency in Python, Pandas, and Scikit-learn?",
    "Welcome! Ask me anything about Prakash's experience in building end-to-end data pipelines.",
    "Greetings! I can share details on how Prakash tackled the customer churn prediction model at TechSolutions.",

    # Context-Aware (Assuming a resume context)
    "Hello! Are you interested in Prakash's work history or his technical skill set?",
    "Hi! I can walk you through Prakash's educational achievements or his latest projects.",
    "Welcome! To get started, you might ask about his role as a Lead Data Analyst or his cloud experience.",

    # Enthusiastic/Motivational
    "Hey there! I'm fully charged and ready to showcase Prakash's capabilities. What's on your mind?",
    "Fantastic! Let's dive into the world of Data Science. What question do you have for Prakash's profile?",
    "A pleasure to connect! I'm the key to unlocking Prakash's professional story. Ask away!"
]

# List of common USER inputs to trigger a quick greeting
user_greetings = [
    # Standard English Greetings
    "hi", 
    "hello", 
    "hey", 
    "greetings",
    
    # Time-Based Greetings
    "good morning",
    "good afternoon",
    "good evening",
    "morning",
    "afternoon",
    "evening",

    # Informal/Slang
    "howdy",
    "hola", 
    "sup",          # Short for 'what's up'
    "yo",           
    "hallo",        # Alternative spelling
    "what's up", 
    
    # Common Conversational Starters/Status Checks
    "how are you",
    "how's it going",
    "how's you",
    "how are things",
    "what's new",
    
    # Hindi/Indian Greetings (kept for multilingual robustness)
    "kaise ho",     
    "namaste",      
    "namaskar",     
    "pranam",       
    "ram ram"
]

while True:
    user_input = input("\nYou: ").strip()
    
    if not user_input:
        continue

    # Exit Logic
    if user_input.lower() in ["bye", "exit", "quit", "goodbye"]:
        print("Chatbot: Goodbye! 👋 Have a great day!")
        break

    # --- FAST GREETING FILTER ---
    # Check if user input is an exact match for a greeting (case-insensitive)
    if any(g == user_input.lower() for g in user_greetings) or (user_input.lower() in user_greetings):
        print("Chatbot:", random.choice(bot_greetings))
        continue

    # --- AI QUERY ---
    try:
        response = qa_chain.invoke({"query": user_input})
        final_answer = response["result"]

        # Cleaning (Ensures prompt rules are strictly followed)
        if "Note:" in final_answer:
            final_answer = final_answer.split("Note:")[0]
        if "[Note" in final_answer:
            final_answer = final_answer.split("[Note")[0]

        print("Chatbot:", final_answer.strip())

    except Exception as e:
        print(f"Error: {e}")