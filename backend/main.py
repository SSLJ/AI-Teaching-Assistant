import os

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory


# -------------------------------
# FASTAPI
# -------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    message: str

# -------------------------------
# API KEY
# -------------------------------
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# -------------------------------
# LOAD SYLLABUS PDF
# -------------------------------

loader = PyPDFLoader("Documents/AI_rag.pdf")
documents = loader.load()


# -------------------------------
# SPLIT DOCUMENT
# -------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

docs = splitter.split_documents(documents)


# -------------------------------
# EMBEDDINGS
# -------------------------------

embeddings = HuggingFaceEmbeddings()


# -------------------------------
# VECTOR DATABASE
# -------------------------------

vectorstore = Chroma.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()


# -------------------------------
# LLM
# -------------------------------

llm = ChatGroq(
    model_name="llama-3.1-8b-instant"
)


# -------------------------------
# RAG QA CHAIN
# -------------------------------

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)


# -------------------------------
# TOOL 1: SYLLABUS QA
# -------------------------------

last_topic = {"value": ""}  # simple state tracker

def retrieve_syllabus(question):
    last_topic["value"] = question  # update last topic
    prompt = f"""
Answer the student's question clearly.

Use:
- short paragraphs
- bullet points when possible
- simple explanations

Question: {question}
"""
    result = qa_chain.invoke({"query": prompt})
    return result["result"]


# -------------------------------
# TOOL 2: QUIZ GENERATOR
# -------------------------------
def generate_quiz(topic):
    # Clean up topic if agent passes a full sentence
    topic = topic.strip().strip('"').strip("'")
    
    # Fallback to last known topic if input is vague
    if len(topic) < 4 or topic.lower() in ["that", "this", "it", "none", ""]:
        topic = last_topic["value"] if last_topic["value"] else topic

    prompt = f"""You are an AI teaching assistant.

Generate exactly 3 quiz questions specifically about: {topic}

Rules:
- Questions MUST be about {topic} only
- Do NOT include answers
- Number each question

Format:
1. [Question about {topic}]
2. [Question about {topic}]
3. [Question about {topic}]
"""
    response = llm.invoke(prompt)
    return response.content


# -------------------------------
# AGENT TOOLS
# -------------------------------

tools = [
    Tool(
        name="Syllabus QA",
        func=retrieve_syllabus,
        description="""Use this to answer questions about AI topics, concepts, 
        or explain anything from the syllabus. Input should be the student's question."""
    ),

    Tool(
        name="Quiz Generator",
        func=generate_quiz,
        description="""Use this ONLY when the student explicitly asks for quiz questions, 
        practice questions, or says 'ask me questions'. 
        Input must be the EXACT topic to generate questions about.
        """
    )
]


# -------------------------------
# MEMORY
# -------------------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input",
    output_key="output"
)


# -------------------------------
# AGENT
# -------------------------------

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    max_iterations=3,
    early_stopping_method="generate",
    handle_parsing_errors=True
)


# -------------------------------
# CHAT API
# -------------------------------

@app.post("/chat")
def chat(query: Query):
    try:
        user_message = query.message
        result = agent.invoke({"input": user_message})
        output = result["output"]

        clean_lines = [
            line for line in output.split("\n")
            if not any(line.startswith(prefix) for prefix in ["Thought:", "Action:", "Action Input:", "Observation:"])
        ]
        clean_output = "\n".join(clean_lines).strip()

    except Exception as e:
        print(f"ERROR: {e}")          
        return {"response": str(e)}   

    return {"response": clean_output}
    
