from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict
import requests, os, tempfile, json, hashlib
import urllib.parse
import pdfplumber, docx, email, extract_msg
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI
# from dotenv import load_dotenv

# -------------------- FastAPI Setup --------------------
app = FastAPI()
API_KEY = "37fdc7e9f68374473f706d1a2bc85ad26e59b5f77cae8ebb3226d7dacf138598"
# load_dotenv()
OPENAI_API_KEY = "OPENAI_API_KEY"

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

# -------------------- Globals --------------------
embedding_model = None
llm = None
text_splitter = None
document_cache: Dict[str, str] = {}
vector_store_cache: Dict[str, FAISS] = {}

# -------------------- Model Loading --------------------
def startup_models():
    global embedding_model, llm, text_splitter
    print("\U0001F680 Loading models...")
    embedding_model = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
    llm = AzureChatOpenAI(
        azure_endpoint="https://ver0nica.cognitiveservices.azure.com/",
        api_key=OPENAI_API_KEY,
        azure_deployment="gpt-4o",
        api_version="2024-12-01-preview",
        temperature=0,
        max_tokens=120  # increased token limit to reduce truncation
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    print("✅ Models loaded.")

startup_models()

# -------------------- Helpers --------------------
def get_url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def extract_text_from_pdf(file_path):
    with pdfplumber.open(file_path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def load_docx_text(file_path):
    doc = docx.Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def load_eml_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        msg = email.message_from_file(f)
    if msg.is_multipart():
        return "".join(part.get_payload(decode=True).decode('utf-8', errors='ignore')
                       for part in msg.walk() if part.get_content_type() == "text/plain")
    return msg.get_payload(decode=True).decode('utf-8', errors='ignore')

def load_msg_text(file_path):
    msg = extract_msg.Message(file_path)
    return msg.body

def load_document(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return load_docx_text(file_path)
    elif ext == ".eml":
        return load_eml_text(file_path)
    elif ext == ".msg":
        return load_msg_text(file_path)
    else:
        raise ValueError("Unsupported file type")

def download_and_process_document(doc_url: str) -> str:
    url_hash = get_url_hash(doc_url)
    if url_hash in document_cache:
        print("📋 Using cached document")
        return document_cache[url_hash]

    print("📅 Downloading document...")
    response = requests.get(doc_url, timeout=60)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch document")

    parsed_url = urllib.parse.urlparse(doc_url)
    filename = os.path.basename(parsed_url.path)
    _, ext = os.path.splitext(filename)
    if not ext:
        ext = '.pdf'

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
        tmp_file.write(response.content)
        filepath = tmp_file.name

    try:
        text = load_document(filepath)
        document_cache[url_hash] = text
        return text
    finally:
        if os.path.exists(filepath):
            os.unlink(filepath)

def build_faiss_index(text: str, url_hash: str) -> FAISS:
    if url_hash in vector_store_cache:
        print("🔍 Using cached FAISS index")
        return vector_store_cache[url_hash]
    chunks = [c.strip() for c in text_splitter.split_text(text) if c.strip()]
    if not chunks:
        raise ValueError("Document split failed")
    index = FAISS.from_texts(chunks, embedding=embedding_model)
    vector_store_cache[url_hash] = index
    return index

# -------------------- Prompt Template --------------------
custom_prompt = PromptTemplate.from_template("""
You are a smart assistant that answers questions using insurance, legal, or HR documents.
Use the context below to answer the user's questions. Your answers should be accurate and concise in a single line or two.
Provide answers for each question as per the maximum character limit.
Output must follow this exact JSON format:

{{
  "answers": [
    "Answer to question 1",
    "Answer to question 2",
    ...
  ]
}}

Note: Only return the JSON format above. Do not include explanations or extra text.

Context:
{context}

Questions:
{question}
""")

# -------------------- Main Endpoint --------------------
@app.post("/api/v1/hackrx/run", response_model=AnswerResponse)
def run_rag(payload: QueryRequest, authorization: str = Header(None)):
    if not authorization or authorization.split("Bearer ")[-1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    
    
    print("📄 Incoming RAG Request")
    print(f"Document URL:\n{payload.documents}\n")
    print("❓ Questions:")
    for idx, q in enumerate(payload.questions, start=1):
        print(f"  {idx}. {q}")
    print("=" * 60 + "\n")

    try:
        text = download_and_process_document(payload.documents)
        url_hash = get_url_hash(payload.documents)
        index = build_faiss_index(text, url_hash)
        retriever = index.as_retriever(search_kwargs={"k": 10})

        # ✅ DO NOT use prompt template — use raw output like in IPYNB
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=False,
        )

        answers = []
        for question in payload.questions:
            result = qa_chain.invoke({"query": question})
            answers.append(result["result"])  # ✅ Same as Code 1 behavior

        return AnswerResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


# -------------------- Utility Endpoints --------------------
@app.get("/")
def root():
    return {
        "status": "API running",
        "cache": {
            "documents": len(document_cache),
            "indexes": len(vector_store_cache)
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/api/v1/cache/clear")
def clear_cache(authorization: str = Header(None)):
    if not authorization or authorization.split("Bearer ")[-1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    document_cache.clear()
    vector_store_cache.clear()
    return {"message": "Cache cleared successfully"}

