from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

VECTOR_DB_PATH = 'db_faiss'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
LLM_MODEL = 'deepseek-ai/deepseek-llm-7b-chat'

app = FastAPI()

# Permitir requisições externas (ex: frontend em Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # Carregar embeddings
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)

    # Carregar vetor de documentos
    db = FAISS.load_local(VECTOR_DB_PATH, embeddings)
    retriever = db.as_retriever()

    # Inicializar modelo LLM
    pipe = pipeline(
        model=LLM_MODEL,
        task='text-generation',
        model_kwargs={'temperature': 0.7, 'max_new_tokens': 512}
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # Construir QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
except Exception as e:
    print(f"[LIA] Falha ao inicializar: {e}")
    qa_chain = None

@app.get('/')
def read_root():
    return {'message': 'Lia está rodando localmente no terminal.'}

@app.post('/chat')
async def chat(request: Request):
    if not qa_chain:
        return {'error': 'Sistema indisponível no momento.'}
    
    try:
        data = await request.json()
        question = data.get('question', '').strip()

        if not question:
            return {'error': 'Pergunta inválida.'}

        response = qa_chain.run(question)
        print(f"[LIA][Q]: {question}")
        print(f"[LIA][A]: {response}")
        return {'resposta': response}
    
    except Exception as e:
        print(f"[LIA] Erro no endpoint /chat: {e}")
        return {'error': 'Erro interno no processamento da pergunta.'}
