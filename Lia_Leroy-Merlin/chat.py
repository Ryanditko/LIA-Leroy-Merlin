""" | Arquivo     | Função                                        |
| ----------- | --------------------------------------------- |
| `main.py`   | Interface de linha de comando (modo terminal) |
| `ingest.py` | Alimenta a LIA com documentos                 |
| `app.py`    | Interface visual em navegador (Streamlit)     |
| `chat.py`   | Processa perguntas e gera respostas           |
"""

import logging
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import torch

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurações do modelo
MODEL_ID = 'deepseek-ai/deepseek-llm-7b-chat'
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
VECTOR_DB_PATH = 'db_faiss'

# Carregando modelo DeepSeek via HuggingFace
logger.info(f"Carregando modelo: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Wrapping com LangChain 
transformers_pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    top_p=0.95,
    repetition_penalty=1.15
)
llm = HuggingFacePipeline(pipeline=transformers_pipeline)

# Carregando embeddings e base vetorial
logger.info(f"Carregando embeddings: {EMBEDDING_MODEL}")
embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)

try:
    logger.info(f"Carregando base vetorial de: {VECTOR_DB_PATH}")
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings)
except Exception as e:
    logger.error(f"Erro ao carregar base vetorial: {str(e)}")
    vectorstore = None

# Prompt base customizado
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template='''Você é a LIA, uma IA assistente da Leroy Merlin. Sua missão é ajudar colaboradores e clientes com respostas diretas e bem informadas, sempre com base no conteúdo da empresa.

Use a seguinte base de conhecimento como contexto:

{context}

Agora, responda a esta pergunta de forma clara e objetiva:

{question}

Lembre-se:
1. Seja sempre educado e profissional
2. Forneça informações precisas e relevantes
3. Se não souber a resposta, seja honesto e sugira contatar o atendimento ao cliente
4. Mantenha o foco no contexto da Leroy Merlin
'''
)

# Construção da cadeia de QA
if vectorstore:
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True,
    )
else:
    qa_chain = None
    logger.error("Cadeia QA não inicializada devido a erro na base vetorial")

def ask_lia(question):
    if not qa_chain:
        return "Desculpe, estou com dificuldades técnicas no momento. Por favor, tente novamente mais tarde."
    
    try:
        logger.info(f"Processando pergunta: {question}")
        response = qa_chain({"query": question})
        
        # Extrair resposta e documentos fonte
        answer = response["result"].strip()
        source_docs = response.get("source_documents", [])
        
        # Adicionar referências se disponíveis
        if source_docs:
            answer += "\n\nFontes consultadas:"
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get("source", "Documento desconhecido")
                answer += f"\n{i}. {source}"
        
        logger.info("Resposta gerada com sucesso")
        return answer
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {str(e)}")
        return "Desculpe, ocorreu um erro ao processar sua pergunta. Por favor, tente novamente."

if __name__ == "__main__":
    try:
        logger.info("Iniciando teste do modelo...")
        resposta = ask_lia("O que a Leroy Merlin oferece em termos de serviços?")
        logger.info(f"Resposta da LIA: {resposta}")
    except Exception as e:
        logger.error(f"Erro durante o teste: {str(e)}")
