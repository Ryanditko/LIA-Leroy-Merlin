import os
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Caminhos e parâmetros
KNOWLEDGE_PATH = 'knowledge'
VECTOR_DB_PATH = 'db_faiss'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

def load_documents():
    docs = []
    logger.info(f"Procurando documentos em: {KNOWLEDGE_PATH}")
    
    if not os.path.exists(KNOWLEDGE_PATH):
        logger.error(f"Diretório {KNOWLEDGE_PATH} não encontrado!")
        return docs

    for root, _, files in os.walk(KNOWLEDGE_PATH): 
        for file in files:
            path = os.path.join(root, file)
            try:
                logger.info(f"Processando arquivo: {path}")
                
                if file.endswith('.pdf'):
                    loader = PyPDFLoader(path)
                    logger.info(f"Carregando PDF: {path}")
                elif file.endswith('.txt'): 
                    loader = TextLoader(path, encoding='utf-8')
                    logger.info(f"Carregando TXT: {path}")
                elif file.endswith('.docx'):
                    loader = Docx2txtLoader(path)
                    logger.info(f"Carregando DOCX: {path}")
                else:
                    logger.warning(f"Ignorando arquivo não suportado: {file}")
                    continue
                
                loaded_docs = loader.load()
                logger.info(f"Documento carregado com sucesso: {len(loaded_docs)} páginas/segmentos")
                docs.extend(loaded_docs)
                
            except Exception as e:
                logger.error(f"Erro ao carregar {file}: {str(e)}")
    
    logger.info(f"Total de documentos carregados: {len(docs)}")
    return docs 

def ingest(save_path=VECTOR_DB_PATH):
    logger.info('Iniciando processo de ingestão...')

    # Carregar documentos
    documents = load_documents()
    if not documents:
        logger.error("Nenhum documento encontrado para processar.")
        return None

    # Fragmentar documentos
    logger.info('Fragmentando documentos em chunks...')
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Total de chunks gerados: {len(chunks)}")

    # Gerar embeddings
    logger.info(f'Gerando embeddings com modelo: {EMBEDDING_MODEL}')
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL)

    # Construir base vetorial
    logger.info('Construindo base vetorial com FAISS...')
    db = FAISS.from_documents(chunks, embeddings)

    # Salvar base vetorial
    logger.info(f'Salvando base vetorial em: {save_path}')
    os.makedirs(save_path, exist_ok=True)
    db.save_local(save_path)

    logger.info('Ingestão concluída com sucesso!')
    return db

# Teste simples para verificar se o PDF pode ser carregado
pdf_path = 'knowledge/Modelo_teste_LIA.pdf' 
try:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[TESTE] Total de documentos carregados do PDF: {len(documents)}")
except Exception as e:
    print(f"[TESTE] Erro ao carregar o PDF: {e}")
    
if __name__ == '__main__':
    try:
        db = ingest()
        if db:
            logger.info("Processo de ingestão finalizado com sucesso!")
        else:
            logger.error("Falha no processo de ingestão.")
    except Exception as e:
        logger.error(f"Erro durante a ingestão: {str(e)}")
