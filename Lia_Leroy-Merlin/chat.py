""" | Arquivo     | Função                                        |
| ----------- | --------------------------------------------- |
| `main.py`   | Interface de linha de comando (modo terminal) |
| `ingest.py` | Alimenta a LIA com documentos                 |
| `app.py`    | Interface visual em navegador (Streamlit)     |
| `chat.py`   | Processa perguntas e gera respostas           |
"""

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline

# Inicializa o modelo da DeepSeek via HuggingFace Hub
deepseek_pipeline = pipeline(
    model='deepseek-ai/deepseek-llm-7b-chat',
    task='text-generation',
    model_kwargs={'temperature': 0.7, 'max_new_tokens': 512},
)

# Carregando embeddings e base vetorial
embeddings = HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectorstore = FAISS.load_local('faiss_index', embeddings)

# Prompt base customizado
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template='''Você é a LIA, uma IA assistente da Leroy Merlin. Sua missão é ajudar colaboradores e clientes com respostas diretas e bem informadas, sempre com base no conteúdo da empresa.

Use a seguinte base de conhecimento como contexto:

{context}

Agora, responda a esta pergunta de forma clara e objetiva:

{question}
''')

# Construção da cadeia de QA
qa_chain = RetrievalQA.from_chain_type(
    llm=deepseek_pipeline,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=False,
)

# Função de interface
def ask_lia(question):
    try:
        response = qa_chain.run(question)
        return response.strip()
    except Exception as e:
        return f"Ocorreu um erro ao processar a pergunta: {str(e)}"

