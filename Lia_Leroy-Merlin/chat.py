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
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline


# Carregando modelo DeepSeek via HuggingFace
model_id = 'deepseek-ai/deepseek-llm-7b-chat'
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# Wrapping com LangChain
transformers_pipeline = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
)
llm = HuggingFacePipeline(pipeline=transformers_pipeline)

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
    llm=llm,
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
        
    # Função para rodar a LIA em fase de teste
""" def ask_lia(question):
    # Conjunto de dados de teste
    test_data = {
        "O que a Leroy Merlin oferece em termos de serviços?": "A Leroy Merlin oferece uma ampla gama de produtos e serviços, incluindo materiais de construção, decoração, jardinagem, entrega e instalação.",
        "Qual é a política de devolução da Leroy Merlin?": "A Leroy Merlin aceita devoluções em até 30 dias após a compra, desde que o produto esteja em sua embalagem original.",
        # Adicione mais perguntas e respostas conforme necessário
    }
    
    # Verifica se a pergunta está no conjunto de dados de teste
    if question in test_data:
        return test_data[question]
    
    # Se não estiver, continue com a lógica normal
    try:
        response = qa_chain.run(question)
        return response.strip()
    except Exception as e:
        return f"Ocorreu um erro ao processar a pergunta: {str(e)}"""

# Teste do modelo (opcional)
if __name__ == "__main__":
    try:
        print("Iniciando o carregamento do modelo...")
        resposta = ask_lia("O que a Leroy Merlin oferece em termos de serviços?")
        print("Resposta da LIA:", resposta)
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
