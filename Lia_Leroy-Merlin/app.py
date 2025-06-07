import streamlit as st
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import time
import json
from knowledge_manager import KnowledgeManager
from datetime import datetime

# Carrega variáveis de ambiente
load_dotenv()

# Configuração da página
st.set_page_config(
    page_title="LIA - Assistente Virtual Leroy Merlin",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
<style>
    .main {
        padding: 2rem;
        background-color: #f5f5f5;
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .chat-message.user {
        background-color: #ffffff;
        border-left: 5px solid #FF4B4B;
    }
    .chat-message.assistant {
        background-color: #ffffff;
        border-left: 5px solid #4B7BFF;
    }
    .chat-message .content {
        display: flex;
        flex-direction: row;
        align-items: flex-start;
    }
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 1rem;
        background-color: #f0f0f0;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
    }
    .chat-message .message {
        flex: 1;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        transform: translateY(-2px);
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    .feedback-container {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .feedback-button {
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .feedback-button:hover {
        transform: scale(1.05);
    }
    .feedback-positive {
        background-color: #4CAF50;
        color: white;
    }
    .feedback-negative {
        background-color: #f44336;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.leroymerlin.com.br/static/img/logo-leroy-merlin.svg", width=200)
    st.title("LIA - Assistente Virtual")
    st.markdown("""
    ### Sobre a LIA
    A LIA é sua assistente virtual especializada em:
    - Produtos de construção e decoração
    - Dicas de reforma e manutenção
    - Orçamentos e planejamento
    - Suporte técnico especializado
    
    ### Como usar
    1. Digite sua pergunta na caixa de texto
    2. A LIA irá responder com base no conhecimento da Leroy Merlin
    3. Você pode fazer perguntas de follow-up para mais detalhes
    4. Ajude a LIA a melhorar dando feedback nas respostas
    """)

# Título principal
st.title("🏠 Chat com a LIA")

# Inicializa o gerenciador de conhecimento
@st.cache_resource
def load_knowledge_manager():
    return KnowledgeManager()

# Inicializa o modelo e tokenizer
@st.cache_resource
def load_model():
    model_name = "gpt2"  # Modelo mais estável
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    return model, tokenizer

# Carregar modelo DeepSeek
@st.cache_resource(show_spinner=True)
def load_deepseek_model():
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_deepseek_model()

# Função para gerar resposta com DeepSeek
def gerar_resposta_deepseek(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta[len(prompt):].strip()

# Inicializa histórico
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Carrega os recursos
try:
    knowledge_manager = load_knowledge_manager()
    model, tokenizer = load_model()
except Exception as e:
    st.error(f"Erro ao carregar os recursos: {str(e)}")
    st.stop()

# Exibe estatísticas de aprendizado
with st.expander("📊 Estatísticas de Aprendizado"):
    stats = knowledge_manager.get_learning_stats()
    st.write(f"Total de interações: {stats['total_interacoes']}")
    st.write(f"Novo conhecimento adicionado: {stats['novo_conhecimento']}")
    st.write(f"Tamanho da base vetorial: {stats['vector_store_size']}")

# Exibe o histórico
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        # Adiciona botões de feedback para mensagens da LIA
        if message['role'] == 'assistant' and 'feedback' not in message:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("��", key=f"positive_{message['index']}"):
                    knowledge_manager.add_interaction(
                        st.session_state.chat_history[-2]['content'],
                        message['content'],
                        feedback=1.0
                    )
                    message['feedback'] = 1.0
                    st.success("Obrigado pelo feedback positivo!")
            with col2:
                if st.button("��", key=f"negative_{message['index']}"):
                    knowledge_manager.add_interaction(
                        st.session_state.chat_history[-2]['content'],
                        message['content'],
                        feedback=0.0
                    )
                    message['feedback'] = 0.0
                    st.error("Obrigado pelo feedback. Vou melhorar!")

# Input do usuário
user_input = st.text_input(
    'Digite sua pergunta:',
    placeholder='Ex: Quais são os melhores produtos para reforma de banheiro?',
    key='user_input'
)

# Opções de geração
col1, col2 = st.columns(2)
with col1:
    max_length = st.slider('Tamanho da resposta', 50, 500, 200)
with col2:
    temperature = st.slider('Criatividade', 0.1, 1.0, 0.7)

# Processa nova mensagem
if user_input:
    # Adiciona mensagem do usuário ao histórico
    message_index = len(st.session_state.chat_history)
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input,
        'index': message_index
    })
    
    # Exibe mensagem do usuário
    with st.chat_message('user'):
        st.markdown(user_input)

    # Exibe resposta da LIA
    with st.chat_message('assistant'):
        with st.spinner('LIA está pensando...'):
            try:
                # Busca conhecimento relevante
                relevant_knowledge = knowledge_manager.get_relevant_knowledge(user_input)
                
                # Prepara o contexto da conversa
                conversation_context = "\n".join([f"{'Usuário' if msg['role'] == 'user' else 'LIA'}: {msg['content']}" 
                                               for msg in st.session_state.chat_history[-3:]])
                
                # Prepara o prompt
                knowledge_context = json.dumps(relevant_knowledge, ensure_ascii=False)
                prompt = f"""Você é a LIA, uma assistente virtual da Leroy Merlin.
Responda de forma clara, amigável e profissional.
Sempre que possível, priorize informações sobre produtos, serviços, dicas e soluções da Leroy Merlin.
Se a pergunta não for sobre Leroy Merlin, responda normalmente como um assistente inteligente.

Base de conhecimento Leroy Merlin:
{knowledge_context}

Histórico da conversa:
{conversation_context}

Pergunta do usuário:
{user_input}

LIA:"""
                
                # Gera a resposta
                resposta = gerar_resposta_deepseek(prompt)
                
                # Adiciona um pequeno delay para simular "digitação"
                time.sleep(1)
                
                # Exibe a resposta
                st.markdown(resposta)
                message_index = len(st.session_state.chat_history)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': resposta,
                    'index': message_index
                })
                
                # Registra a interação
                knowledge_manager.add_interaction(user_input, resposta)
                
            except Exception as e:
                error_message = f"Desculpe, ocorreu um erro ao gerar a resposta: {str(e)}"
                st.error(error_message)
                message_index = len(st.session_state.chat_history)
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': error_message,
                    'index': message_index
                })

# Botão para limpar histórico
if st.button("Limpar Conversa"):
    st.session_state.chat_history = []
    st.rerun()

# Rodapé
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>LIA - Assistente Virtual Leroy Merlin | Desenvolvido com ❤️</p>
</div>
""", unsafe_allow_html=True)
    