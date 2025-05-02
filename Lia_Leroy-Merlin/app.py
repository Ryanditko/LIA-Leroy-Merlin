import streamlit as st
from chat import ask_lia

st.set_page_config('LIA - Assistente Leroy Merlin', page_icon=':robot_face:', layout='wide')
st.title('LIA - Assistente Leroy Merlin')

# Inicializa histórico
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Input do usuário
user_input = st.text_input('Digite sua pergunta:', placeholder='Digite sua pergunta para a LIA...')

# Exibe o histórico
for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Processa nova pergunta
if user_input:
    st.session_state.chat_history.append({'role': 'user', 'content': user_input})
    
    with st.chat_message('user'):
        st.markdown(user_input)

    with st.chat_message('assistant'):
        response = ask_lia(user_input)
        st.markdown(response)

    st.session_state.chat_history.append({'role': 'assistant', 'content': response})
