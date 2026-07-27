# -*- coding: utf-8 -*-

"""
app.py

Interfaz web del agente NubeFlow usando Streamlit.
"""

import os
import sys

import streamlit as st


# Importar archivos de src
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)


from src.agent import preguntar



# Configuración

st.set_page_config(
    page_title="NubeFlow AI",
    page_icon="🤖"
)


st.title("🤖 NubeFlow AI Agent")

st.caption(
    "Agente inteligente basado en RAG "
    "(Gemini Embeddings + Chroma + Groq)"
)


# Historial

if "messages" not in st.session_state:

    st.session_state.messages = []



# Mostrar historial

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )



# Entrada usuario

pregunta = st.chat_input(
    "Pregunta sobre tus documentos..."
)



if pregunta:


    st.session_state.messages.append(
        {
            "role": "user",
            "content": pregunta
        }
    )


    with st.chat_message("user"):

        st.markdown(pregunta)



    with st.chat_message("assistant"):

        with st.spinner(
            "Buscando información..."
        ):

            respuesta = preguntar(
                pregunta
            )


            st.markdown(respuesta)


            with st.expander("📚 Ver fuentes utilizadas"):

                for fuente in respuesta["fuentes"]:
                    st.write(
                            "📄",
                            fuente
                )


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta
        }
    )