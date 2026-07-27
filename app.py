
import os
import sys
import subprocess

import streamlit as st






sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)


from src.agent import preguntar




st.set_page_config(
    page_title="NubeFlow AI",
    page_icon="🤖"
)


st.title("🤖 NubeFlow AI Agent")

st.caption(
    "Agente inteligente basado en RAG "
    "(Gemini Embeddings + Chroma + Groq)"
)


if "messages" not in st.session_state:

    st.session_state.messages = []



for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )


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
                
                if isinstance(respuesta, dict) and "fuentes" in respuesta and respuesta["fuentes"]:
                    with st.expander("Ver fuentes utilizadas"):
                        for fuente in respuesta["fuentes"]:
                            st.markdown(f"📄 `{fuente}`")
                else:
                    st.caption("ℹ️ No se requirieron documentos internos para esta respuesta.")


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": respuesta
        }
    )
    

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from agent import preguntar
if not os.path.exists("chroma_db"):
    with st.spinner("🚀 Procesando tus documentos PDF por primera vez en la nube..."):
        try:
            subprocess.run(["python", "src/ingest.py"], check=True)
        except Exception as e:
            st.error(f"❌ Error al inicializar la base de datos: {e}")
            st.stop()