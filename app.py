import os
import sys
import streamlit as st
 
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
 

for key in ("GOOGLE_API_KEY", "GROQ_API_KEY"):
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = st.secrets[key]
 
from agent import preguntar
import ingest
 
st.set_page_config(page_title="NubeFlow AI", page_icon="🤖")
st.title("🤖 NubeFlow AI Agent")
st.caption("Agente inteligente basado en RAG (Gemini Embeddings + Chroma + Groq)")
 
 
def construir_base_de_datos():
    with st.spinner("🚀 Procesando documentos PDF e indexando en Chroma..."):
        try:
            ingest.main()
        except Exception as e:
            st.error(f"❌ Error al construir la base de datos: {e}")
            st.stop()
 
if not ingest.base_de_datos_valida():
    construir_base_de_datos()

with st.sidebar:
    st.subheader("⚙️ Estado de la base de conocimiento")
    try:
        vectorstore = ingest.Chroma(
            persist_directory=ingest.CHROMA_DIR,
            embedding_function=ingest.obtener_embeddings(),
        )
        total_docs = vectorstore._collection.count()
        st.metric("Fragmentos indexados", total_docs)
        if total_docs == 0:
            st.warning("La base está vacía. Usa el botón de abajo para reconstruirla.")
    except Exception as e:
        st.error(f"No se pudo leer la base vectorial: {e}")
 
    if st.button("🔄 Reconstruir base de conocimiento"):
        construir_base_de_datos()
        st.rerun()

    st.subheader("🧾 Registro de ejecución")
    import agent as _agent_mod 
    if os.path.exists(_agent_mod.LOG_FILE):
        with open(_agent_mod.LOG_FILE, "r", encoding="utf-8") as f:
            contenido_log = f.read()
        total_lineas = contenido_log.count("\n")
        st.caption(f"{total_lineas} interacción(es) registrada(s) en esta sesión.")
        st.download_button(
            "📥 Descargar log (JSON Lines)",
            data=contenido_log,
            file_name="nubeflow_agent_log.jsonl",
            mime="application/jsonl",
        )
    else:
        st.caption("Todavía no hay interacciones registradas.")
 
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("fuentes"):
            with st.expander("📚 Ver fuentes utilizadas"):
                for f in message["fuentes"]:
                    st.markdown(f"📄 `{f}`")
 
pregunta = st.chat_input("Pregunta sobre tus documentos...")
 
if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
 
    with st.chat_message("assistant"):
        with st.spinner("Buscando información..."):
            resultado = preguntar(pregunta)
 
        texto_respuesta = resultado["respuesta"]
        fuentes = resultado.get("fuentes", [])
 
        st.markdown(texto_respuesta)
 
        if fuentes:
            with st.expander("📚 Ver fuentes utilizadas"):
                for fuente in fuentes:
                    st.markdown(f"📄 `{fuente}`")
        else:
            st.caption("ℹ️ No se encontraron documentos internos relevantes para esta respuesta.")
 
    st.session_state.messages.append({
        "role": "assistant",
        "content": texto_respuesta,
        "fuentes": fuentes,
    })
 