import os
import sys
import streamlit as st
 
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

for key in ("GOOGLE_API_KEY", "GROQ_API_KEY"):
    if key in st.secrets and not os.getenv(key):
        os.environ[key] = st.secrets[key]
 
from agent import preguntar
import agent as agent_mod
import ingest  
 
st.set_page_config(page_title="NubeFlow AI", page_icon="🤖")
st.title("🤖 NubeFlow AI Agent")
st.caption("Agente inteligente basado en RAG (Gemini Embeddings + Chroma + Groq)")
st.info(
    "🤖 Estás conversando con un **agente de inteligencia artificial**, no con una persona. "
    "Las respuestas se generan a partir de los documentos internos indexados.",
    icon="ℹ️",
)
 
 
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
    try:
        log_file = getattr(agent_mod, "LOG_FILE", None)
        if log_file and os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
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
    except Exception as e:
        st.caption(f"⚠️ No se pudo cargar el panel de registro ({e}).")
 
    st.subheader("🔍 Buscar en una categoría")
    try:
        categorias_disponibles = [agent_mod.CATEGORIA_TODAS] + agent_mod.obtener_categorias_disponibles()
    except Exception:
        categorias_disponibles = [agent_mod.CATEGORIA_TODAS]
    categoria_seleccionada = st.selectbox(
        "Restringir búsqueda a:",
        categorias_disponibles,
        key="categoria_seleccionada",
    )
 
 
def _mostrar_feedback(idx: int, message: dict):
    """Muestra botones 👍/👎 bajo una respuesta del agente, o el feedback ya dado."""
    feedback_previo = message.get("feedback")
    if feedback_previo:
        emoji = "👍" if feedback_previo == "positivo" else "👎"
        st.caption(f"Feedback registrado: {emoji} ¡Gracias!")
        return
 
    col1, col2, _ = st.columns([1, 1, 8])
    with col1:
        if st.button("👍", key=f"fb_up_{idx}"):
            agent_mod.registrar_feedback(
                message.get("pregunta_relacionada", ""), message["content"], "positivo"
            )
            st.session_state.messages[idx]["feedback"] = "positivo"
            st.rerun()
    with col2:
        if st.button("👎", key=f"fb_down_{idx}"):
            agent_mod.registrar_feedback(
                message.get("pregunta_relacionada", ""), message["content"], "negativo"
            )
            st.session_state.messages[idx]["feedback"] = "negativo"
            st.rerun()
 
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("fuentes"):
            with st.expander("📚 Ver fuentes utilizadas"):
                for f in message["fuentes"]:
                    st.markdown(f"📄 `{f}`")
        if message["role"] == "assistant":
            _mostrar_feedback(i, message)
 
pregunta = st.chat_input("Pregunta sobre tus documentos...")
 
if pregunta:
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
 
    with st.chat_message("assistant"):
        with st.spinner("Buscando información..."):
            resultado = preguntar(pregunta, categoria=categoria_seleccionada)
 
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
        "pregunta_relacionada": pregunta,
        "feedback": None,
    })
    st.rerun()