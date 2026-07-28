import os
import re
import json
import tempfile
from datetime import datetime, timezone
 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
 
load_dotenv()
 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Debe apuntar exactamente al mismo directorio que ingest.py (ver comentario ahí)
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "nubeflow_chroma_db")
 
# --- Etapa 8: Registro de ejecución ---
# Se guarda en /tmp por el mismo motivo que chroma_db (ver ingest.py): el
# filesystem del propio repositorio en Streamlit Cloud no soporta bien las
# escrituras/locks, mientras que /tmp sí es un disco local real y escribible.
LOG_DIR = os.path.join(tempfile.gettempdir(), "nubeflow_logs")
LOG_FILE = os.path.join(LOG_DIR, "agent_log.jsonl")
 
EMBEDDING_MODEL = "models/gemini-embedding-001"
GROQ_MODEL = "llama-3.3-70b-versatile"
 
TOP_K = 8                  # candidatos iniciales de la búsqueda vectorial (antes: 4)
TOP_N = 4                  # fragmentos finales que se envían al LLM tras filtrar
UMBRAL_RELEVANCIA = 0.5    # score mínimo (0-1) para considerar un fragmento relevante
 
CATEGORIA_TODAS = "Todas las categorías"
 
# Palabras muy comunes en español que no aportan al reranking por palabras clave
PALABRAS_VACIAS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "y", "o", "que", "en", "es", "son", "por", "para", "con", "sin", "sobre",
    "cual", "cuales", "cuanto", "cuanta", "cuantos", "cuantas", "como", "donde",
    "quien", "quienes", "mi", "tu", "su", "sus", "se", "lo", "le", "les",
    "a", "e", "u", "pero", "si", "no", "que", "cual", "qué", "cuál", "cómo",
    "dónde", "quién",
}
 
 
def cargar_vectorstore():
    google_key = os.getenv("GOOGLE_API_KEY")
    if not google_key:
        raise Exception("Falta GOOGLE_API_KEY en el entorno")
 
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=google_key)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
 
 
def crear_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise Exception("Falta GROQ_API_KEY en el entorno")
    return ChatGroq(model=GROQ_MODEL, temperature=0.2, api_key=groq_key)
 
 
def _registrar_log(pregunta: str, respuesta: str, fuentes: list, encontro_contexto: bool, scores: list):
    """
    Etapa 8 del desafío: registro de ejecución.
    Guarda cada interacción (pregunta, fuentes usadas, respuesta, timestamp, scores
    de relevancia) en un archivo JSON Lines, para poder auditar y depurar el
    comportamiento del agente en producción.
 
    El logging nunca debe romper la respuesta del agente: si falla, solo se
    imprime en consola y la conversación sigue normalmente.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entrada = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pregunta": pregunta,
            "fuentes_utilizadas": fuentes,
            "encontro_contexto_relevante": encontro_contexto,
            "scores_relevancia": scores,
            "respuesta": respuesta,
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"No se pudo escribir el log de ejecución: {e}")
 
 
def obtener_categorias_disponibles() -> list:
    """
    Devuelve las categorías de documentos realmente indexadas (leídas de los
    metadatos en Chroma), para poder ofrecer el filtro en la interfaz sin tener
    que hardcodear la lista.
    """
    try:
        vectorstore = cargar_vectorstore()
        data = vectorstore.get()
        categorias = {m.get("categoria", "General") for m in data.get("metadatas", []) if m}
        return sorted(categorias)
    except Exception as e:
        print(f"No se pudieron leer las categorías disponibles: {e}")
        return []
 
 
def _rerank_por_palabras_clave(pregunta: str, candidatos: list) -> list:
    """
    Reranking simple y liviano (sin modelos adicionales): además del score
    semántico de embeddings, le da un pequeño impulso a los fragmentos que
    comparten más palabras clave literales con la pregunta. Esto ayuda a
    desempatar casos donde la similitud semántica es pareja, pero un
    fragmento en particular menciona los términos exactos de la pregunta.
 
    No reemplaza un cross-encoder "de verdad", pero mejora el orden sin
    agregar dependencias ni latencia significativa.
    """
    palabras_pregunta = set(re.findall(r"\w+", pregunta.lower())) - PALABRAS_VACIAS
 
    con_boost = []
    for doc, score in candidatos:
        palabras_doc = set(re.findall(r"\w+", doc.page_content.lower()))
        coincidencias = len(palabras_pregunta & palabras_doc)
        boost = min(coincidencias * 0.02, 0.1)
        con_boost.append((doc, score, score + boost))
 
    con_boost.sort(key=lambda x: x[2], reverse=True)
    return [(doc, score_original) for doc, score_original, _ in con_boost]
 
 
def registrar_feedback(pregunta: str, respuesta: str, feedback: str):
    """
    Guarda un feedback (positivo/negativo) del usuario sobre una respuesta ya
    dada, como una entrada adicional en el mismo archivo de log.
    """
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        entrada = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tipo": "feedback",
            "pregunta": pregunta,
            "respuesta": respuesta,
            "feedback": feedback,  
        }
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"No se pudo registrar el feedback: {e}")
 
 
def preguntar(pregunta, categoria=None):
    vectorstore = cargar_vectorstore()
    filtro = None
    if categoria and categoria != CATEGORIA_TODAS:
        filtro = {"categoria": {"$eq": categoria}}
    resultados_con_score = vectorstore.similarity_search_with_relevance_scores(
        pregunta, k=TOP_K, filter=filtro
    )
    resultados_con_score = _rerank_por_palabras_clave(pregunta, resultados_con_score)
 
    relevantes = [(doc, score) for doc, score in resultados_con_score if score >= UMBRAL_RELEVANCIA]
    relevantes.sort(key=lambda x: x[1], reverse=True)
    relevantes = relevantes[:TOP_N]
 
    scores_para_log = [round(float(score), 3) for _, score in resultados_con_score]
 
    if not relevantes:
  
        respuesta_fallback = "No encontré esta información en los documentos disponibles."
        _registrar_log(
            pregunta, respuesta_fallback, [],
            encontro_contexto=False, scores=scores_para_log,
        )
        return {"respuesta": respuesta_fallback, "fuentes": []}
 
    llm = crear_llm()
 
    contexto = ""
    for doc, _ in relevantes:
        fuente = doc.metadata.get("fuente") or doc.metadata.get("source", "desconocido")
        contexto += f"\nFuente: {fuente}\n\nContenido:\n{doc.page_content}\n-------------------\n"
 
    prompt = ChatPromptTemplate.from_template(
        """
Eres un asistente corporativo inteligente del ecosistema NubeFlow.
 
Responde a la consulta utilizando únicamente la información proporcionada en el contexto.
NOTA IMPORTANTE: Si el usuario pregunta "¿Qué es NubeFlow?", "¿De qué trata la empresa?" o
pide un resumen general, asume que "NubeFlow" es la plataforma, empresa o producto del cual
hablan los documentos del contexto y elabora un resumen claro con la información disponible.
 
Si la información para responder a la pregunta no aparece en absoluto en el contexto, indica
amablemente que no tienes información suficiente en los documentos internos.
 
Contexto disponible:
{contexto}
 
Pregunta del usuario:
{pregunta}
 
Respuesta:
"""
    )
 
    mensaje = prompt.format(contexto=contexto, pregunta=pregunta)
    respuesta = llm.invoke(mensaje)
 
    fuentes_usadas = set()
    for doc, _ in relevantes:
        ruta = doc.metadata.get("fuente") or doc.metadata.get("source", "Documento sin nombre")
        fuentes_usadas.add(os.path.basename(ruta))
 
    resultado_final = {
        "respuesta": respuesta.content,
        "fuentes": list(fuentes_usadas),
    }
 
    _registrar_log(
        pregunta, resultado_final["respuesta"], resultado_final["fuentes"],
        encontro_contexto=True, scores=scores_para_log,
    )
 
    return resultado_final