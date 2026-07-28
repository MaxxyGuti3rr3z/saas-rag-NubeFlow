import os
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

CHROMA_DIR = os.path.join(tempfile.gettempdir(), "nubeflow_chroma_db")
 

LOG_DIR = os.path.join(tempfile.gettempdir(), "nubeflow_logs")
LOG_FILE = os.path.join(LOG_DIR, "agent_log.jsonl")
 
EMBEDDING_MODEL = "models/gemini-embedding-001"
GROQ_MODEL = "llama-3.3-70b-versatile"
 
TOP_K = 8                 
TOP_N = 4                  
UMBRAL_RELEVANCIA = 0.5    
 
 
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
 
 
def preguntar(pregunta):
    vectorstore = cargar_vectorstore()

    resultados_con_score = vectorstore.similarity_search_with_relevance_scores(pregunta, k=TOP_K)
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