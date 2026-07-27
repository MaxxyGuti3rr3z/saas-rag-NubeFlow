import os
 
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
 
load_dotenv()
 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
 
EMBEDDING_MODEL = "models/gemini-embedding-001"
GROQ_MODEL = "llama-3.3-70b-versatile"
TOP_K = 4
 
 
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
 
 
def preguntar(pregunta):
    vectorstore = cargar_vectorstore()
    llm = crear_llm()
 
    resultados = vectorstore.similarity_search(pregunta, k=TOP_K)
 
    if not resultados:

        return {
            "respuesta": "No encontré esta información en los documentos disponibles.",
            "fuentes": [],
        }
 
    contexto = ""
    for doc in resultados:
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
    for doc in resultados:
        ruta = doc.metadata.get("fuente") or doc.metadata.get("source", "Documento sin nombre")
        fuentes_usadas.add(os.path.basename(ruta))
 
    return {
        "respuesta": respuesta.content,
        "fuentes": list(fuentes_usadas),
    }