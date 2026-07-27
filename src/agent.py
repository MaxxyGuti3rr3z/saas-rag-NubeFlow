import os

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

CHROMA_DIR = "chroma_db"

EMBEDDING_MODEL = "models/gemini-embedding-001"

GROQ_MODEL = "llama-3.3-70b-versatile"

TOP_K = 4


def cargar_vectorstore():

    google_key = os.getenv(
        "GOOGLE_API_KEY"
    )


    if not google_key:

        raise Exception(
            "Falta GOOGLE_API_KEY en .env"
        )


    embeddings = GoogleGenerativeAIEmbeddings(

        model=EMBEDDING_MODEL,

        google_api_key=google_key

    )


    vectorstore = Chroma(

        persist_directory=CHROMA_DIR,

        embedding_function=embeddings

    )


    return vectorstore


def crear_llm():

    groq_key = os.getenv(
        "GROQ_API_KEY"
    )


    if not groq_key:

        raise Exception(
            "Falta GROQ_API_KEY en .env"
        )


    llm = ChatGroq(

        model=GROQ_MODEL,

        temperature=0.2,

        api_key=groq_key

    )


    return llm



def preguntar(pregunta):


    vectorstore = cargar_vectorstore()


    llm = crear_llm()


   

    resultados = vectorstore.similarity_search(

        pregunta,

        k=TOP_K

    )


    if not resultados:

        return "No encontré información relacionada."


    

    contexto = ""


    for doc in resultados:


        fuente = doc.metadata.get(
            "source",
            "desconocido"
        )


        contexto += f"""
Fuente: {fuente}

Contenido:
{doc.page_content}

-------------------
"""




    prompt = ChatPromptTemplate.from_template(
        """
Eres un asistente experto.

Responde utilizando únicamente la información
proporcionada en el contexto.

Si la respuesta no aparece en el contexto,
indica que no tienes información suficiente.

Contexto:

{contexto}


Pregunta:

{pregunta}


Respuesta:
"""
    )


    mensaje = prompt.format(

        contexto=contexto,

        pregunta=pregunta

    )


    respuesta = llm.invoke(
    mensaje
    )


    fuentes = "\n\nFuentes utilizadas:\n"
    fuentes_usadas = set()

    for doc in resultados:
        ruta_completa = doc.metadata.get("source", "Documento sin nombre")
        nombre_limpio = os.path.basename(ruta_completa)
        fuentes_usadas.add(nombre_limpio)
        
    for fuente in fuentes_usadas:
        fuentes += f"📄 {fuente}\n"

    return {
        "respuesta": respuesta.content,
        "fuentes": list(fuentes_usadas)
    }   

