import os
import glob
import shutil
import tempfile
 
from dotenv import load_dotenv
from datetime import datetime
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
 
load_dotenv()
 
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
 
# IMPORTANTE: Chroma usa SQLite por debajo, que necesita locks de archivo reales.
# El directorio del repo en Streamlit Cloud corre sobre un filesystem que NO
# soporta bien esos locks (da "attempt to write a readonly database").
# /tmp sí es un disco local real y escribible, así que guardamos ahí la base.
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "nubeflow_chroma_db")
 
EMBEDDING_MODEL = "models/gemini-embedding-001"
 
 
def cargar_pdfs():
    documentos = []
    archivos = glob.glob(os.path.join(DATA_DIR, "**", "*.pdf"), recursive=True)
 
    if not archivos:
        raise Exception(
            f"No se encontraron archivos PDF en '{DATA_DIR}'. "
            "Verifica que la carpeta data/ con los PDFs esté realmente en el "
            "repositorio de GitHub (no excluida por .gitignore)."
        )
 
    print(f"PDFs encontrados: {len(archivos)}")
 
    for archivo in archivos:
        print(f"Procesando: {archivo}")
        loader = PyPDFLoader(archivo)
        docs = loader.load()
 
        nombre_archivo = os.path.basename(archivo)
 
        categoria = "General"
        if "recurso" in archivo.lower() or "rh" in archivo.lower():
            categoria = "Recursos Humanos"
        elif "finanz" in archivo.lower() or "financ" in archivo.lower():
            categoria = "Financiero"
        elif "operaci" in archivo.lower() or "manual" in archivo.lower():
            categoria = "Operaciones"
 
        for doc in docs:
            doc.metadata.update({
                "fuente": nombre_archivo,
                "categoria": categoria,
                "fecha_procesamiento": datetime.now().strftime("%Y-%m-%d"),
            })
 
        documentos.extend(docs)
        print(f"   Páginas cargadas: {len(docs)}")
 
    return documentos
 
 
def dividir_documentos(documentos):
    print("\nCreando fragmentos")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documentos)
    print(f"Fragmentos creados: {len(chunks)}")
    return chunks
 
 
def obtener_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise Exception("Falta GOOGLE_API_KEY (variable de entorno o .env)")
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
 
 
def crear_vectorstore(chunks):
    print("\nGenerando embeddings")
    embeddings = obtener_embeddings()
 
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
 
    print("Guardando en Chroma")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
 
    total = vectorstore._collection.count()
    print(f"\nBase vectorial creada con {total} fragmentos indexados")
 
    if total == 0:
        # Si esto pasa, algo falló silenciosamente al generar embeddings
        raise Exception(
            "La base vectorial se creó pero quedó con 0 documentos indexados. "
            "Revisa la cuota/errores de la API de Google Embeddings."
        )
 
    return vectorstore
 
 
def base_de_datos_valida() -> bool:
    """
    Verifica que chroma_db exista Y tenga documentos realmente indexados.
    Reemplaza el chequeo ingenuo de 'la carpeta existe' que dejaba la app
    atascada con una base vacía si una ingesta anterior falló a la mitad.
    """
    if not os.path.exists(CHROMA_DIR):
        return False
    try:
        embeddings = obtener_embeddings()
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        return vectorstore._collection.count() > 0
    except Exception as e:
        print(f"Error verificando la base vectorial existente: {e}")
        return False
 
 
def main():
    print("Iniciando ingestión\n")
    documentos = cargar_pdfs()
    chunks = dividir_documentos(documentos)
    crear_vectorstore(chunks)
 
 
if __name__ == "__main__":
    main()