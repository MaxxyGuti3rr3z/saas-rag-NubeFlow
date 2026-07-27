

import os
import glob
import shutil

from dotenv import load_dotenv

from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()




DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

EMBEDDING_MODEL = "models/gemini-embedding-001"


def cargar_pdfs():

    documentos = []

    archivos = glob.glob(
        os.path.join(DATA_DIR, "**", "*.pdf"),
         recursive=True
    )

    if not archivos:
        raise Exception(
            "No se encontraron archivos PDF en la carpeta data/"
        )


    print(f"PDFs encontrados: {len(archivos)}")


    for archivo in archivos:

        print(f"Procesando: {archivo}")

        loader = PyPDFLoader(archivo)

        docs = loader.load()


        nombre_archivo = os.path.basename(
            archivo
        )


      
        categoria = "General"

        if "recursos_humanos" in archivo.lower():
            categoria = "Recursos Humanos"

        elif "financiero" in archivo.lower():
            categoria = "Financiero"

        elif "operaciones" in archivo.lower():
            categoria = "Operaciones"



        for doc in docs:

            doc.metadata.update(
            {
                "fuente": nombre_archivo,
                "categoria": categoria,
                "fecha_procesamiento": datetime.now().strftime(
                    "%Y-%m-%d"
                )
            }
        )


        documentos.extend(docs)


        print(
            f"   Páginas cargadas: {len(docs)}"
            )   


    return documentos



def dividir_documentos(documentos):

    print("\nCreando fragmentos")


    splitter = RecursiveCharacterTextSplitter(

        chunk_size=1000,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]

    )


    chunks = splitter.split_documents(documentos)


    print(
        f"Fragmentos creados: {len(chunks)}"
    )


    return chunks



def crear_vectorstore(chunks):


    api_key = os.getenv(
        "GOOGLE_API_KEY"
    )


    if not api_key:

        raise Exception(
            "Falta GOOGLE_API_KEY en .env"
        )


    print(
        "\nGenerando embeddings"
    )


    embeddings = GoogleGenerativeAIEmbeddings(

        model=EMBEDDING_MODEL,

        google_api_key=api_key

    )



    if os.path.exists(CHROMA_DIR):

        shutil.rmtree(CHROMA_DIR)


    print(
        "Guardando en Chroma"
    )


    Chroma.from_documents(

        documents=chunks,

        embedding=embeddings,

        persist_directory=CHROMA_DIR

    )


    print(
        "\n Base vectorial creada"
    )



def main():

    print(
        "Iniciando ingestión\n"
    )


    documentos = cargar_pdfs()


    chunks = dividir_documentos(
        documentos
    )


    crear_vectorstore(
        chunks
    )


if __name__ == "__main__":
    main()