# 🤖 NubeFlow AI
 
## Agente inteligente RAG para consulta de documentos empresariales
 
NubeFlow es un agente de inteligencia artificial basado en arquitectura **RAG (Retrieval
Augmented Generation)** capaz de responder preguntas utilizando información contenida en
documentos internos.
 
El objetivo del proyecto es facilitar la búsqueda de información dentro de documentos
empresariales, permitiendo que los usuarios realicen preguntas en lenguaje natural y
reciban respuestas fundamentadas en la documentación disponible.
 
---
 
## 🚀 Características principales
 
- 📄 Procesamiento automático de documentos PDF.
- ✂️ División del contenido en fragmentos (*chunks*).
- 🧠 Generación de embeddings mediante Google Gemini.
- 🔎 Búsqueda semántica utilizando una base vectorial ChromaDB.
- 🤖 Generación de respuestas mediante modelos LLM de Groq.
- 📚 Recuperación de fuentes utilizadas para responder.
- 💬 Interfaz conversacional desarrollada con Streamlit.
---
 
## 🏗️ Arquitectura del proyecto
 
El flujo general de NubeFlow funciona de la siguiente manera:
 
```
Documentos PDF
      │
      ▼
Proceso de ingestión
      │
      ▼
División en chunks
      │
      ▼
Embeddings Gemini
      │
      ▼
Base vectorial ChromaDB
      │
      ▼
Pregunta del usuario
      │
      ▼
Búsqueda semántica
      │
      ▼
Contexto relevante
      │
      ▼
Modelo Groq Llama
      │
      ▼
Respuesta generada
      │
      ▼
Interfaz Streamlit
```
 
---
 
## 🛠️ Tecnologías utilizadas
 
**Lenguaje**
- Python 3.x
**Inteligencia Artificial**
- LangChain
- Google Gemini Embeddings
- Groq LLM
**Base de datos vectorial**
- ChromaDB
**Interfaz**
- Streamlit
**Gestión de variables**
- python-dotenv
---
 
## 📂 Estructura del proyecto
 
```
NubeFlow/
│
├── app.py                 # Interfaz web Streamlit
├── ingest.py               # Procesamiento e indexación de documentos
├── agent.py                # Lógica del agente RAG
├── data/                   # Documentos utilizados por el agente
├── chroma_db/              # Base vectorial generada
├── requirements.txt        # Dependencias del proyecto
├── .env                    # Variables de entorno
└── README.md
```
 
---
 
## ⚙️ Instalación
 
Clonar el repositorio:
```bash
git clone URL_DEL_REPOSITORIO
```
 
Entrar al proyecto:
```bash
cd NubeFlow
```
 
Crear entorno virtual:
```bash
python -m venv venv
```
 
Activar entorno virtual:
 
Windows:
```bash
venv\Scripts\activate
```
 
Linux/Mac:
```bash
source venv/bin/activate
```
 
Instalar dependencias:
```bash
pip install -r requirements.txt
```
 
---
 
## 🔑 Variables de entorno
 
Crear un archivo `.env` en la raíz del proyecto:
```env
GOOGLE_API_KEY=tu_api_key_de_google
GROQ_API_KEY=tu_api_key_de_groq
```
 
Estas claves permiten:
- **Google Gemini** → generación de embeddings.
- **Groq** → generación de respuestas del agente.
---
 
## 📚 Crear la base de conocimiento
 
Antes de ejecutar el agente es necesario procesar los documentos:
```bash
python ingest.py
```
 
Este proceso:
1. Lee los documentos almacenados en `data/`.
2. Divide el contenido en fragmentos.
3. Genera embeddings.
4. Guarda la información en ChromaDB.
---
 
## ▶️ Ejecutar la aplicación
 
Ejecutar Streamlit:
```bash
streamlit run app.py
```
 
Luego abrir en el navegador:
```
http://localhost:8501
```
 
---
 
## 💬 Ejemplos de preguntas
 
Ejemplos de consultas:
```
¿Cuáles son las políticas disponibles en el documento?
```
```
¿Qué información contiene la guía de servicios?
```
```
¿Cuáles son los procedimientos definidos?
```
 
El agente responderá utilizando únicamente la información encontrada dentro de los
documentos procesados.
 
---
 
## 🧩 Control de respuestas
 
NubeFlow utiliza una estrategia RAG para reducir respuestas incorrectas:
1. Busca información relevante en ChromaDB.
2. Envía únicamente el contexto encontrado al modelo.
3. Solicita al modelo responder solamente utilizando dicho contexto.
4. Muestra las fuentes utilizadas.
---
 
## 📌 Estado actual del proyecto
 
- ✅ Procesamiento de documentos
- ✅ Generación de embeddings
- ✅ Base vectorial
- ✅ Agente RAG funcional
- ✅ Interfaz web
- ✅ Recuperación de fuentes
---
 
## 🔮 Próximas mejoras
 
- Soporte para más formatos:
  - Word
  - Excel
  - CSV
  - Markdown
- Integración con almacenamiento en la nube.
- Deploy en servicios cloud.
- Sistema de evaluación de respuestas.
- Mejor gestión de usuarios.