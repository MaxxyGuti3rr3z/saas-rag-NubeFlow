# 🤖 NubeFlow AI Agent

## Agente inteligente RAG para consulta de documentos empresariales

NubeFlow es un agente de inteligencia artificial basado en arquitectura **RAG (Retrieval
Augmented Generation)** capaz de responder preguntas utilizando información contenida en
documentos internos de la empresa (Recursos Humanos, Financiero y Producto/Manual de
Usuario).

El objetivo del proyecto es facilitar la búsqueda de información dentro de documentos
empresariales, permitiendo que los colaboradores realicen preguntas en lenguaje natural y
reciban respuestas fundamentadas en la documentación disponible, siempre citando la
fuente.

🔗 **Demo en vivo:** [saas-rag-nubeflow-ayau4mnstt6xqsxtjwt758.streamlit.app](https://saas-rag-nubeflow-ayau4mnstt6xqsxtjwt758.streamlit.app/)

---

## 🚀 Características principales

- 📄 Procesamiento automático de documentos PDF, organizados por categoría de negocio
  (Recursos Humanos, Financiero, Producto/Usuario).
- ✂️ División del contenido en fragmentos (*chunks*) con metadatos de origen.
- 🧠 Generación de embeddings mediante Google Gemini (`gemini-embedding-001`).
- 🔎 Búsqueda semántica utilizando una base vectorial ChromaDB.
- 🎯 **Reranking simple** por coincidencia de palabras clave, además del score semántico.
- 🧵 **Filtrado opcional por categoría** de documento (RH / Financiero / Usuario / Todas).
- ⚖️ **Umbral de confianza**: si ningún fragmento recuperado es suficientemente relevante,
  el agente no inventa una respuesta.
- 🤖 Generación de respuestas mediante modelos LLM de Groq (`llama-3.3-70b-versatile`).
- 📚 Citación de las fuentes utilizadas en cada respuesta.
- 🚫 Manejo explícito de "no lo sé" cuando la pregunta no está cubierta por los documentos.
- 💬 Interfaz conversacional en Streamlit, con historial de conversación.
- 👍👎 **Botones de feedback** por respuesta.
- 🧾 **Registro de ejecución**: cada interacción (y cada feedback) queda registrada con
  timestamp, fuentes y scores de relevancia, descargable en JSON Lines desde la interfaz.
- ☁️ Desplegado públicamente en **Streamlit Community Cloud**.

---

## 🏗️ Arquitectura del proyecto

```
Documentos PDF (data/RecursosHumanos, data/Finanzas, data/Usuario)
      │
      ▼
Ingesta (ingest.py) — PyPDFLoader + metadatos (fuente, categoría, fecha)
      │
      ▼
Chunking — RecursiveCharacterTextSplitter (1000 caracteres, 150 de overlap)
      │
      ▼
Embeddings — Google Gemini (gemini-embedding-001)
      │
      ▼
Base vectorial — ChromaDB (persistida en /tmp en la nube)
      │
      ▼
Pregunta del usuario (+ filtro de categoría opcional)
      │
      ▼
Búsqueda semántica (similarity_search_with_relevance_scores, top-k)
      │
      ▼
Reranking simple por palabras clave
      │
      ▼
Umbral de confianza (0.5) — descarta fragmentos poco relevantes
      │
      ▼
Contexto relevante + metadatos de fuente
      │
      ▼
Modelo Groq (Llama 3.3 70B) — genera la respuesta restringida al contexto
      │
      ▼
Interfaz Streamlit — respuesta + fuentes + feedback + historial
      │
      ▼
Registro de ejecución (JSON Lines en /tmp, descargable)
```

---

## 🛠️ Tecnologías utilizadas

**Lenguaje**
- Python 3.x

**Inteligencia Artificial**
- LangChain
- Google Gemini Embeddings
- Groq LLM (Llama 3.3 70B)

**Base de datos vectorial**
- ChromaDB

**Interfaz**
- Streamlit

**Gestión de variables**
- python-dotenv / Streamlit Secrets

**Hosting**
- Streamlit Community Cloud

---

## 📂 Estructura del proyecto

```
NubeFlow/
│
├── app.py                        # Interfaz web Streamlit
├── src/
│   ├── ingest.py                 # Procesamiento e indexación de documentos
│   └── agent.py                  # Lógica del agente RAG (búsqueda, reranking, logging)
├── data/
│   ├── RecursosHumanos/          # Políticas de RH, beneficios, onboarding
│   ├── Finanzas/                 # Resultados, presupuesto, KPIs
│   └── Usuario/                  # Manual de uso / base de conocimiento del producto
├── requirements.txt               # Dependencias del proyecto
├── .env.example                   # Plantilla de variables de entorno (no subir .env real)
├── .gitignore
└── README.md
```

> Nota: la carpeta `chroma_db/` **no se versiona** en el repositorio. Se genera en tiempo de
> ejecución (en `/tmp`) a partir de los documentos en `data/`. Lo mismo aplica al registro
> de ejecución (`nubeflow_logs/`), que también vive en `/tmp`.

---

## ⚙️ Instalación local

```bash
git clone URL_DEL_REPOSITORIO
cd NubeFlow
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔑 Variables de entorno

Crear un archivo `.env` en la raíz del proyecto (basado en `.env.example`):
```env
GOOGLE_API_KEY=tu_api_key_de_google
GROQ_API_KEY=tu_api_key_de_groq
```

- **Google Gemini** → generación de embeddings (gratis en [Google AI Studio](https://aistudio.google.com/apikey)).
- **Groq** → generación de las respuestas del agente (gratis en [Groq Console](https://console.groq.com)).

---

## 📚 Crear la base de conocimiento (ingesta)

```bash
python src/ingest.py
```

Este proceso:
1. Lee los documentos PDF almacenados en `data/` (y subcarpetas por categoría).
2. Divide el contenido en fragmentos con overlap.
3. Genera embeddings con Gemini.
4. Guarda los vectores e índices en ChromaDB.

---

## ▶️ Ejecutar la aplicación localmente

```bash
streamlit run app.py
```

Abrir en el navegador: `http://localhost:8501`

La primera vez que se corre, la app detecta que no existe base vectorial (o que está vacía)
y ejecuta la ingesta automáticamente antes de habilitar el chat.

---

## ☁️ Deploy en Streamlit Community Cloud

Este proyecto está desplegado en **Streamlit Community Cloud** en lugar de Oracle Cloud
Infrastructure (OCI), como alternativa válida sugerida por el propio desafío ("estas son
sugerencias, no obligaciones... si contamos con una herramienta que conocemos mejor,
podemos usarla").

1. Subir el repositorio a GitHub (público).
2. Entrar a [share.streamlit.io](https://share.streamlit.io) y crear una nueva app, apuntando
   al repositorio y a `app.py` como archivo principal.
3. En **Settings → Secrets**, configurar:
   ```toml
   GOOGLE_API_KEY = "tu_api_key_de_google"
   GROQ_API_KEY = "tu_api_key_de_groq"
   ```
4. Desplegar. La primera carga procesa los documentos y construye la base vectorial
   automáticamente (puede tardar 1-2 minutos).

> 📌 **Nota técnica:** ChromaDB usa SQLite por debajo, que necesita locks de archivo reales.
> El filesystem del propio repositorio en Streamlit Cloud no los soporta correctamente
> (produce el error `attempt to write a readonly database`), por lo que la base vectorial y
> el registro de ejecución se persisten en `/tmp` en tiempo de ejecución.

## 📸 Capturas y evidencia

Chat funcionando en producción

<img width="1917" height="912" alt="Captura de pantalla 2026-07-27 222219" src="https://github.com/user-attachments/assets/f59ede7e-743f-4b86-b859-d0db267e5dbc" />

 Ante pregunta sin respuesta

<img width="1917" height="912" alt="Captura de pantalla 2026-07-27 222419" src="https://github.com/user-attachments/assets/7ef7dabc-c145-4b44-ad70-859286e368be" />



---

## 💬 Ejemplos de preguntas

| Pregunta | Categoría / documento |
|---|---|
| "¿Qué es NubeFlow?" | Usuario (resumen general) |
| "¿Cuántos días de vacaciones tengo por año?" | Recursos Humanos |
| "¿Cuánto cubre el seguro de salud?" | Recursos Humanos |
| "¿Cuál fue la utilidad neta del último año?" | Finanzas |
| "¿Cómo creo una automatización en NubeFlow?" | Usuario / Manual |
| "¿Dónde falleció San Martín?" *(no existe en la base)* | El agente responde que no encontró la información, en vez de inventarla |

Desde la barra lateral se puede restringir la búsqueda a una categoría específica (por
ejemplo, "Recursos Humanos") para evitar que el agente traiga contexto de otras áreas.

---

## 🧩 Control de alucinaciones

1. El agente busca los fragmentos más relevantes en ChromaDB antes de responder.
2. Aplica un reranking simple por palabras clave sobre los candidatos recuperados.
3. Descarta los fragmentos que no superan un umbral de confianza (0.5) — si ninguno lo
   supera, ni siquiera se llama al LLM.
4. Solo envía al modelo el contexto recuperado (no usa conocimiento externo).
5. El prompt instruye explícitamente a admitir cuando la información no está disponible.

---

## 🧾 Registro de ejecución

Cada pregunta (y cada feedback dado) queda registrada en un archivo JSON Lines
(`/tmp/nubeflow_logs/agent_log.jsonl` en producción), con:
- Timestamp
- Pregunta del usuario
- Fuentes utilizadas
- Si se encontró contexto relevante (umbral de confianza)
- Scores de relevancia de los candidatos
- Respuesta generada
- Feedback del usuario (si lo dio)

El archivo se puede descargar directamente desde la barra lateral de la aplicación
("📥 Descargar log"), para auditoría o como evidencia de ejecución.

---

## 📌 Estado actual del proyecto

- ✅ Procesamiento de documentos (PDF, múltiples categorías)
- ✅ Chunking con metadatos (fuente, categoría, fecha)
- ✅ Generación de embeddings (Gemini)
- ✅ Base vectorial (ChromaDB, persistida en `/tmp` en producción)
- ✅ Búsqueda semántica + reranking simple por palabras clave
- ✅ Filtrado de búsqueda por categoría de documento
- ✅ Umbral de confianza en la búsqueda semántica
- ✅ Agente RAG funcional (Groq)
- ✅ Interfaz web (Streamlit) con historial de conversación
- ✅ Citación de fuentes en cada respuesta
- ✅ Fallback ante preguntas sin respuesta en los documentos
- ✅ Aviso explícito de que se trata de un agente de IA
- ✅ Botones de feedback (👍/👎) por respuesta
- ✅ Registro de ejecución (logs de pregunta/contexto/respuesta/timestamp/feedback)
- ✅ Deploy público (Streamlit Community Cloud)

---

## 🔮 Próximas mejoras

- Soporte para más formatos: Word, Excel, CSV, Markdown.
- Dashboard de monitoreo con las métricas del log (tasa de preguntas sin respuesta,
  feedback negativo, tiempo de respuesta).

---

## 📄 Licencia

Proyecto educativo desarrollado como parte del programa Oracle Next Education (ONE) / Alura.
