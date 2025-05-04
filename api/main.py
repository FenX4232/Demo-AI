from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import requests
import json
import webbrowser
import threading
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
from typing import Dict, List
from chromadb import HttpClient

# Укажите URL вашего Chroma-сервера (Ngrok/Cloudflare/прямой IP)
CHROMA_SERVER_URL = "http://213.88.11.15:8000"  # или "https://abc123.ngrok.io"

# Подключение к удалённой Chroma
db = HttpClient(
    host="213.88.11.15",  # или "abc123.ngrok.io" (без http://)
    port=8000,              # порт Chroma
    # Если есть аутентификация:
    auth_provider="basic",
    auth_credentials="user:pass"  # если настроено
)

# Настройки
API_KEY = "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"
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_HISTORY = 5

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Папка с шаблонами

# Инициализация Chroma и эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Хранение сессий
sessions: Dict[str, List[Dict[str, str]]] = {}

# Функция запроса к OpenRouter
def ask_ai(user_message: str, session_id: str, model: str = "deepseek/deepseek-chat-v3-0324:free", temperature: float = 0.7):
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]

    # Семантический поиск в Chroma
    search_results = db.similarity_search(user_message, k=3)
    context = "\n\n".join([
        f"Вопрос: {result.metadata['human']}\nОтвет: {result.metadata['assistant']}"
        for result in search_results
    ])

    # Формируем историю диалога
    history_text = "\n".join([
        f"Пользователь: {msg['user']}\nИИ: {msg['ai']}"
        for msg in history[-MAX_HISTORY:]
    ])

    # Промпт для ИИ
    prompt = f"""
Ты — ИИ-агент службы технической поддержки. Ты общаешься с пользователями, которые сталкиваются с проблемами, связанными с программным обеспечением, компьютерами, доступом к системам и другим техническим обслуживанием.

У тебя есть доступ к базе знаний компании через семантический поиск. Используй найденные статьи и документы для формирования ответа.

Если вопрос неполный — задай уточняющий вопрос. Если обращение типовое — отвечай кратко, понятно и по существу.

Формат ответа:

Четкое и структурированное объяснение (можно с пунктами).

При необходимости — ссылка на источник или название документа, если он найден через поиск.

Не выдумывай факты. Если информации нет — скажи об этом и предложи связаться с оператором.

Примеры задач, которые ты решаешь:

«Не запускается Outlook»

«Нет доступа к общему диску»

«Как установить VPN»

«Ошибка 0x80070005 при обновлении Windows»

Тон общения: профессиональный, вежливый, без излишне формального языка.

Контекст: {context}

История:
{history_text}

Запрос: {user_message}
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        ai_response = result["choices"][0]["message"]["content"]

        # Сохраняем в историю
        history.append({"user": user_message, "ai": ai_response})
        sessions[session_id] = history[-MAX_HISTORY:]

        return ai_response
    except Exception as e:
        return f"Ошибка при обращении к API: {e}"

# Главная страница
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    session_id = str(uuid.uuid4())
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "session_id": session_id}
    )

# Обработка запроса
@app.post("/ask", response_class=HTMLResponse)
async def post_form(
    request: Request,
    session_id: str = Form(...),
    message: str = Form(...),
    temperature: float = Form(0.7)
):
    ai_response = ask_ai(message, session_id, temperature=temperature)
    history = sessions.get(session_id, [])
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "session_id": session_id,
            "answer": ai_response,
            "history": history
        }
    )

