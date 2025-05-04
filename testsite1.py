from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
import requests
import json
import webbrowser
import threading
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import uuid
from typing import Dict, List

# Настройки
API_KEY = "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"  # Ваш ключ OpenRouter
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_HISTORY = 5  # Максимальное количество сообщений в истории

app = FastAPI()

# Инициализация Chroma и эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Хранение сессий (в памяти, для простоты)
sessions: Dict[str, List[Dict[str, str]]] = {}

# Функция отправки запроса к OpenRouter с контекстом и историей
def ask_ai(user_message: str, session_id: str, model: str = "deepseek/deepseek-chat-v3-0324:free", temperature: float = 0.7):
    # Получаем или создаем историю для сессии
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]

    # Выполняем семантический поиск в базе Chroma
    search_results = db.similarity_search(user_message, k=3)
    context = "\n\n".join([
        f"Вопрос: {result.metadata['human']}\nОтвет: {result.metadata['assistant']}"
        for result in search_results
    ])

    # Формируем историю диалога для промпта
    history_text = "\n".join([
        f"Пользователь: {msg['user']}\nИИ: {msg['ai']}"
        for msg in history[-MAX_HISTORY:]
    ])

    # Формируем промпт
    prompt = f"""
Вы - ИИ-помощник технической поддержки. Используйте следующий контекст и историю диалога для ответа на запрос пользователя:

Контекст из базы знаний:
{context}

История диалога:
{history_text}

Текущий запрос пользователя: {user_message}

Предоставьте точный и полезный ответ на русском языке, учитывая контекст и историю.
"""

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        ai_response = result["choices"][0]["message"]["content"]

        # Сохраняем сообщение и ответ в историю
        history.append({"user": user_message, "ai": ai_response})
        sessions[session_id] = history[-MAX_HISTORY:]  # Ограничиваем историю

        return ai_response
    except Exception as e:
        return f"Ошибка при обращении к API: {e}"

# HTML-шаблон главной страницы
def render_form(session_id: str, answer: str = "", history: List[Dict[str, str]] = None):
    history_html = ""
    if history:
        history_html = "<h2>История диалога:</h2><div style='text-align: left; max-width: 70%; margin: 20px auto; background: #ffffffcc; padding: 15px; border-radius: 10px;'>"
        for msg in history:
            history_html += f"<p><strong>Вы:</strong> {msg['user']}</p><p><strong>ИИ:</strong> {msg['ai']}</p><hr>"
        history_html += "</div>"

    return f"""
    <html>
        <head>
            <title>Помощник ИИ</title>
            <style>
                body {{
                    font-family: 'Arial', sans-serif;
                    background: linear-gradient(to right, #83a4d4, #b6fbff);
                    text-align: center;
                    padding-top: 50px;
                }}
                textarea {{
                    width: 60%;
                    height: 150px;
                    padding: 10px;
                    border-radius: 10px;
                    border: 1px solid #ccc;
                    font-size: 16px;
                }}
                select, input[type=number], input[type=hidden] {{
                    margin-top: 10px;
                    padding: 8px;
                    font-size: 16px;
                    border-radius: 10px;
                    border: 1px solid #ccc;
                }}
                button {{
                    margin-top: 20px;
                    padding: 10px 20px;
                    font-size: 18px;
                    border-radius: 10px;
                    border: none;
                    background-color: #4CAF50;
                    color: white;
                    cursor: pointer;
                }}
                button:hover {{
                    background-color: #45a049;
                }}
                .answer {{
                    margin-top: 40px;
                    background: #ffffffcc;
                    padding: 20px;
                    border-radius: 10px;
                    display: inline-block;
                    max-width: 70%;
                    word-wrap: break-word;
                }}
            </style>
        </head>
        <body>
            <h1>ИИ-помощник сотрудников</h1>
            <form action="/ask" method="post">
                <input type="hidden" name="session_id" value="{session_id}">
                <textarea name="message" placeholder="Опишите вашу проблему..."></textarea><br><br>
                <input type="number" step="0.1" min="0" max="1" name="temperature" value="0.7" placeholder="Temperature"><br><br>
                <button type="submit">Отправить</button>
            </form>
            {history_html}
            {"<div class='answer'><h2>Последний ответ ИИ:</h2><p>" + answer + "</p></div>" if answer else ""}
        </body>
    </html>
    """

# Главная страница
@app.get("/", response_class=HTMLResponse)
def get_form():
    session_id = str(uuid.uuid4())  # Генерируем новый session_id
    return HTMLResponse(content=render_form(session_id=session_id))

# Обработка запроса пользователя
@app.post("/ask", response_class=HTMLResponse)
async def post_form(
    session_id: str = Form(...),
    message: str = Form(...),
    model: str = Form("deepseek/deepseek-chat-v3-0324:free"),
    temperature: float = Form(0.7)
):
    ai_response = ask_ai(message, session_id, model, temperature)
    history = sessions.get(session_id, [])
    return HTMLResponse(content=render_form(session_id=session_id, answer=ai_response, history=history))

# Открытие браузера автоматически
def open_browser():
    webbrowser.open("http://localhost:8000")

# Запуск сервера
if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)