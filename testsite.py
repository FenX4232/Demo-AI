from fastapi import FastAPI, Form, HTTPException
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
import sqlite3
from datetime import datetime

# Настройки
API_KEY = "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"  # Ваш ключ OpenRouter
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_HISTORY = 5
OPERATOR_LIMIT = 5
COMPLEXITY_THRESHOLD = 1.2  # Увеличен порог релевантности
LENGTH_THRESHOLD = 300  # Увеличен порог длины запроса

app = FastAPI()

# Инициализация Chroma
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Инициализация SQLite
conn = sqlite3.connect("requests.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS requests (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    user_message TEXT,
    ai_response TEXT,
    operator_response TEXT,
    status TEXT,
    created_at TEXT,
    assigned_operator TEXT
)
""")
conn.commit()

# Хранилище сессий
sessions: Dict[str, List[Dict[str, str]]] = {}

# Оценка сложности запроса
def evaluate_complexity(user_message: str):
    # Длина запроса
    is_long = len(user_message) > LENGTH_THRESHOLD
    
    # Семантическая уверенность
    search_results = db.similarity_search_with_score(user_message, k=3)
    max_score = max([result[1] for result in search_results]) if search_results else float("inf")
    is_uncertain = max_score > COMPLEXITY_THRESHOLD
    
    # Ключевые слова
    complex_keywords = ["срочно", "не работает", "помогите", "ошибка"]
    has_keywords = any(keyword in user_message.lower() for keyword in complex_keywords)
    
    # Отладочный вывод
    print(f"Оценка запроса: '{user_message[:50]}...'")
    print(f"  Длина: {len(user_message)} (is_long: {is_long})")
    print(f"  Max score: {max_score:.2f} (is_uncertain: {is_uncertain})")
    print(f"  Ключевые слова: {has_keywords}")
    
    # Запрос считается сложным, только если выполнены несколько условий
    is_complex = (is_long and is_uncertain) or (has_keywords and is_uncertain)
    
    return is_complex, search_results

# Оценка нагрузки операторов
def get_operator_load(operator_id: str):
    cursor.execute("SELECT COUNT(*) FROM requests WHERE assigned_operator = ? AND status = 'pending'", (operator_id,))
    return cursor.fetchone()[0]

# Назначение оператора
def assign_operator():
    operators = ["operator1", "operator2"]
    for op in operators:
        load = get_operator_load(op)
        if load < OPERATOR_LIMIT:
            print(f"Назначен оператор: {op} (нагрузка: {load})")
            return op
    print("Все операторы заняты")
    return None

# Функция отправки запроса к OpenRouter
def ask_ai(user_message: str, session_id: str, model: str, temperature: float):
    history = sessions.get(session_id, [])
    search_results = db.similarity_search(user_message, k=3)
    context = "\n\n".join([f"Вопрос: {r.metadata['human']}\nОтвет: {r.metadata['assistant']}" for r in search_results])
    history_text = "\n".join([f"Пользователь: {msg['user']}\nИИ: {msg['ai']}" for msg in history[-MAX_HISTORY:]])

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
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "temperature": temperature, "messages": [{"role": "user", "content": prompt}]}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]
        
        # Сохраняем в историю
        history.append({"user": user_message, "ai": ai_response})
        sessions[session_id] = history[-MAX_HISTORY:]
        
        return ai_response
    except Exception as e:
        return f"Ошибка API: {e}"

# HTML-шаблон для пользователей
def render_user_form(session_id: str, answer: str = "", history: List[Dict[str, str]] = None, status: str = ""):
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
                body {{ font-family: Arial, sans-serif; background: linear-gradient(to right, #83a4d4, #b6fbff); text-align: center; padding-top: 50px; }}
                textarea {{ width: 60%; height: 150px; padding: 10px; border-radius: 10px; border: 1px solid #ccc; font-size: 16px; }}
                select, input[type=number], input[type=hidden] {{ margin-top: 10px; padding: 8px; font-size: 16px; border-radius: 10px; border: 1px solid #ccc; }}
                button {{ margin-top: 20px; padding: 10px 20px; font-size: 18px; border-radius: 10px; border: none; background-color: #4CAF50; color: white; cursor: pointer; }}
                button:hover {{ background-color: #45a049; }}
                .answer {{ margin-top: 40px; background: #ffffffcc; padding: 20px; border-radius: 10px; display: inline-block; max-width: 70%; word-wrap: break-word; }}
                .status {{ color: #555; font-size: 16px; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>ИИ-помощник сотрудников</h1>
            <form action="/ask" method="post">
                <input type="hidden" name="session_id" value="{session_id}">
                <textarea name="message" placeholder="Опишите вашу проблему..."></textarea><br><br>
                <select name="model">
                    <option value="deepseek/deepseek-chat-v3-0324:free">DeepSeek Chat v3 (Free)</option>
                    <option value="openai/gpt-3.5-turbo">GPT-3.5 Turbo</option>
                    <option value="openai/gpt-4">GPT-4</option>
                    <option value="anthropic/claude-3-haiku">Claude 3 Haiku</option>
                </select><br>
                <input type="number" step="0.1" min="0" max="1" name="temperature" value="0.7" placeholder="Temperature"><br><br>
                <button type="submit">Отправить</button>
            </form>
            <div class="status">{status}</div>
            {history_html}
            {"<div class='answer'><h2>Последний ответ:</h2><p>" + answer + "</p></div>" if answer else ""}
        </body>
    </html>
    """

# HTML-шаблон для оператора
def render_operator_panel(pending_requests: List[Dict]):
    requests_html = "<h2>Очередь запросов:</h2><div style='text-align: left; max-width: 80%; margin: 20px auto; background: #ffffffcc; padding: 15px; border-radius: 10px;'>"
    for req in pending_requests:
        history = sessions.get(req["session_id"], [])
        history_text = "<br>".join([f"Пользователь: {msg['user']}<br>ИИ: {msg['ai']}" for msg in history])
        requests_html += f"""
        <div style='margin-bottom: 20px;'>
            <p><strong>Запрос ID:</strong> {req['id']}</p>
            <p><strong>Сообщение:</strong> {req['user_message']}</p>
            <p><strong>История:</strong><br>{history_text}</p>
            <p><strong>Предложенный ИИ:</strong> {req['ai_response'] or 'Нет'}</p>
            <form action="/operator/respond" method="post">
                <input type="hidden" name="request_id" value="{req['id']}">
                <textarea name="response" placeholder="Введите ответ..." style="width: 100%; height: 100px;"></textarea><br>
                <button type="submit">Отправить ответ</button>
            </form>
        </div><hr>
        """
    requests_html += "</div>"

    return f"""
    <html>
        <head>
            <title>Панель оператора</title>
            <style>
                body {{ font-family: Arial, sans-serif; background: linear-gradient(to right, #83a4d4, #b6fbff); text-align: center; padding-top: 50px; }}
                textarea {{ width: 60%; padding: 10px; border-radius: 10px; border: 1px solid #ccc; font-size: 16px; }}
                button {{ padding: 10px 20px; font-size: 16px; border-radius: 10px; border: none; background-color: #4CAF50; color: white; cursor: pointer; }}
                button:hover {{ background-color: #45a049; }}
            </style>
        </head>
        <body>
            <h1>Панель оператора</h1>
            {requests_html}
        </body>
    </html>
    """

# Главная страница для пользователей
@app.get("/", response_class=HTMLResponse)
def get_form():
    session_id = str(uuid.uuid4())
    return HTMLResponse(content=render_user_form(session_id=session_id))

# Обработка запроса пользователя
@app.post("/ask", response_class=HTMLResponse)
async def post_form(
    session_id: str = Form(...),
    message: str = Form(...),
    model: str = Form("deepseek/deepseek-chat-v3-0324:free"),
    temperature: float = Form(0.7)
):
    request_id = str(uuid.uuid4())
    is_complex, search_results = evaluate_complexity(message)
    operator_id = assign_operator()

    # Сохраняем запрос в базу
    cursor.execute(
        "INSERT INTO requests (id, session_id, user_message, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (request_id, session_id, message, "pending", datetime.now().isoformat())
    )
    conn.commit()

    # Принудительно обрабатываем короткие запросы ИИ
    if len(message) < 100:
        is_complex = False
        print(f"Короткий запрос, обрабатывается ИИ: '{message[:50]}...'")

    if is_complex and operator_id:
        # Передаем оператору
        cursor.execute(
            "UPDATE requests SET status = ?, assigned_operator = ? WHERE id = ?",
            ("assigned", operator_id, request_id)
        )
        conn.commit()
        status = f"Запрос передан оператору (ID: {request_id})"
        return HTMLResponse(content=render_user_form(
            session_id=session_id,
            status=status,
            history=sessions.get(session_id, [])
        ))
    else:
        # Обрабатываем ИИ
        ai_response = ask_ai(message, session_id, model, temperature)
        cursor.execute(
            "UPDATE requests SET ai_response = ?, status = ? WHERE id = ?",
            (ai_response, "completed", request_id)
        )
        conn.commit()
        status = "Обработано ИИ"
        return HTMLResponse(content=render_user_form(
            session_id=session_id,
            answer=ai_response,
            history=sessions.get(session_id, []),
            status=status
        ))

# Панель оператора
@app.get("/operator", response_class=HTMLResponse)
def operator_panel():
    cursor.execute("SELECT * FROM requests WHERE status = 'assigned'")
    pending_requests = [
        {"id": row[0], "session_id": row[1], "user_message": row[2], "ai_response": row[3], "operator_response": row[4], "status": row[5], "assigned_operator": row[7]}
        for row in cursor.fetchall()
    ]
    return HTMLResponse(content=render_operator_panel(pending_requests))

# Обработка ответа оператора
@app.post("/operator/respond", response_class=HTMLResponse)
async def operator_respond(request_id: str = Form(...), response: str = Form(...)):
    cursor.execute("SELECT session_id FROM requests WHERE id = ?", (request_id,))
    result = cursor.fetchone()
    if not result:
        raise HTTPException(status_code=404, detail="Запрос не найден")
    session_id = result[0]

    cursor.execute(
        "UPDATE requests SET operator_response = ?, status = ? WHERE id = ?",
        (response, "completed", request_id)
    )
    conn.commit()

    # Добавляем ответ в историю
    if session_id in sessions:
        sessions[session_id].append({"user": "Оператор", "ai": response})
        sessions[session_id] = sessions[session_id][-MAX_HISTORY:]

    return HTMLResponse(content=render_operator_panel([]))

# Открытие браузера
def open_browser():
    webbrowser.open("http://localhost:8000")

# Запуск сервера
if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)