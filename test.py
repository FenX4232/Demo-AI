import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Загрузка существующей базы Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 2. Получение всех документов (чанков)
all_docs = db.get(include=["metadatas", "documents"])  # Получаем тексты и метаданные

# 3. Вывод информации о базе
print(f"Общее количество чанков в базе: {len(all_docs['documents'])}")
print("\nПример первых 3 чанков (или всех, если их меньше):")
for i, (doc, metadata) in enumerate(zip(all_docs['documents'][:3], all_docs['metadatas'][:3])):
    print(f"\nЧанк {i+1}:")
    print(f"Текст (первые 500 символов): {doc[:500]}...")
    print(f"Метаданные: {metadata}")

# 4. Тестовый семантический поиск
query = "Как исправить проблему с сетевым подключением?"  # Замените на релевантный запрос
results = db.similarity_search(query, k=3)  # Ищем 3 наиболее похожих чанка

print("\nРезультаты семантического поиска:")
for i, result in enumerate(results):
    print(f"\nРезультат {i+1}:")
    print(f"Текст (первые 500 символов): {result.page_content[:500]}...")
    print(f"Метаданные: {result.metadata}")