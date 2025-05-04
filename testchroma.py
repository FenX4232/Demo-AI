import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from statistics import mean, median

# Настройки
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Загрузка базы Chroma
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# 1. Анализ содержимого базы
all_docs = db.get(include=["metadatas", "documents"])
texts = all_docs["documents"]
metadatas = all_docs["metadatas"]

# Статистика
text_lengths = [len(t) for t in texts]
print(f"Общее количество чанков: {len(texts)}")
print(f"Средняя длина чанка: {mean(text_lengths):.1f} символов")
print(f"Медианная длина чанка: {median(text_lengths):.1f} символов")
print(f"Мин/Макс длина чанка: {min(text_lengths)}/{max(text_lengths)}")
print("\nПримеры чанков (первые 3):")
for i, (text, meta) in enumerate(zip(texts[:3], metadatas[:3])):
    print(f"\nЧанк {i+1}:")
    print(f"Текст (первые 100 символов): {text[:100]}...")
    print(f"Метаданные: {meta}")

# 2. Тестовые запросы
test_queries = [
    "Как перезагрузить роутер?",
    "У меня проблема с VPN",
    "Свяжи меня с оператором",
    "Wi-Fi не работает после обновления"
]

# Сохранение результатов поиска
results_data = []
for query in test_queries:
    results = db.similarity_search_with_score(query, k=3)
    print(f"\nЗапрос: {query}")
    for i, (result, score) in enumerate(results):
        print(f"  Чанк {i+1}: score={score:.2f}")
        print(f"    Вопрос: {result.metadata['human'][:100]}...")
        print(f"    Ответ: {result.metadata['assistant'][:100]}...")
        results_data.append({
            "query": query,
            "chunk_id": i+1,
            "score": score,
            "human": result.metadata["human"][:200],
            "assistant": result.metadata["assistant"][:200],
            "text": result.page_content[:200]
        })

# Сохранение в CSV
df = pd.DataFrame(results_data)
df.to_csv("chroma_diagnosis.csv", index=False)
print("\nРезультаты сохранены в 'chroma_diagnosis.csv'")