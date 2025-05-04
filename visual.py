import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import matplotlib.pyplot as plt
import seaborn as sns

# Настройки
CHROMA_DB_PATH = "./chroma_db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
QUERY = "у меня проблемы с vpn подключением?"  # Тестовый запрос

# Загрузка базы Chroma
embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

# Выполнение семантического поиска
results = db.similarity_search_with_score(QUERY, k=5)  # Берем 5 чанков с оценками схожести

# Создание DataFrame для результатов
df_results = pd.DataFrame({
    "source": [result[0].metadata["source"] for result in results],
    "human": [result[0].metadata["human"][:100] + "..." for result in results],
    "assistant": [result[0].metadata["assistant"][:100] + "..." for result in results],
    "text": [result[0].page_content[:100] + "..." for result in results],
    "score": [result[1] for result in results]  # Оценка схожести
})

# Вывод таблицы
print(f"\nРезультаты поиска для запроса: {QUERY}")
print(df_results)

# Визуализация оценок схожести
plt.figure(figsize=(8, 6))
sns.barplot(data=df_results, x="score", y=df_results.index, hue="source")
plt.title(f"Оценки схожести для запроса: {QUERY}")
plt.xlabel("Оценка схожести (меньше = лучше)")
plt.ylabel("Чанк")
plt.tight_layout()
plt.savefig("search_scores.png")
plt.show()

# Сохранение результатов в CSV
df_results.to_csv("search_results.csv", index=False)
print("\nРезультаты поиска сохранены в 'search_results.csv'")