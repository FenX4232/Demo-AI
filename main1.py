import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from typing import List, Dict

# 1. Загрузка данных из CSV
def load_data_from_csv(file_path: str) -> List[Dict]:
    """Загружает данные из CSV файла и возвращает список словарей"""
    df = pd.read_csv(file_path)
    documents = []
    
    for _, row in df.iterrows():
        text = str(row['text']).strip()
        if "### Human:" in text and "### Assistant:" in text:
            # Разделяем на Human и Assistant части
            human_part = text.split("### Assistant:")[0].replace("### Human:", "").strip()
            assistant_part = text.split("### Assistant:")[1].strip()
            
            documents.append({
                "original": text,
                "human": human_part,
                "assistant": assistant_part
            })
    
    return documents

# 2. Загрузка и обработка данных
documents = load_data_from_csv("train.csv")

# 3. Подготовка текстов для векторной базы
texts = [doc["original"] for doc in documents]
metadatas = [{
    "source": f"doc_{i}", 
    "human": doc["human"], 
    "assistant": doc["assistant"]
} for i, doc in enumerate(documents)]

# 4. Разделение текста на чанки
text_splitter = CharacterTextSplitter(
    chunk_size=6000,  # Увеличен до 6000 для чанка размером 5557
    chunk_overlap=200,
    separator="###",
    keep_separator=True
)
split_texts = []
split_metadatas = []

for text, metadata in zip(texts, metadatas):
    chunks = text_splitter.split_text(text)
    for chunk in chunks:
        split_texts.append(chunk)
        split_metadatas.append(metadata)

# 5. Создание векторной базы данных
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma.from_texts(
    split_texts,
    embeddings,
    metadatas=split_metadatas,
    persist_directory="./chroma_db"
)

print(f"Успешно загружено {len(documents)} документов и создано {len(split_texts)} чанков.")