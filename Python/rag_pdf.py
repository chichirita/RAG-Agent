import os
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

logger.add("log/02_Simple_RAG_PDF.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")


def get_index_db():
    logger.debug('...get_index_db')
    logger.debug('Embeddings')
    from langchain_huggingface import HuggingFaceEmbeddings
    model_id = 'intfloat/multilingual-e5-large'
    model_kwargs = {'device': 'cpu'}
    # model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs
    )

    db_file_name = 'db/db_01'
    logger.debug('Загрузка векторной Базы-Знаний из файла')
    file_path = db_file_name + "/index.faiss"
    import os.path
    if os.path.exists(file_path):
        logger.debug('Уже существует векторная База-знаний')
        db = FAISS.load_local(db_file_name, embeddings, allow_dangerous_deserialization=True)

    else:
        logger.debug('Еще не создана векторная База-Знаний')
        from langchain_community.document_loaders import PyPDFLoader

        dir = 'pdf'
        logger.debug(f'Document loaders. dir={dir}')
        documents = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".pdf"):
                    logger.debug(f'root={root} file={file}')
                    loader = PyPDFLoader(os.path.join(root, file))
                    documents.extend(loader.load())


        logger.debug('Разделение на chunks')
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=150)
        source_chunks = text_splitter.split_documents(documents)
        logger.debug(type(source_chunks))
        logger.debug(len(source_chunks))
        logger.debug(source_chunks[10].metadata)
        logger.debug(source_chunks[10].page_content)


        logger.debug('Векторная База-Знаний')
        db = FAISS.from_documents(source_chunks, embeddings)

        logger.debug('Сохранение векторной Базы-Знаний в файл')
        db.save_local(db_file_name)

    return db

def get_message_content(topic, db, NUMBER_RELEVANT_CHUNKS):
    import re
    logger.debug('...get_message_content: Similarity search')
    docs = db.similarity_search(topic, k = NUMBER_RELEVANT_CHUNKS)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n#### {i+1} Relevant chunk ####\n' + str(doc.metadata) + '\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    logger.debug(message_content)
    return message_content


def get_model_response(topic, message_content):
    logger.debug('...get_model_response')

    from langchain_ollama import ChatOllama
    logger.debug('LLM')
    local_llm = "bambucha/saiga-llama3:8b"
    llm = ChatOllama(model=local_llm, temperature=0)


    rag_prompt = """Ты являешься менеджером компании PolarAgency и отвечаешь на вопросы, основываясь только на предоставленном контексте.
    Вот контекст, который нужно использовать для ответа:
    {context}
    Внимательно проанализируй приведённый контекст.
    Теперь ознакомься с вопросом пользователя:
    {question}
    Дай ответ от лица менеджера компании PolarAgency, используя только вышеуказанный контекст.
Ф   ормулируй кратко, не более трёх предложений. На третье твое сообщение попроси контакт пользователя(номер телефона, почта) с целью дальнейщей связи."""

    rag_prompt_formatted = rag_prompt.format(context=message_content, question=topic)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    model_response = generation.content
    logger.debug(model_response)
    return model_response

def chat_loop(db, NUMBER_RELEVANT_CHUNKS=3):
    history = []
    print("💬 Чат с PolarAgency. Введите 'exit' для выхода.\n")

    while True:
        user_input = input("Вы: ")
        if user_input.lower() in ["exit", "quit", "выход"]:
            print("Чат завершён.")
            break

        message_content = get_message_content(user_input, db, NUMBER_RELEVANT_CHUNKS)

        user_questions_count = sum(isinstance(m, HumanMessage) for m in history) + 1

        context_with_history = "\n".join([
            f"Пользователь: {m.content}" if isinstance(m, HumanMessage) else f"PolarAgency: {m.content}"
            for m in history
        ])

        rag_prompt = f"""
Ты менеджер компании PolarAgency и отвечаешь на вопросы, основываясь только на предоставленном контексте.

Контекст:
{message_content}

История чата:
{context_with_history}

Вопрос пользователя:
{user_input}
Дай ответ от лица менеджера компании PolarAgency, используя только вышеуказанный контекст.
Формулируй кратко, не более трёх предложений. На третье твое сообщение попроси контакт пользователя(номер телефона, почта) с целью дальнейщей связи.
.
"""

        if user_questions_count == 3:
            rag_prompt += """
Так как это уже третий вопрос пользователя, обязательно в конце ответа попроси оставить контактные данные (телефон или e-mail) для дальнейшей связи.
"""

        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="bambucha/saiga-llama3:8b", temperature=0)

        print("PolarAgency: ", end="", flush=True)
        response_text = ""
        for chunk in llm.stream([HumanMessage(content=rag_prompt)]):
            token = chunk.content
            print(token, end="", flush=True)
            response_text += token
        print("\n")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response_text))


if __name__ == "__main__":
    db = get_index_db()
    chat_loop(db)
