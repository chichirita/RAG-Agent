from loguru import logger

def get_model_response(topic):
    from langchain_ollama import ChatOllama
    local_llm = "llama3.2:3b-instruct-fp16"
    llm = ChatOllama(model=local_llm, temperature=0)


    rag_prompt = """Ты являешься помощником для выполнения заданий по ответам на вопросы. 
    Внимательно подумайте над вопросом пользователя:
    {question}
    Дайте ответ на этот вопрос. 
    Используйте не более трех предложений и будьте лаконичны в ответе.
    Ответ:"""

    from langchain_core.messages import HumanMessage
    rag_prompt_formatted = rag_prompt.format( question=topic)

    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    model_response = generation.content
    logger.debug(model_response)
    return model_response

if __name__ == "__main__":
    topic = 'Объясни понятие RAG (Retrieval-Augmented Generation).'
    logger.debug(topic)

    model_response = get_model_response(topic)
