import os
from flask import Flask, render_template, request, Response
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from loguru import logger
from Simple_RAG_PDF import get_index_db, get_message_content
import sqlite3

app = Flask(__name__)
db = get_index_db()
NUMBER_RELEVANT_CHUNKS = 3

history = [
    AIMessage(content="Приветствую! Меня зовут Марта, я менеджер на основе ИИ PolarAgency. Подскажите, как могу к вам обращаться?")
]

content=system_prompt = SystemMessage(content="""
Ты — менеджер компании PolarAgency, женщина.
Отвечай строго от первого лица, от имени Марты.
Никогда не используй фразу "Ответ от менеджера компании PolarAgency" или что-то похожее.
Не пиши вводные слова перед ответом — сразу отвечай по делу.
Ни в коем случае не здороваться с пользователем.
Отвечай максимально кратко, по делу, максимум 3 предложения.
Если вопрос не связан с услугами компании, отвечай: "Не знаю".
Отвечай на языке пользователя.
""")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["message"]

    user_messages_count = sum(isinstance(m, HumanMessage) for m in history)

    if user_messages_count == 0:
        history.append(HumanMessage(content=user_input))
        response_text = f"Приятно познакомиться, {user_input}! Чем могу помочь?"
        history.append(AIMessage(content=response_text))
        return Response(response_text, mimetype="text/plain")

    ask_for_contact = (user_messages_count == 2)

    message_content = get_message_content(user_input, db, NUMBER_RELEVANT_CHUNKS)

    context_with_history = "\n".join([
        f"Пользователь: {m.content}" for m in history if isinstance(m, HumanMessage)
    ])

    messages = [
        system_prompt,
        HumanMessage(content=f"""
Контекст:
{message_content}

История чата:
{context_with_history}

Вопрос пользователя:
{user_input}
""")
    ]

    llm = ChatOllama(model="bambucha/saiga-llama3:8b", temperature=0)

    def generate():
        response_text = ""
        for chunk in llm.stream(messages):
            token = chunk.content
            response_text += token
            yield token

        if ask_for_contact:
            contact_request = "\n\nЧтобы я могла передать ваш запрос коллеге, оставьте, пожалуйста, контакт (телефон или email)."
            response_text += contact_request
            yield contact_request

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response_text))

    return Response(generate(), mimetype="text/plain")

if __name__ == "__main__":
    app.run(debug=True, port=5000)