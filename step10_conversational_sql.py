
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import URL

load_dotenv()

# 1. Initialize Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# 2. Connect to PostgreSQL
connection_url = URL.create(
    drivername="postgresql+psycopg",
    username="optimon_user",
    password="Optimon@2026",
    host="127.0.0.1",
    port=5433,
    database="optimonai-tenantdb-demo"
)

db = SQLDatabase.from_uri(
    connection_url,
    include_tables=["plc_signals", "furnace_signals","tags","tag_data_latest"]
)

schema = db.get_table_info()

# 3. SQL generation prompt with conversation history
sql_prompt = ChatPromptTemplate.from_template("""
You are a PostgreSQL SQL expert.

Database schema:
{schema}

Previous conversation:
{history}

Current user question:
{question}

Instructions:
- Generate one PostgreSQL SELECT query.
- Use only tables and columns in the supplied schema.
- Resolve follow-up questions using the conversation history.
- If the question is ambiguous, ask for clarification rather
  than guessing.
- Return only the SQL query, without Markdown.
""")

sql_chain = sql_prompt | llm

# 4. Answer generation prompt
answer_prompt = ChatPromptTemplate.from_template("""
User question: {question}

SQL executed:
{sql}

Database result:
{result}

Give a clear, concise answer based only on the result.
""")

answer_chain = answer_prompt | llm

# 5. Conversation history
history = []

def get_text(response):
    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict)
            else str(item)
            for item in content
        ).strip()

    return str(content).strip()


print("Conversational SQL is ready.")
print("Type 'exit' to stop.\n")

while True:
    question = input("You: ").strip()

    if question.lower() in ("exit", "quit"):
        print("Conversation ended.")
        break

    if not question:
        continue

    # 6. Pass previous conversation to Gemini
    history_text = "\n".join(
        f"User: {q}\nAssistant: {a}"
        for q, a in history
    )

    response = sql_chain.invoke({
        "schema": schema,
        "history": history_text or "No previous conversation.",
        "question": question
    })

    sql = get_text(response)

    # 7. Execute SQL
    print("\nGenerated SQL:")
    print(sql)

    try:
        # Basic demonstration check only.
        # Use a read-only database account for protection.
        if not sql.upper().lstrip().startswith("SELECT"):
            raise ValueError("Only SELECT queries are allowed.")

        result = db.run(sql)

        print("\nQuery Result:")
        print(result)

        # 8. Generate natural-language answer
        answer_response = answer_chain.invoke({
            "question": question,
            "sql": sql,
            "result": result
        })

        answer = get_text(answer_response)

        print("\nAssistant:", answer)

        # 9. Remember this interaction for follow-up questions
        history.append((question, answer))

    except Exception as e:
        print("\nError:", e)

    print("\n" + "-" * 50 + "\n")
