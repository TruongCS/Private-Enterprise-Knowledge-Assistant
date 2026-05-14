# agent.py
import re
import sqlite3
import pandas as pd
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain.chat_models import init_chat_model
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from config import (
    DB_PATH, VECTORSTORE_PATH, REPORT_PATH,
    EMBEDDING_MODEL, LLM_MODEL,
    CHUNK_SIZE, CHUNK_OVERLAP, OPENAI_API_KEY
)
import os

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ── Retriever ────────────────────────────────────────────────

def load_retriever():
    embedding   = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(
        Path(VECTORSTORE_PATH), embedding, allow_dangerous_deserialization=True
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = list(vectorstore.docstore._dict.values())

    if not docs:
        with open(REPORT_PATH, "r", encoding="utf-8") as f:
            text = re.sub(r'((?:\|.+\n)+)', '', f.read())
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        docs = [Document(page_content=c) for c in splitter.split_text(text)]
    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = 3

    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[0.7, 0.3]
    )

hybrid_retriever = load_retriever()


# ── Tools ────────────────────────────────────────────────────

@tool
def retrieve_financial_context(query: str) -> str:
    """Search the Uber annual report for narrative information —
    strategy, risks, business model, qualitative descriptions."""
    results = hybrid_retriever.invoke(query)
    if not results:
        return "No relevant content found."
    return "\n\n---\n\n".join(doc.page_content for doc in results)


@tool
def list_available_tables(_: str = "") -> str:
    """List all financial tables available in the database."""
    con = sqlite3.connect(DB_PATH)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", con
    )
    con.close()
    return "\n".join(tables['name'].tolist())


@tool
def search_tables_for_keyword(keyword: str) -> str:
    """Search all tables for ones containing a keyword in column names
    or data. Use before query_financial_table when you don't know the
    table name. Search for: 'revenue', 'income', 'ebitda', 'segment'."""
    con = sqlite3.connect(DB_PATH)
    tables = pd.read_sql_query(
        "SELECT name FROM sqlite_master WHERE type='table'", con
    )
    matches = []
    for name in tables['name']:
        try:
            df = pd.read_sql_query(f"SELECT * FROM '{name}' LIMIT 2", con)
            combined = " ".join(
                df.columns.tolist() +
                df.astype(str).values.flatten().tolist()
            ).lower()
            if keyword.lower() in combined:
                matches.append(f"{name}: columns = {df.columns.tolist()}")
        except:
            continue
    con.close()
    return "\n".join(matches) if matches else f"No tables found containing '{keyword}'"


@tool
def query_financial_table(sql: str) -> str:
    """Run a SELECT query against Uber's financial tables.
    Always inspect columns first with SELECT * FROM table LIMIT 3."""
    if re.search(r'\b(insert|update|delete|drop|alter|create)\b',
                 sql, re.IGNORECASE):
        return "Only SELECT queries are allowed."
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(sql, con)
        return "No rows returned." if df.empty else df.to_markdown(index=False)
    except Exception as e:
        return f"SQL error: {e}"
    finally:
        con.close()


@tool
def calculate(expression: str) -> str:
    """Evaluate arithmetic: growth rates, ratios, totals.
    Only use this to derive NEW numbers — not to restate retrieved ones.
    Example: '(43978 - 37281) / 37281 * 100'"""
    if not re.match(r'^[\d\s\.\+\-\*\/\(\)\%]+$', expression):
        return "Invalid expression."
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(round(result, 4))
    except Exception as e:
        return f"Calculation error: {e}"


tools = [
    retrieve_financial_context,
    list_available_tables,
    search_tables_for_keyword,
    query_financial_table,
    calculate,
]


# ── Agent ────────────────────────────────────────────────────

system_prompt = """
You are a precise financial analyst assistant specialising in Uber's annual report.

You have five tools — use the right one for each job:

- `list_available_tables`: see all table names (they are generic: table_0, table_1...).
- `search_tables_for_keyword`: use this FIRST for any numerical question.
  Search for 'revenue', 'income', 'ebitda', 'segment', 'cash' etc.
- `query_financial_table`: once you know the table, query it with SELECT SQL.
  Always run SELECT * FROM table LIMIT 3 first to inspect the schema.
- `retrieve_financial_context`: for narrative questions only — strategy, risks,
  business model. Not for specific numbers.
- `calculate`: ONLY to derive new numbers (growth rates, ratios).
  Never use it to restate a number already in the query result.

Workflow for numerical questions:
1. search_tables_for_keyword → find the right table
2. SELECT * LIMIT 3 → inspect columns
3. Precise SELECT → get the answer
4. calculate → only if a new number needs to be derived
5. Report exact figures from the data. Never approximate or invent numbers.
""".strip()

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

llm   = init_chat_model(LLM_MODEL, temperature=0.2)
agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True,
)


# ── Memory helpers ───────────────────────────────────────────

def ask(question: str, chat_history: list) -> tuple[str, list]:
    """Run a question through the agent, returns (answer, updated_history)."""
    response = agent_executor.invoke({
        "input": question,
        "chat_history": chat_history,
    })
    answer = response["output"]
    steps  = response.get("intermediate_steps", [])
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    return answer, chat_history, steps