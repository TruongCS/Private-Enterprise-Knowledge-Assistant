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
import streamlit as st
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ── Retriever ────────────────────────────────────────────────
@st.cache_resource
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
    
    parts = []
    for doc in results:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    
    return "\n\n---\n\n".join(parts)


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
        except Exception as e:
            print(f"[search_tables_for_keyword] skipped table '{name}': {e}")
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
    except sqlite3.OperationalError as e:
        return f"SQL error: {e}. Check the table name and column names are correct."
    except Exception as e:
        return f"Unexpected error running query: {e}"
    finally:
        con.close()


@tool
def calculate(expression: str) -> str:
    """Evaluate arithmetic: growth rates, ratios, totals.
    Only use this to derive NEW numbers — not to restate retrieved ones.
    Example: '(43978 - 37281) / 37281 * 100'"""
    if not re.match(r'^[\d\s\.\+\-\*\/\(\)\%]+$', expression):
        return f"Invalid expression '{expression}'. Only use numbers and operators: + - * / ( ) %"
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(round(result, 4))
    except ZeroDivisionError:
        return "Calculation error: division by zero."
    except Exception as e:
        return f"Calculation error: {e}. Check the expression is valid arithmetic."
@tool
def lookup_registry(keyword: str) -> str:
    """Look up the _registry table to find which table contains data about
    a keyword. Always use this FIRST before search_tables_for_keyword.
    Search for: 'revenue', 'income', 'segment', 'ebitda', 'cash'."""
    con = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM '_registry'", con)
        keyword_lower = keyword.lower()
        matches = df[
            df.apply(lambda row: keyword_lower in row.astype(str).str.lower().values.any(), axis=1)
        ]
        if matches.empty:
            return f"No tables found in registry for '{keyword}'"
        return matches[["table_name", "heading", "columns", "sample"]].to_markdown(index=False)
    except Exception as e:
        return f"Registry lookup error: {e}"
    finally:
        con.close()

tools = [
    lookup_registry,              # ← add first so agent uses it first
    retrieve_financial_context,
    list_available_tables,
    search_tables_for_keyword,
    query_financial_table,
    calculate,
]

# ── Agent ────────────────────────────────────────────────────

system_prompt = """
You are a precise financial analyst assistant specialising in Uber's annual report.

You have six tools — use the right one for each job:

- `lookup_registry`: use this FIRST for any numerical question.
  It searches a registry table mapping table names to headings, columns and sample rows.
  Search for: 'revenue', 'income', 'ebitda', 'segment', 'cash', 'expense' etc.
- `search_tables_for_keyword`: fallback if lookup_registry finds nothing.
  Scans all table contents directly for a keyword match.
- `list_available_tables`: lists all table names in the database.
  Use only if both lookup_registry and search_tables_for_keyword return nothing.
- `query_financial_table`: once you know the table name, query it with SELECT SQL.
  Always run SELECT * FROM table LIMIT 3 first to inspect the schema before a precise query.
- `retrieve_financial_context`: for narrative questions ONLY — strategy, risks,
  business model, qualitative descriptions. Never use for specific numbers.
- `calculate`: ONLY to derive NEW numbers such as growth rates, ratios, and totals.
  Never use it to restate a number already returned by a query result.

Workflow for numerical questions:
1. lookup_registry → find the right table name, columns and a sample row
2. SELECT * LIMIT 3 → confirm the schema
3. Precise SELECT → get the exact answer
4. calculate → only if a new number needs to be derived from retrieved values
5. Report exact figures from the data. Never approximate or invent numbers.

Workflow for narrative questions:
1. retrieve_financial_context → search the report for relevant passages
2. Summarise the retrieved passages accurately
3. Never invent information not present in the retrieved context

When answering, always mention the source of your information at the end.
For numerical answers: (Source: table name, e.g. uber_report_consolidated_statements)
For narrative answers: (Source: uber_report.md)
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

def ask(question: str, chat_history: list) -> tuple[str, list, list]:
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