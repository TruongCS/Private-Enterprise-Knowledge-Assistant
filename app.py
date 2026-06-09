# app.py
import streamlit as st
from agent import ask

st.set_page_config(
    page_title="Uber Financial Agent",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Uber Financial Agent")
st.caption("Ask questions about Uber's 2025 Annual Report")

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Example questions
with st.sidebar:
    st.header("Example questions")
    examples = [
        "What was Uber's total revenue in 2024?",
        "Which segment grew the fastest YoY?",
        "What is Uber's net income in 2024 vs 2023?",
        "What are the main risks Uber faces?",
        "What was the Mobility segment revenue in 2024?",
    ]
    for example in examples:
        if st.button(example, use_container_width=True):
            st.session_state.pending_question = example

    if st.button("🗑 Reset conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.chat_history = []
        st.rerun()

# Handle example button clicks
question = st.chat_input("Ask a financial question...")
if not question and "pending_question" in st.session_state:
    question = st.session_state.pop("pending_question")

# Process question
# app.py - update the question processing block
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Analysing..."):
            answer, st.session_state.chat_history, steps = ask(
                question, st.session_state.chat_history
            )

        # Show reasoning steps
        if steps:
            with st.expander("🔍 View reasoning steps"):
                for i, (action, observation) in enumerate(steps, 1):
                    st.markdown(f"**Step {i} — Tool: `{action.tool}`**")
                    st.code(str(action.tool_input), language="json")
                    st.markdown("**Result:**")
                    st.code(str(observation)[:1000])
                    st.divider()

        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})