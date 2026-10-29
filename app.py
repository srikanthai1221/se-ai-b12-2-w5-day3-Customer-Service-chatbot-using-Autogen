# Block A - Imports and the Key
import asyncio
import os

import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Block B - the five agents
DEPARTMENTS = {
    "IT_Agent": {
        "label": "💻 IT",
        "description": "Handles IT queries: laptops, passwords, VPN, email, software installs, network, system access.",
        "system": "You are the IT Support agent of a company help desk. Answer IT questions clearly in short steps.",
    },
    "Admin_Agent": {
        "label": "🏢 Admin",
        "description": "Handles admin/facilities queries: office access cards, seating, meeting rooms, stationery, travel desk, transport, canteen.",
        "system": "You are the Admin/Facilities agent of a company help desk. Answer admin and facilities questions helpfully.",
    },
    "HR_Agent": {
        "label": "👥 HR",
        "description": "Handles HR queries: leave, holidays, onboarding, appraisals, benefits, policies, resignation, payroll letters.",
        "system": "You are the HR agent of a company help desk. Answer HR questions politely and clearly.",
    },
    "Finance_Agent": {
        "label": "💰 Finance",
        "description": "Handles finance queries: salary credit, reimbursements, expense claims, invoices, tax deductions, Form 16.",
        "system": "You are the Finance agent of a company help desk. Answer finance questions accurately and simply.",
    },
    "Legal_Agent": {
        "label": "⚖️ Legal",
        "description": "Handles legal queries: contracts, NDAs, compliance, data privacy, intellectual property, disputes.",
        "system": "You are the Legal agent of a company help desk. Give general guidance and suggest consulting the legal team for specific cases.",
    },
}

COMMON_RULES = (
    " Keep answers under 150 words. If the question is clearly outside your department, "
    "say so politely and name the department that can help."
)

# Block C - routing
SELECTOR_PROMPT = """You are the router of a company customer service desk.
Read the user's query and choose the ONE department agent best suited to answer it.

Available agents:
{roles}

Conversation:
{history}

Pick exactly one agent from {participants}. Reply with only the agent name."""

# Block D - ask a question to the chatbot
async def ask_team(query: str):
    model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

    agents = [
        AssistantAgent(
            name=name,
            model_client=model_client,
            description=info["description"],
            system_message=info["system"] + COMMON_RULES,
        )
        for name, info in DEPARTMENTS.items()
    ]

    team = SelectorGroupChat(
        participants=agents,
        model_client=model_client,
        selector_prompt=SELECTOR_PROMPT,
        termination_condition=MaxMessageTermination(2),  # user question + 1 agent reply
    )

    result = await team.run(task=query)
    await model_client.close()

    reply = result.messages[-1]          # last message = the chosen agent's answer
    return reply.source, reply.content   # e.g. ("HR_Agent", "You get ...")

# Block E - building the streamlit frontend
st.set_page_config(page_title="Customer Service Chatbot", page_icon="🤖")
st.title("🤖 Multi-Agent Customer Service Chatbot")
st.caption("Powered by AutoGen — your query is routed to IT, Admin, HR, Finance or Legal.")

with st.sidebar:
    st.header("Departments")
    for info in DEPARTMENTS.values():
        st.write(info["label"])
    if st.button("Clear chat"):
        st.session_state.messages = []

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Check your .env file.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask anything — IT, Admin, HR, Finance or Legal...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Routing your query to the right department..."):
            agent_name, answer = asyncio.run(ask_team(query))
        label = DEPARTMENTS.get(agent_name, {}).get("label", agent_name)
        text = f"**Answered by: {label}**\n\n{answer}"
        st.markdown(text)

    st.session_state.messages.append({"role": "assistant", "content": text})