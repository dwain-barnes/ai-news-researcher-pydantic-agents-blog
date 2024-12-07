import os
import asyncio
import datetime
from typing import Any
from dataclasses import dataclass

import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient
from devtools import debug

# Ensure you have your environment variables set
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", None)

if not OPENAI_API_KEY:
    raise ValueError("Please set OPENAI_API_KEY environment variable.")
if not TAVILY_API_KEY:
    raise ValueError("Please set TAVILY_API_KEY environment variable.")

tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

@dataclass
class SearchDataclass:
    max_results: int
    todays_date: str

@dataclass
class ResearchDependencies:
    todays_date: str

class ResearchResult(BaseModel):
    research_title: str = Field(description='Markdown heading describing the article topic, prefixed with #')
    research_main: str = Field(description='A main section that provides a detailed news article')
    research_bullets: str = Field(description='A set of bullet points summarizing key points')

# Create the agent
search_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=ResearchDependencies,
    result_type=ResearchResult,
    system_prompt='You are a helpful research assistant, you are an expert in research. '
                  'When given a query, you will identify strong keywords to do 3-5 searches using the provided search tool. '
                  'Then combine results into a detailed response.'
)

@search_agent.system_prompt
async def add_current_date(ctx: RunContext[ResearchDependencies]) -> str:
    todays_date = ctx.deps.todays_date
    system_prompt = (
        f"You're a helpful research assistant and an expert in research. "
        f"When given a question, write strong keywords to do 3-5 searches in total "
        f"(each with a query_number) and then combine the results. "
        f"If you need today's date it is {todays_date}. "
        f"Focus on providing accurate and current information."
    )
    return system_prompt

@search_agent.tool
async def get_search(search_data: RunContext[SearchDataclass], query: str, query_number: int) -> dict[str, Any]:
    """Perform a search using the Tavily client."""
    max_results = search_data.deps.max_results
    results = await tavily_client.get_search_context(query=query, max_results=max_results)
    return results

async def do_search(query: str, max_results: int):
    # Prepare dependencies
    current_date = datetime.date.today()
    date_string = current_date.strftime("%Y-%m-%d")
    deps = SearchDataclass(max_results=max_results, todays_date=date_string)
    result = await search_agent.run(query, deps=deps)
    return result.data


# ---- Streamlit UI ----

st.set_page_config(page_title="AI News Researcher", layout="centered")

st.title("AI Security News Researcher")
st.write("Stay updated on the latest trends and developments in AI Security.")

# User input section
st.sidebar.title("Search Parameters")
query = st.sidebar.text_input("Enter your query:", value="latest AI security news")
max_results = st.sidebar.slider("Number of search results:", min_value=3, max_value=10, value=5)

st.write("Use the sidebar to adjust search parameters.")

if st.button("Get Latest AI Security News"):
    with st.spinner("Researching, please wait..."):
        result_data = asyncio.run(do_search(query, max_results))

    st.markdown(result_data.research_title)
    # A bit of styling for the main article
    st.markdown(f"<div style='line-height:1.6;'>{result_data.research_main}</div>", unsafe_allow_html=True)

    st.markdown("### Key Takeaways")
    st.markdown(result_data.research_bullets)
