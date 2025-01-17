import os
import json
import pysqlite3 as sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from textwrap import dedent
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from crewai import Agent, Crew, Process, Task
from crewai_tools import tool
from langchain.schema.output import LLMResult
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI  # Importing OpenAI model

# Streamlit page config
st.set_page_config(
    page_title="Pulse ID - AI Data Assistant",
    page_icon="logo.png",
    layout="wide",
)

# Display logo and title
st.image("logo.png", width=150)
st.title("Pulse ID - AI Data Assistant")

# Input API key for OpenAI
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
os.environ["OPENAI_API_KEY"] = api_key  # Set the environment variable for OpenAI

# Upload CSV file and create SQLite database
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:", df.head())
    connection = sqlite3.connect("salaries.db")
    df.to_sql(name="salaries", con=connection, if_exists="replace", index=False)
    db = SQLDatabase.from_uri("sqlite:///salaries.db")

    @dataclass
    class Event:
        event: str
        timestamp: str
        text: str

    def _current_time() -> str:
        return datetime.now(timezone.utc).isoformat()

    class LLMCallbackHandler(BaseCallbackHandler):
        def __init__(self, log_path: Path):
            self.log_path = log_path

        def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
            event = Event(event="llm_start", timestamp=_current_time(), text=prompts[0])
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(event)) + "\n")

        def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
            generation = response.generations[-1][-1].message.content
            event = Event(event="llm_end", timestamp=_current_time(), text=generation)
            with self.log_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(event)) + "\n")

    # Initialize OpenAI model instead of ChatGroq
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o",  # Specify the model you want to use (e.g., gpt-4)
        callbacks=[LLMCallbackHandler(Path("prompts.jsonl"))],
    )

    @tool("list_tables")
    def list_tables() -> str:
        """List all tables in the database."""
        return ListSQLDatabaseTool(db=db).invoke("")

    @tool("tables_schema")
    def tables_schema(tables: str) -> str:
        """Retrieve the schema of specified tables."""
        tool = InfoSQLDatabaseTool(db=db)
        return tool.invoke(tables)

    @tool("execute_sql")
    def execute_sql(sql_query: str) -> str:
        """Execute the provided SQL query and return results."""
        return QuerySQLDataBaseTool(db=db).invoke(sql_query)

    @tool("check_sql")
    def check_sql(sql_query: str) -> str:
        """Check the provided SQL query for correctness."""
        return QuerySQLCheckerTool(db=db, llm=llm).invoke({"query": sql_query})

    sql_dev = Agent(
        role="Senior Database Developer",
        goal="Construct and execute SQL queries based on a request",
        backstory=dedent(
            """
            You are an experienced database engineer who is a master at creating efficient and complex SQL queries.
            Use the `list_tables` to find available tables.
            Use the `tables_schema` to understand the metadata for the tables.
            Use the `execute_sql` to execute queries against the database.
            Use the `check_sql` to check queries for correctness before execution.
            """
        ),
        llm=llm,
        tools=[list_tables, tables_schema, execute_sql, check_sql],
        allow_delegation=False,
    )

    data_analyst = Agent(
        role="Senior Data Analyst",
        goal="Analyze data from the database",
        backstory=dedent(
            """
            You have deep experience with analyzing datasets using Python. Your analyses are detailed and insightful.
            """
        ),
        llm=llm,
        allow_delegation=False,
    )

    report_writer = Agent(
        role="Senior Report Editor",
        goal="Write executive summaries based on analysis",
        backstory=dedent(
            """
            You are known for creating concise and effective executive summaries.
            """
        ),
        llm=llm,
        allow_delegation=False,
    )

    extract_data = Task(
        description="Extract data required for the query {query}.",
        expected_output="Database result for the query",
        agent=sql_dev,
    )

    analyze_data = Task(
        description="Analyze the extracted data for {query}.",
        expected_output="Detailed analysis text/plots",
        agent=data_analyst,
        context=[extract_data],
    )

    write_report = Task(
        description="Write an executive summary of the analysis.",
        expected_output="Markdown report text",
        agent=report_writer,
        context=[analyze_data],
    )

    crew = Crew(
        agents=[sql_dev, data_analyst, report_writer],
        tasks=[extract_data, analyze_data, write_report],
        process=Process.sequential,
        verbose=2,
        memory=False,
        output_log_file="crew.log",
    )

    query = st.text_input("Enter your query:")
    if st.button("Run Query"):
        inputs = {"query": query}
        try:
            result = crew.kickoff(inputs=inputs)

            st.write("### Result")
            if isinstance(result, dict):
                st.json(result)
            elif isinstance(result, str):
                st.markdown(result)
            else:
                st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
