import streamlit as st
import pandas as pd
import openai
import io  # For capturing printed output
from contextlib import redirect_stdout  # Corrected import
import plotly.express as px  # Plotly for graphing

# Define the function to query OpenAI
def query_openai(prompt):
    """
    Send a query to OpenAI and get Python code as a response.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4o",  # or use "gpt-3.5-turbo" / "gpt-4" if needed
        messages=[
            {"role": "system", "content": "You are a helpful data assistant that generates Python code to process data."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()

# Streamlit app starts here
st.title("M1 Dynamic Data Bot")
st.write("Interact with your M1 data dynamically!")

# User Input for OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key:", type="password")
if api_key:
    openai.api_key = api_key  # Set the OpenAI API key

    # File uploader for CSV file
    uploaded_file = st.file_uploader("Upload your CSV file (e.g., m1_data.csv)", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:", df.head())

        # User Input for basic queries
        query = st.text_input("Enter your query (e.g., total order amount and cashback)")

        if st.button("Submit"):
            if query:
                st.write(f"Your query: **{query}**")

                # Use OpenAI to generate Python code based on the user query
                openai_prompt = f"""
                You are a helpful assistant that generates Python code for data analysis. The dataset contains the following columns:
                Conversion ID, Advertiser, Status, Order ID, Conversion Time, Click Time, Currency, 
                EUID, Estimate Order Value, Estimate Optimise Cashback Value, Cashback % or Amount, 
                Estimate M1 Cashback Value, Estimate Cashback Value (Exclude GST).

                The relevant columns for specific terms are as follows:
                - "cashback", "total cashback", or similar terms refer to "Estimate Optimise Cashback Value".
                - "order amount", "total order amount", or similar terms refer to "Estimate Order Value".

                The 'Conversion Time' column uses the format `dd/mm/yyyy hh:mm`. 
                Please ensure that the 'Conversion Time' column is parsed with the correct format (`%d/%m/%Y %H:%M`), and handle any errors during the parsing process by using the `errors='coerce'` option so that any invalid date values are converted to `NaT`.
                If a column uses the to_period() function (e.g., for months), ensure that the result is converted to a string using .astype(str) to avoid serialization errors when plotting with Plotly or processing JSON data. 
                The user has requested the following:
                {query}

                Generate Python code that can process this request and provide the answer.
                The Python code should use the pandas library to process the dataset. If the query involves generating a graph or plot, use Plotly for visualization,
                and ensure to display the plot using st.plotly_chart(fig) for integration with a Streamlit app instead of using fig.show().
                Return only the Python code, no explanations or extra text.
                Please generate Python code that can process this request and provide the answer. Do **not include** any markdown or code block formatting (` `````` `). Just give the Python code directly.
                my dataset file name is: m1_data.csv
                """
                
                python_code = query_openai(openai_prompt)

                st.write("Generated Python Code:")
                st.code(python_code)

                # Now, execute the generated Python code dynamically (using `exec`)
                try:
                    # Prepare a string buffer to capture printed output
                    output_buffer = io.StringIO()
                    
                    # Define a dictionary for the code execution environment
                    exec_globals = {"df": df, "pd": pd, "px": px, "st": st}

                    # Execute the Python code in the given environment, redirecting stdout
                    with redirect_stdout(output_buffer):
                        exec(python_code, exec_globals)
                    
                    # Retrieve the printed output
                    printed_output = output_buffer.getvalue().strip()

                    if printed_output:
                        # Try to evaluate if the output is a DataFrame (can be interpreted as Python code)
                        try:
                            output_data = eval(printed_output)
                            
                            # If it's a DataFrame, show it in an interactive table
                            if isinstance(output_data, pd.DataFrame):
                                st.write("Query Result (DataFrame):")
                                st.dataframe(output_data)
                            
                            # If it's a List or Dictionary, display in JSON format
                            elif isinstance(output_data, (list, dict)):
                                st.write("Query Result (List/Dict):")
                                st.json(output_data)
                            
                            # For graphs (if Plotly code was generated), display the graph
                            elif isinstance(output_data, (px.scatter, px.bar, px.line)):  # If it's a Plotly figure
                                st.write("Query Result (Graph):")
                                st.plotly_chart(output_data)
                            
                            # For other complex objects, display as text
                            else:
                                st.write(f"Query Result: {output_data}")
                        
                        except Exception as e:
                            # If eval() fails, treat the result as plain text
                            st.markdown("### Plain Text Result:")
                            st.text(printed_output)
                    
                    else:
                        st.warning("No result generated. Please check the query.")
                        
                except SyntaxError as e:
                    st.error(f"Syntax Error in generated code: {e}")
                except Exception as e:
                    st.error(f"An error occurred while executing the code: {e}")

        # Add Deep Dive Button
        if st.button("Deep Dive"):
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

            llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-4o",
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
            
                        
            
            
            query_deep_dive = st.text_input("Enter your deep dive query:")
            if st.button("Run Deep Dive Query"):
                 inputs_deep_dive = {"query": query_deep_dive}
                 try:
                     result_deep_dive = crew.kickoff(inputs=inputs_deep_dive)
            
                     st.write("### Deep Dive Result")
                     if isinstance(result_deep_dive, dict):
                         st.json(result_deep_dive)
                     elif isinstance(result_deep_dive, str):
                         st.markdown(result_deep_dive)
                     else:
                         st.write(result_deep_dive)
                 except Exception as e:
                     st.error(f"An error occurred during deep dive analysis: {e}")
