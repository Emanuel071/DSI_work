##################################################################################################
# 01 - Bring in .env information
##################################################################################################

import os
from dotenv import load_dotenv
load_dotenv()

##################################################################################################
# 02 - Create the connection string for the postgres database
##################################################################################################

POSTGRES_URI = (f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
                f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/{os.getenv('POSTGRES_DBNAME')}?sslmode=require")
##################################################################################################

##################################################################################################
# 03 - Create the database engine
##################################################################################################

import sqlalchemy as sa

# create the database engine
engine = sa.create_engine(POSTGRES_URI,
                          pool_pre_ping=True, # check the connection before using it
                          connect_args={"options": "-c statement_timeout=15000"} # 15 second timeout
                          ) 

# # check the connection
# queries database, not needed to run every time, just to check the connection is working
# with engine.connect() as conn:
#     conn.exec_driver_sql("select 1")

##################################################################################################
# 04 - Setup the database connection
##################################################################################################

from langchain_community.utilities import SQLDatabase

db = SQLDatabase(engine=engine, # pass in the engine
                 schema="grocery_db", # specify the schema
                 include_tables=["customer_details", "transactions"], # specify the tables to include
                 sample_rows_in_table_info=5) # specify the number of rows to sample for table information

print("Usable tables:", db.get_usable_table_names())

##################################################################################################
# 05 - Create our SQL AI Agent
##################################################################################################

from langchain_openai import ChatOpenAI

sql_agent = ChatOpenAI(model="gpt-4.1",
                       temperature=0)

##################################################################################################
# 06 - Build the SQL Toolkit and tools
##################################################################################################

from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=sql_agent)
tools = toolkit.get_tools()

##################################################################################################
# 07 - Bring In System Prompt
##################################################################################################

# bring in the system instructions
with open("sql-agent-system-prompt.txt", "r", encoding="utf-8") as f:
    system_text = f.read()

##################################################################################################
# 08 - Create the Agent
##################################################################################################

from langchain.agents import create_agent

agent = create_agent(model=sql_agent, # specify the LLM
                     tools=tools, # specify the tools
                     system_prompt=system_text) # specify the system prompt


##################################################################################################
# 09 - Run test queries through the agent and extract the response
##################################################################################################

from langchain_core.messages import HumanMessage

# user_query = "Of average, which gender Lives furthest from the store?"
# user_query = "What is the average transaction value in September 2020 for male customers who have a credit score above 0.5?"
user_query = ""

result = agent.invoke({"messages": [HumanMessage(content=user_query)]})
print(result["messages"][-1].content)
