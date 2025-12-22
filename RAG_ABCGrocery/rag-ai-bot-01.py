########################################################################################################################
# 01 - SECURELY PULL IN API KEYS & GIVE LANGCHAIN PROJECT NAME
########################################################################################################################

from dotenv import load_dotenv
loaded_in = load_dotenv(dotenv_path="/Users/eacalder/Documents/Github/DSI_work/RAG_ABCGrocery/GenAI/RAG/.env")

print("API keys loaded:", loaded_in)

########################################################################################################################
# 02 - LOAD DOCUMENT
########################################################################################################################

from langchain_community.document_loaders import TextLoader

raw_filename = 'abc-grocery-help-desk-data.md'
loader = TextLoader(raw_filename, encoding="utf-8")
docs = loader.load()
print(docs)
text = docs[0].page_content
print(len(text))
print(text)