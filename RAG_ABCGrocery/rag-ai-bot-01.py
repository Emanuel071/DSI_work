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
# print(docs)
text = docs[0].page_content
print(len(text))
# print(text)

########################################################################################################################
# 03 - SPLIT DOCUMENT INTO CHUNKS
########################################################################################################################

from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("###", "id")],
    strip_headers=True)

chunked_docs = splitter.split_text(text)
print(len(chunked_docs), "Q/A chunks")
print(chunked_docs[0])
print(chunked_docs[0].page_content)