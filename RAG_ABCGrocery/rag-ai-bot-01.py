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
print("chunked_docs[0]")
print(chunked_docs[0])
print("chunked_docs[0].page_content")
print(chunked_docs[0].page_content)

########################################################################################################################
# 04 - TURN EACH CHUNK INTO AN EMBEDDING VECTOR & STORE IN VECTOR DB
########################################################################################################################

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# # create the embeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# # create vector database
# vectorstore = Chroma.from_documents(documents=chunked_docs,
#                                     embedding=embeddings,
#                                     collection_metadata={"hnsw:space": "cosine"},
#                                     persist_directory="abc_vector_db_chroma",
#                                     collection_name="abc_help_qa")


# code to load DB once saved


from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="abc_vector_db_chroma",
                     collection_name="abc_help_qa",
                     embedding_function=embeddings)


########################################################################################################################
# 05 - TEST RETRIEVAL
########################################################################################################################

query = ("What hours are you open on Easter?")

# 01. best k chunks based on similarity distance
top_docs_regular = vectorstore.similarity_search(query, k=4)

# # print the chunks
# print("\n\ntop_docs_regular:")
# for i in top_docs_regular:
#     print(i.metadata)
#     print(i.page_content)

# 02. best k chunks based on similarity distance with distance score provided (lower = a better match)
top_docs_distance = vectorstore.similarity_search_with_score(query, k=4)

# # print the chunks
# print("\n\ntop_docs_distance:")
# for i in top_docs_distance:
#     # with lower scores being more relevant
#     print(i[0].metadata)
#     print(i[1])
#     print(i[0].page_content)

# 03. best k chunks based on similarity distance with relevance (normalised similarity) score provided (higher = a better match)
top_docs_relevance = vectorstore.similarity_search_with_relevance_scores(query, k=4, score_threshold=0.40)

# print the chunks
print("\n\ntop_docs_relevance:")
for i in top_docs_relevance:
    print(i[0].metadata)
    print(i[1])
    print(i[0].page_content)

