import re
import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

#ignoring warnings
import warnings
warnings.filterwarnings('ignore')

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_fn = HuggingFaceEmbeddings(model_name=model_name)

api_key = "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"

#Main model client
chat = ChatGroq(temperature = 1,
                groq_api_key = api_key,
                model_name = "mixtral-8x7b-32768")


#Load and read the pdf document
print("Loading and Reading the pdf file...")
pdf_file_path = "D:/data science/GenAI course/data/IPCC_AR6_WGII_TechnicalSummary.pdf"
reader = PdfReader(pdf_file_path)
pdf_texts = [page.extract_text() for page in reader.pages]


#Filter out some pages from begining and the end
texts_filt = pdf_texts[5:-5]

#Cleaning the pdf file
print("Cleaning the pdf file...")
pdf_wo_head_foot = [re.sub(r'\d+\nTechnical Summary', '', s) for s in texts_filt]
pdf_wo_head_foot = [re.sub(r'\nTS', '', s) for s in pdf_wo_head_foot]
pdf_wo_head_foot = [re.sub(r'TS\n', '', s) for s in pdf_wo_head_foot]

#Spliting the text by characters
print('spliting the text by Char and tokens...')
char_splitter = RecursiveCharacterTextSplitter(
    separators = ['\n', '\n\n', ' ', '. ', ', ', ''],
    chunk_size = 1000,
    chunk_overlap = 0.2
)

text_char_split = char_splitter.split_text("\n\n".join(pdf_wo_head_foot))

#Spliting the text by Tokens
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap = 0.2,
    tokens_per_chunk = 256
)

texts_token_splitted = []
for text in text_char_split:
    try:
        texts_token_splitted.extend(token_splitter.split_text(text))
    except:
        pass

documents = [Document(page_content=text) for text in texts_token_splitted]

vector_path = 'D:/data science/GenAI course/Climate ChatBot/climate_Vector_db'
vector_db = FAISS.from_documents(documents=documents, embedding=embedding_fn)
vector_db.save_local(vector_path)

loaded_db = FAISS.load_local(
    vector_path,
    embeddings=embedding_fn,
    allow_dangerous_deserialization=True
)

# Create memory
memory = ConversationBufferWindowMemory(
    k=2,
    memory_key="chat_history",
    output_key="answer",
    return_messages=True,
)

# Create the ConversationalRetrievalChain
qa_conversation = ConversationalRetrievalChain.from_llm(
    llm=chat,
    chain_type="stuff",
    retriever=loaded_db.as_retriever(),
    return_source_documents=True,
    memory=memory
)

def RAG_Chain(query):
    print("Generating the response...")
    # Pass the query with the key 'question'
    response = qa_conversation({"question": query})
    return response.get("answer")

while(1):
    query = input("Enter the query : ")
    if query == "1":
        print("Thank You")
        break
    response = RAG_Chain(query)
    print(response)
    print()
    print("Hope it is helpful for you...")
    print("Enter 1 if you don't want to continue or ask the queries.")
    print()