from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import os

def configure():
    load_dotenv()
    return os.getenv('OPENAI_API_KEY')

model = ChatOpenAI(
    model="gpt-4",
    api_key=configure(),
)

persist_directory = 'db'
embedding = OpenAIEmbeddings()

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding,
    collection_name='laptop_manual',
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

def format_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

system_prompt = '''
    Use o contexto para responder as perguntas.
    Contexto: {context}
'''

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}'),
    ]
)

question_answer_chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt,
)

chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

try:
    while True:
        query = input('Qual sua duvida? ')
        
        response = chain.invoke(
            {'input': query},
        )
        
        print(response.get('answer'))
        print('\n')
        
except KeyboardInterrupt:
    exit()