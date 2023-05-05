import os
from typing import Any, Dict, List

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone


pinecone.init(
    api_key="e1f6a543-85e7-4fe6-bfb3-c2f9db496263",
    environment="us-west1-gcp-free",
)

INDEX_NAME = "langchain-doc"


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings(
        openai_api_key="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"
    )
    docsearch = Pinecone.from_existing_index(
        embedding=embeddings,
        index_name=INDEX_NAME,
    )
    chat = ChatOpenAI(
        openai_api_key="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",
        verbose=True,
        temperature=0,
    )

    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )

    print(query)

    return qa({"query": query})
