import os
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

pinecone.init(
    api_key="e1f6a543-85e7-4fe6-bfb3-c2f9db496263",
    environment="us-west1-gcp-free",
)


def ingest_docs() -> None:
    loader = ReadTheDocsLoader(
        path="langchain-docs/python.langchain.com/en/latest", encoding="utf8"
    )
    raw_documents = loader.load()
    print(f"loaded {len(raw_documents) }documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to insert {len(documents)} to Pinecone")
    embeddings = OpenAIEmbeddings(
        openai_api_key="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"
    )
    Pinecone.from_documents(documents[3969:], embeddings, index_name="langchain-doc")
    print("****** Added to Pinecone vectorstore vectors")


if __name__ == "__main__":
    ingest_docs()
