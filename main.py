import os
import argparse
import pandas as pd
import kagglehub
from transformers import pipeline, logging as hf_logging

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


def enable_verbose(verbose: bool) -> None:
    print("==== STARTED: Verbose configuration ====")

    if verbose:
        hf_logging.set_verbosity_debug()
        os.environ["LANGCHAIN_DEBUG"] = "true"
        print("[VERBOSE] Transformers logging set to DEBUG.")
        print("[VERBOSE] LANGCHAIN_DEBUG enabled.")
    else:
        hf_logging.set_verbosity_info()

    print("---- COMPLETED SUCCESSFULLY: Verbose configuration ----")


def download_dataset() -> str:
    print("==== STARTED: Downloading Kaggle dataset (gpreda/bbc-news) ====")

    path = kagglehub.dataset_download("gpreda/bbc-news")
    
    print(f"---- COMPLETED SUCCESSFULLY: Dataset downloaded to: {path} ----")
    return path


def load_dataframe(dataset_path: str) -> pd.DataFrame:
    print("==== STARTED: Loading CSV into DataFrame ====")

    csv_path = os.path.join(dataset_path, "bbc_news.csv")
    df = pd.read_csv(csv_path)
    
    print(f"---- COMPLETED SUCCESSFULLY: DataFrame loaded with {len(df)} rows ----")
    return df


def build_documents(df: pd.DataFrame) -> list:
    print("==== STARTED: Converting rows to LangChain Documents ====")
    
    documents = []
    for _, row in df.iterrows():
        content = f"""
        Title: {row['title']}
        Description: {row['description']}
        """
        metadata = {
            "source": row.get('link', ''),
            "published": row.get('pubDate', ''),
            "guid": row.get('guid', ''),
        }
        documents.append(Document(page_content=content, metadata=metadata))

    print(f"---- COMPLETED SUCCESSFULLY: Created {len(documents)} documents ----")
    return documents


def split_into_chunks(documents: list, chunk_size: int = 400, chunk_overlap: int = 50) -> list:
    print("==== STARTED: Splitting documents into chunks ====")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)

    print(f"---- COMPLETED SUCCESSFULLY: Created {len(chunks)} chunks ----")
    return chunks


def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    print("==== STARTED: Loading embeddings model ====")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print(f"---- COMPLETED SUCCESSFULLY: Embeddings model '{model_name}' loaded ----")
    return embeddings


def build_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings, persist_directory: str = "./chroma_bbc") -> Chroma:
    print("==== STARTED: Building Chroma vectorstore ====")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
    )
    
    # Ensure data is persisted to disk
    try:
        vectorstore.persist()
    except Exception:
        pass

    print(f"---- COMPLETED SUCCESSFULLY: Vectorstore built at '{persist_directory}' ----")
    return vectorstore


def build_llm(model: str = "google/flan-t5-base", max_new_tokens: int = 256) -> HuggingFacePipeline:
    print("==== STARTED: Initializing HuggingFace text generation pipeline ====")

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        max_new_tokens=max_new_tokens,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    print(f"---- COMPLETED SUCCESSFULLY: LLM pipeline '{model}' initialized ----")
    return llm


def build_qa_chain(vectorstore: Chroma, llm: HuggingFacePipeline, k: int = 4) -> RetrievalQA:
    print("==== STARTED: Creating RetrievalQA chain ====")

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    print("---- COMPLETED SUCCESSFULLY: RetrievalQA chain is ready ----")
    return qa_chain


def run_query(qa_chain: RetrievalQA, query: str) -> dict:
    print("==== STARTED: Running query against RetrievalQA ====")

    result = qa_chain(query)

    print("---- COMPLETED SUCCESSFULLY: Query executed ----")
    return result


def print_result(result: dict) -> None:
    print("\n==================== ANSWER ====================")

    print(result.get("result", ""))

    print("\n==================== SOURCES ====================")

    for doc in result.get("source_documents", []):
        src = getattr(doc, "metadata", {}).get("source", "")
        print("-", src)


def main(query: str, verbose: bool) -> None:
    enable_verbose(verbose)

    dataset_path    = download_dataset()
    df              = load_dataframe(dataset_path)
    documents       = build_documents(df)
    chunks          = split_into_chunks(documents)
    embeddings      = load_embeddings()
    vectorstore     = build_vectorstore(chunks, embeddings, persist_directory="./chroma_bbc")
    llm             = build_llm()
    qa_chain        = build_qa_chain(vectorstore, llm)
    result          = run_query(qa_chain, query)

    print_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BBC News RAG")

    parser.add_argument(
        "--query",
        type=str,
        default="What recent news discusses developments in technology?",
        help="Question to ask the RAG chain",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    
    args = parser.parse_args()
    main(query=args.query, verbose=args.verbose)
