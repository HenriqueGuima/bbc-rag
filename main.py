import os
import argparse
import pandas as pd
import kagglehub
import math
from transformers import pipeline, logging as hf_logging, AutoTokenizer
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def enable_verbose(verbose: bool) -> None:
    # print("==== STARTED: Verbose configuration ====")

    if verbose:
        hf_logging.set_verbosity_debug()
        os.environ["LANGCHAIN_DEBUG"] = "true"
        print("[VERBOSE] Transformers logging set to DEBUG.")
        print("[VERBOSE] LANGCHAIN_DEBUG enabled.")
    else:
        hf_logging.set_verbosity_info()

    # print("---- COMPLETED SUCCESSFULLY: Verbose configuration ----")


def clear_console() -> None:
    """Clear the console on Windows (cls) and Unix (clear)."""
    try:
        os.system('cls' if os.name == 'nt' else 'clear')
    except Exception:
        pass


def download_dataset() -> str:
    # print("==== STARTED: Downloading Kaggle dataset (gpreda/bbc-news) ====")

    path = kagglehub.dataset_download("gpreda/bbc-news")
    
    print(f"---- COMPLETED SUCCESSFULLY: Dataset downloaded to: {path} ----")
    return path


def load_dataframe(dataset_path: str) -> pd.DataFrame:
    # print("==== STARTED: Loading CSV into DataFrame ====")

    csv_path    = os.path.join(dataset_path, "bbc_news.csv")
    df          = pd.read_csv(csv_path)
    
    print(f"---- COMPLETED SUCCESSFULLY: DataFrame loaded with {len(df)} rows ----")
    return df


def build_documents(df: pd.DataFrame) -> list:
    # print("==== STARTED: Converting rows to LangChain Documents ====")
    
    documents = []
    for _, row in df.iterrows():

        content = (
            f"Title: {row['title']}\n"
            f"Description: {row['description']}\n"
            f"URL: {row['link']}"
        )

        metadata = {
            "published": row.get('pubDate', ''),
            "guid": row.get('guid', ''),
        }

        documents.append(Document(page_content=content, metadata=metadata))

    print(f"---- COMPLETED SUCCESSFULLY: Created {len(documents)} documents ----")
    return documents


def split_into_chunks(documents: list, chunk_size: int = 400, chunk_overlap: int = 50) -> list:
    # print("==== STARTED: Splitting documents into chunks ====")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)

    print(f"---- COMPLETED SUCCESSFULLY: Created {len(chunks)} chunks ----")
    return chunks


def load_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
    # print("==== STARTED: Loading embeddings model ====")

    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print(f"---- COMPLETED SUCCESSFULLY: Embeddings model '{model_name}' loaded ----")
    return embeddings


def build_vectorstore(chunks: list, embeddings: HuggingFaceEmbeddings, persist_directory: str = "./chroma_bbc", batch_size: int = 500) -> Chroma:
    # print("==== STARTED: Building Chroma vectorstore ====")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    total       = len(chunks)
    num_batches = math.ceil(total / batch_size)

    for i in tqdm(range(num_batches), desc="Embedding & inserting batches"):

        start   = i * batch_size
        end     = min(start + batch_size, total)
        batch   = chunks[start:end]

        vectorstore.add_documents(batch)

    # Ensure data is persisted to disk
    try:
        vectorstore.persist()
    except Exception:
        pass

    print(f"---- COMPLETED SUCCESSFULLY: Vectorstore built at '{persist_directory}' ----")
    return vectorstore


def vectorstore_available(persist_directory: str = "./chroma_bbc") -> bool:
    """Check if a persisted Chroma database exists at the directory."""
    db_path = os.path.join(persist_directory, "chroma.sqlite3")
    return os.path.exists(db_path)


def load_vectorstore(embeddings: HuggingFaceEmbeddings, persist_directory: str = "./chroma_bbc") -> Chroma:
    # print("==== STARTED: Loading existing Chroma vectorstore ====")

    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    print(f"---- COMPLETED SUCCESSFULLY: Loaded vectorstore from '{persist_directory}' ----")
    return vectorstore


def build_llm(
    model: str = "google/flan-t5-base",
    max_new_tokens: int = 512,
    num_beams: int = 4,
    do_sample: bool = False,
    input_max_length: int = 512,
) -> HuggingFacePipeline:
    print("==== STARTED: Initializing HuggingFace text generation pipeline ====")

    tokenizer = AutoTokenizer.from_pretrained(model)
    # Ensure tokenizer truncates inputs to avoid indexing errors/warnings
    tokenizer.model_max_length = input_max_length
    tokenizer.truncation_side = "left"

    hf_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=do_sample,
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    print(f"---- COMPLETED SUCCESSFULLY: LLM pipeline '{model}' initialized ----")
    return llm


def build_qa_chain(
    vectorstore: Chroma,
    llm: HuggingFacePipeline,
    k: int = 6,
    chain_type: str = "stuff",
    search_type: str = "mmr",
) -> RetrievalQA:
    print("==== STARTED: Creating RetrievalQA chain ====")

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "fetch_k": max(k * 3, 10), "lambda_mult": 0.5},
    )

    # prompt = PromptTemplate(
    #     input_variables=["context", "question"],
    #     template=(
    #         "Using the provided BBC article excerpts, answer the question with a list of items.\n"
    #         "Each item must be formatted exactly as:\n"
    #         "Title: <title>\n"
    #         "Description: <description>\n\n"
    #         "Question: {question}\n"
    #         "Context:\n{context}"
    #     ),
    # )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are given excerpts from BBC articles, each with Title, Description, and Source URL.\n"
            "Answer the question by listing the most relevant items.\n"
            "Format each item exactly like this:\n"
            "Title: <title>\n"
            "Description: <description>\n"
            "URL: <url>\n\n"
            "Question: {question}\n"
            "Context:\n{context}\n\n"
            "Provide a concise, factual list using the format above."
        ),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    print("---- COMPLETED SUCCESSFULLY: RetrievalQA chain is ready ----")
    return qa_chain


def run_query(qa_chain: RetrievalQA, query: str) -> dict:
    # print("==== STARTED: Running query against RetrievalQA ====")

    result = qa_chain(query)

    # print("---- COMPLETED SUCCESSFULLY: Query executed ----")
    return result


def print_result(result: dict) -> None:
    print("\n==================== ANSWER ====================")

    print(result.get("result", ""))

    print("\n==================== SOURCES ====================")

    for doc in result.get("source_documents", []):
        content = doc.page_content
        metadata = doc.metadata

        # Extract title from the content
        title_line = content.splitlines()[0] if content else ""
        title = title_line.replace("Title:", "").strip()

        source_url = metadata.get("source", "N/A")
        print(f"- {title} ({source_url})")


def main(query: str | None, verbose: bool) -> None:
    clear_console()
    enable_verbose(verbose)

    persist_directory   = "./chroma_bbc"
    embeddings          = load_embeddings()

    # Load vector DB if available, otherwise build it
    if vectorstore_available(persist_directory):
        vectorstore = load_vectorstore(embeddings, persist_directory=persist_directory)
    else:
        dataset_path    = download_dataset()
        df              = load_dataframe(dataset_path)
        documents       = build_documents(df)
        chunks          = split_into_chunks(documents)
        vectorstore     = build_vectorstore(chunks, embeddings, persist_directory=persist_directory)

    llm                 = build_llm()
    qa_chain            = build_qa_chain(vectorstore, llm)

    print("\nAssistant: how can i help you?")
    print("(type 'exit' to quit)\n")

    while True:
        try:
            user_query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_query:
            continue
        if user_query.lower() in {"exit", "quit", ":q", "q"}:
            print("Goodbye.")
            break

        result = run_query(qa_chain, user_query)
        print_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BBC News RAG")

    # parser.add_argument(
    #     "--query",
    #     type=str,
    #     default=None,
    #     help="Question to ask the RAG chain",
    # )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output",
    )
    
    args = parser.parse_args()

    main(query=None, verbose=args.verbose)
