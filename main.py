import os
import json
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import TextNode
from tqdm.notebook import tqdm
import pandas as pd
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from pathlib import Path


def get_file_paths_train():
    paths=[]
    for r, d, f in os.walk(r'./data/train'):
        for file in f:
            if '.pdf' in file:
                paths.append(os.path.join(r, file))
    return paths

def get_file_paths_val():
    paths=[]
    for r, d, f in os.walk(r'./data/val'):
        for file in f:
            if '.pdf' in file:
                paths.append(os.path.join(r, file))
    return paths

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f"Loaded {len(docs)} docs")

    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f"Parsed {len(nodes)} nodes")

    return nodes

def evaluate_st(
    dataset,
    model_id,
    name,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name=name
    )
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)

def training():
    TRAIN_FILES =get_file_paths_train()
    VAL_FILES = get_file_paths_val()

    train_nodes = load_corpus(TRAIN_FILES, verbose=True)
    val_nodes = load_corpus(VAL_FILES, verbose=True)
    print("--Corpus Loaded--")

    OPENAI_API_TOKEN = "your open ai key"
    os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN
    print("generating Qa-embedding Pairs")
    train_dataset = generate_qa_embedding_pairs(
    llm=OpenAI(model="gpt-3.5-turbo"), nodes=train_nodes
    )
    val_dataset = generate_qa_embedding_pairs(
    llm=OpenAI(model="gpt-3.5-turbo"), nodes=val_nodes
    )


    train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
    val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

    train_dataset.save_json("train_dataset.json")
    val_dataset.save_json("val_dataset.json")

    print("--Model Initialized--")
    finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-base-en-v1.5",
    model_output_path="test_model",
    val_dataset=val_dataset,
    )
    print("--Training started--")
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()
    print("--Training Finished--")

    print("Evalution Started")
    base_result=evaluate_st(val_dataset, "BAAI/bge-base-en-v1.5", name="bge")
    finetuned_result=evaluate_st(val_dataset, "test_model", name="finetuned")
    print("---Base-BGE-Result:",base_result)
    print("---Finetuned-BGE-Result:",finetuned_result)




if __name__ == "__main__":
    print('--Running script--')
    training()
    