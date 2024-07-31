#!/usr/bin/env python
# coding: utf-8


from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pdf_parse import DataProcess
import torch

class FaissRetriever(object):
    # Initialize document block index and then insert into Faiss library
    def __init__(self, model_path, data):
        self.embeddings = HuggingFaceEmbeddings(
                               model_name=model_path,
                               model_kwargs={"device": "cuda"}
                               # model_kwargs = {"device":"cuda:1"}
                           )
        docs = []
        for idx, line in enumerate(data):
            line = line.strip("\n").strip()
            words = line.split("\t")
            docs.append(Document(page_content=words[0], metadata={"id": idx}))
        self.vector_store = FAISS.from_documents(docs, self.embeddings)
        del self.embeddings
        torch.cuda.empty_cache()

    # Get the top-K highest scoring document blocks
    def GetTopK(self, query, k):
       context = self.vector_store.similarity_search_with_score(query, k=k)
       return context

    # Return the Faiss vector retrieval object
    def GetvectorStore(self):
        return self.vector_store


if __name__ == "__main__":
    base = "."
    model_name = base + "/pre_train_model/m3e-large" #text2vec-large-chinese
    dp =  DataProcess(pdf_path = base + "/data/train_a.pdf")
    dp.ParseBlock(max_seq = 1024)
    dp.ParseBlock(max_seq = 512)
    print(len(dp.data))
    dp.ParseAllPage(max_seq = 256)
    dp.ParseAllPage(max_seq = 512)
    print(len(dp.data))
    dp.ParseOnePageWithRule(max_seq = 256)
    dp.ParseOnePageWithRule(max_seq = 512)
    print(len(dp.data))
    data = dp.data

    faissretriever = FaissRetriever(model_name, data)
    faiss_ans = faissretriever.GetTopK("How to prevent COVID-19", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("How to handle traffic accidents", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("Who is the chairman of Geely Group", 6)
    print(faiss_ans)
    faiss_ans = faissretriever.GetTopK("What is the name of the voice group of Geely Auto", 6)
    print(faiss_ans)

