#!/usr/bin/env python
# coding: utf-8

import json
import jieba
import pandas as pd
import numpy as np
from tqdm import tqdm
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain import PromptTemplate, LLMChain
from langchain.chains import RetrievalQA
import time
import re

from vllm_model import ChatLLM
from rerank_model import reRankLLM
from faiss_retriever import FaissRetriever
from bm25_retriever import BM25
from pdf_parse import DataProcess

# get langchain
def get_qa_chain(llm, vector_store, prompt_template):

    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])

    return RetrievalQA.from_llm(llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 10}), prompt=prompt)

# get prompt，use merged faiss and bm25 retriver to get answer
def get_emb_bm25_merge(faiss_context, bm25_context, query):
    max_length = 2500
    emb_ans = ""
    cnt = 0
    for doc, score in faiss_context:
        cnt =cnt + 1
        if(cnt>6):
            break
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    bm25_ans = ""
    cnt = 0
    for doc in bm25_context:
        cnt = cnt + 1
        if(len(bm25_ans + doc.page_content) > max_length):
            break
        bm25_ans = bm25_ans + doc.page_content
        if(cnt > 6):
            break

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案"，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                2: {bm25_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, bm25_ans = bm25_ans, question = query)
    return prompt_template


def get_rerank(emb_ans, query):

    prompt_template = """基于以下已知信息，简洁和专业的来回答用户的问题。
                                如果无法从中得到答案，请说 "无答案"或"无答案" ，不允许在答案中添加编造成分，答案请使用中文。
                                已知内容为吉利控股集团汽车销售有限公司的吉利用户手册:
                                1: {emb_ans}
                                问题:
                                {question}""".format(emb_ans=emb_ans, question = query)
    return prompt_template


def question(text, llm, vector_store, prompt_template):

    chain = get_qa_chain(llm, vector_store, prompt_template)

    response = chain({"query": text})
    return response

def reRank(rerank, top_k, query, bm25_ans, faiss_ans):
    items = []
    max_length = 4000
    for doc, score in faiss_ans:
        items.append(doc)
    items.extend(bm25_ans)
    rerank_ans = rerank.predict(query, items)
    rerank_ans = rerank_ans[:top_k]
    # docs_sort = sorted(rerank_ans, key = lambda x:x.metadata["id"])
    emb_ans = ""
    for doc in rerank_ans:
        if(len(emb_ans + doc.page_content) > max_length):
            break
        emb_ans = emb_ans + doc.page_content
    return emb_ans

if __name__ == "__main__":

    start = time.time()

    base = "."
    qwen7 = "/afs/crc.nd.edu/user/d/dding3/.cache/modelscope/hub/qwen/Qwen-7B-Chat/"
    m3e =  base + "/pre_train_model/m3e-large"
    bge_reranker_large = base + "/pre_train_model/bge-reranker-large"

    # parse pdf
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
    print("data load ok")

    # Faiss retriever
    faissretriever = FaissRetriever(m3e, data)
    vector_store = faissretriever.vector_store
    print("faissretriever load ok")

    # BM25 retriver
    bm25 = BM25(data)
    print("bm25 load ok")

    # LLM 
    llm = ChatLLM(qwen7)
    print("llm qwen load ok")

    # reRank
    rerank = reRankLLM(bge_reranker_large)
    print("rerank model load ok")

    # get answer 
    with open(base + "/data/test_question.json", "r") as f:
        jdata = json.loads(f.read())
        print(len(jdata))
        max_length = 4000
        for idx, line in enumerate(jdata):
            query = line["question"]

            # faiss topk
            faiss_context = faissretriever.GetTopK(query, 15)
            faiss_min_score = 0.0
            if(len(faiss_context) > 0):
                faiss_min_score = faiss_context[0][1]
            cnt = 0
            emb_ans = ""
            for doc, score in faiss_context:
                cnt =cnt + 1
                # max length
                if(len(emb_ans + doc.page_content) > max_length):
                    break
                emb_ans = emb_ans + doc.page_content
                # six as max
                if(cnt>6):
                    break

            # bm2.5 topk
            bm25_context = bm25.GetBM25TopK(query, 15)
            bm25_ans = ""
            cnt = 0
            for doc in bm25_context:
                cnt = cnt + 1
                if(len(bm25_ans + doc.page_content) > max_length):
                    break
                bm25_ans = bm25_ans + doc.page_content
                if(cnt > 6):
                    break

            # Construct a prompt for merging BM25 recall and vector recall
            emb_bm25_merge_inputs = get_emb_bm25_merge(faiss_context, bm25_context, query)
            
            # Construct a prompt for BM25 recall
            bm25_inputs = get_rerank(bm25_ans, query)
            
            # Construct a prompt for vector recall
            emb_inputs = get_rerank(emb_ans, query)
            
            # Rerank the recall candidates and sort them by relevance score
            rerank_ans = reRank(rerank, 6, query, bm25_context, faiss_context)
            # Construct a prompt for generating the answer after reranking
            rerank_inputs = get_rerank(rerank_ans, query)
            
            batch_input = []
            batch_input.append(emb_bm25_merge_inputs)
            batch_input.append(bm25_inputs)
            batch_input.append(emb_inputs)
            batch_input.append(rerank_inputs)
            # Execute batch inference
            batch_output = llm.infer(batch_input)
            line["answer_1"] = batch_output[0] # Result of merging the two recalls
            line["answer_2"] = batch_output[1] # Result of BM recall
            line["answer_3"] = batch_output[2] # Result of vector recall
            line["answer_4"] = batch_output[3] # Result after reranking multiple recalls
            line["answer_5"] = emb_ans
            line["answer_6"] = bm25_ans
            line["answer_7"] = rerank_ans
            # If the distance between the Faiss retrieval and the query is greater than 500, output no answer

            if(faiss_min_score >500):
                line["answer_5"] = "无答案"
            else:
                line["answer_5"] = str(faiss_min_score)

        # save the results
        json.dump(jdata, open(base + "/data/result.json", "w", encoding='utf-8'), ensure_ascii=False, indent=2)
        end = time.time()
        print("cost time: " + str(int(end-start)/60))
