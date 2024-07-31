# FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:2.0.0-py3.9.12-cuda11.8.0-u22.04-cudnn

FROM registry.cn-shanghai.aliyuncs.com/aicar/vllm:base


# RUN apt-get update && apt-get install curl


#pip3 install numpy --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com

# RUN pip install --progress-bar off numpy pandas PyPDF2 langchain jieba rank_bm25 sentence-transformers faiss-gpu modelscope tiktoken transformers_stream_generator accelerate pdfplumber --index-url=http://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com


COPY app /app


WORKDIR /app


CMD ["bash", "run.sh"]
