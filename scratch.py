import pdfplumber
from PyPDF2 import PdfReader


def ParseBlock(pdf_path, max_seq=1024):
    with pdfplumber.open(pdf_path) as pdf:

        for i, p in enumerate(pdf.pages):
            texts = p.extract_words(use_text_flow=True, extra_attrs=["size"])[::]
            print(texts)
            print(texts.type)
            break


if __name__ == "__main__":
    pdf_path = "./data/train_a.pdf"
    ParseBlock(pdf_path,max_seq = 1024)
