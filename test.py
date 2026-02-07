# # # TESSERACT TESTING

# import pytesseract
# from PIL import Image

# pytesseract.pytesseract.pytesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Now you can use pytesseract

# # # OCR TESTING

# from ocr.ocr_engine import OCREngine
# ocr = OCREngine(
#     tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# )

# result = ocr.extract_text(r"P:\Final_project\K\data\uploads\test_doc_page-0006.jpg")

# print(result["confidence"])
# print(result["status"])
# print(result["text"][:500])

# # # TEXT CLEANER TESTING 

# from utils.text_cleaner import TextCleaner

# cleaner = TextCleaner()

# raw_text = """
# THIS AGREEMENT   is made on 12/12/2023...
# ____ The party shall   indemnify   the other party;;;;;
# """

# cleaned = cleaner.clean_text(raw_text)
# clauses = cleaner.split_into_clauses(cleaned)

# print(cleaned)
# print(len(clauses))

# # # SUMMARIZER TESTING 

from nlp.summarizer import LegalSummarizer

summarizer = LegalSummarizer()

sample_text = """
TENDER NOTICE 1. On behalf of Governer of Tamil Nadu sealed tenders wall be received by 1-1 The tender should be in the prescnbed form obtainable from the office Of IN eee eee Phe tenders will be opened by the, 0.00000... at the place and on the date before mentioned. 1-2 The tenderers or their agents are expected to he present at the time of opening of tenders. The tender receiving officer will on opening each tender, prepare a statement of the attested and unattested corections there 
"""

summary = summarizer.summarize(sample_text)
print(summary)


