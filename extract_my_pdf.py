import urllib.request
import os

try:
    from pypdf import PdfReader
except ImportError:
    os.system("pip install pypdf")
    from pypdf import PdfReader

try:
    reader = PdfReader("Attention Is Not All You Need Beating Hallucinations Using Causal Dynamic Routing.pdf")
    pages = [page.extract_text() for page in reader.pages]
    with open("report_dump.txt", "w", encoding="utf-8") as f:
        f.write("\n=======================\n".join(pages))
    print("SUCCESS: report_dump.txt created")
except Exception as e:
    print(f"FAILED: {e}")
