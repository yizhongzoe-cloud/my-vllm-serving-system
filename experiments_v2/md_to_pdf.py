#!/usr/bin/env python3
"""Convert results.md to PDF via pandoc (HTML) + weasyprint."""

import os
import subprocess
import weasyprint

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE = os.path.join(BASE_DIR, "results.md")
CSS_FILE = os.path.join(BASE_DIR, "_style.css")
HTML_FILE = os.path.join(BASE_DIR, "results.html")
PDF_FILE = os.path.join(BASE_DIR, "results.pdf")

# Write CSS to temp file
CSS = """
body {
    font-family: "Noto Sans CJK SC", "Noto Sans", "Helvetica Neue", Arial, sans-serif;
    max-width: 210mm;
    margin: 0 auto;
    padding: 20px;
    font-size: 11pt;
    line-height: 1.5;
}
h1 { font-size: 20pt; margin-top: 0; }
h2 { font-size: 16pt; page-break-before: auto; margin-top: 24pt; }
h3 { font-size: 13pt; margin-top: 16pt; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9pt;
}
th, td {
    border: 1px solid #333;
    padding: 4px 8px;
    text-align: center;
}
th { background: #f0f0f0; font-weight: bold; }
img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 8px 0;
}
hr { margin: 16px 0; }
@page {
    size: A4;
    margin: 15mm;
}
"""

with open(CSS_FILE, "w") as f:
    f.write(CSS)

# Step 1: pandoc md -> self-contained HTML with embedded images
subprocess.run(
    [
        "pandoc", MD_FILE,
        "-o", HTML_FILE,
        "--self-contained",
        "--metadata", "title= ",
        "--css", CSS_FILE,
    ],
    check=True,
    cwd=BASE_DIR,
)
os.remove(CSS_FILE)
print(f"HTML generated: {HTML_FILE}")

# Step 2: weasyprint HTML -> PDF
wp = weasyprint.HTML(filename=HTML_FILE, base_url=BASE_DIR)
wp.write_pdf(PDF_FILE)
print(f"PDF generated: {PDF_FILE}")
