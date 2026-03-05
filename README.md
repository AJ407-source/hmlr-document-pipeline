# HMLR Document Processing Pipeline

A Python pipeline for processing scanned planning documents from HM Land Registry (HMLR). The pipeline extracts text from PDF scans using OCR, classifies document types using a zero-shot transformer model, and extracts key information (applicant names, application numbers) using NER and regex.

## Overview

This project was built as a technical assessment for the Data Scientist role at HMLR. It demonstrates:

- **OCR text extraction** from scanned PDF documents using Tesseract
- **Image preprocessing** using Pillow to improve OCR quality
- **Text cleaning** with regex-based noise removal and error correction
- **Document classification** using a zero-shot transformer (facebook/bart-large-mnli)
- **Named Entity Recognition (NER)** for applicant name extraction (dslim/bert-base-NER)
- **Regex pattern matching** for application number extraction
- **Modular pipeline architecture** with JSON output

## Project Structure

```
hmlr-document-pipeline/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files excluded from version control
├── data/
│   ├── input/                   # Input PDF documents
│   └── output/                  # Pipeline results (not tracked by Git)
├── src/
│   ├── __init__.py              # Package initialiser
│   ├── pipeline.py              # Main pipeline orchestrator
│   ├── pdf_extractor.py         # PDF to text extraction via OCR
│   ├── image_preprocessor.py    # Image preprocessing for OCR improvement
│   ├── text_cleaner.py          # Text cleaning and normalisation
│   ├── page_classifier.py       # Zero-shot document classification
│   ├── name_extractor.py        # NER-based applicant name extraction
│   └── number_extractor.py      # Regex-based application number extraction
└── reports/
    └── analysis_report.md       # Detailed analysis and findings
```

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- Tesseract OCR engine
- Poppler PDF utilities

### Install System Dependencies (Rocky Linux / RHEL)

```bash
sudo dnf install tesseract poppler-utils
```

### Install Python Dependencies

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Usage

### Run the Full Pipeline

```bash
python -m src.pipeline
```

This processes the PDF in `data/input/`, runs all pipeline steps, and saves structured results to `data/output/anonymised_1_results.json`.

### Run Individual Modules

Each module can be run independently for testing:

```bash
# Test OCR extraction
python -m src.pdf_extractor

# Test image preprocessing
python src/image_preprocessor.py

# Test text cleaning
python -m src.text_cleaner

# Test page classification
python -m src.page_classifier

# Test name extraction
python -m src.name_extractor

# Test application number extraction
python -m src.number_extractor
```

## Pipeline Architecture

The pipeline processes documents through six sequential stages:

```
PDF File
  │
  ▼
┌──────────────────────┐
│ 1. PDF Extraction     │  pdf2image + Tesseract OCR
└──────────────────────┘
  │
  ▼
���──────────────────────┐
│ 2. Text Cleaning      │  Garbage removal, whitespace normalisation, OCR fixes
└──────────────────────┘
  │
  ▼
┌──────────────────────┐
│ 3. Classification     │  Zero-shot transformer (BART-MNLI)
└──────────────────────┘
  │
  ▼
┌──────────────────────┐
│ 4. Name Extraction    │  NER transformer (BERT-NER) + keyword proximity
└──────────────────────┘
  │
  ▼
┌──────────────────────┐
│ 5. Number Extraction  │  Regex pattern matching + date filtering
└──────────────────────┘
  │
  ▼
┌──────────────────────┐
│ 6. JSON Output        │  Structured results saved to file
└──────────────────────┘
```

## Results Summary

Processing the test document (4-page anonymised planning PDF):

| Metric | Result |
|--------|--------|
| Document Classification | 4/4 pages correct (100%) |
| Application Number Extraction | 5/5 numbers found (100%) |
| Applicant Name Extraction | 3/5 names found (60%) |
| False Positives | 0 across all tasks |

### Classification Results

| Page | Category | Confidence |
|------|----------|------------|
| 1 | Planning Register | 46.9% |
| 2 | Approval Notice | 73.8% |
| 3 | General Permission | 47.7% |
| 4 | Approval Notice | 74.7% |

### Key Findings

- **Zero-shot classification** achieved 100% accuracy but with varying confidence levels
- **Regex-based number extraction** was the most reliable component (100% with zero false positives)
- **NER name extraction** worked well on clean pages (99%+ confidence) but struggled with noisy OCR text
- **Image preprocessing** did not always improve OCR quality — gentle contrast enhancement outperformed aggressive thresholding
- Full analysis available in [reports/analysis_report.md](reports/analysis_report.md)

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.9 | Primary programming language |
| Tesseract OCR | Optical Character Recognition |
| pdf2image + Poppler | PDF to image conversion |
| Pillow (PIL) | Image preprocessing |
| Hugging Face Transformers | Pre-trained NLP models |
| facebook/bart-large-mnli | Zero-shot document classification |
| dslim/bert-base-NER | Named Entity Recognition |
| PyTorch | Deep learning backend |
| Regular Expressions | Pattern-based text extraction |

## Limitations and Future Improvements

- **OCR quality** is the main bottleneck — noisy text reduces downstream accuracy
- **NER performance** degrades on heavily corrupted OCR text
- **Classification confidence** is low for ambiguous documents (pages with mixed content)
- **Date/number disambiguation** relies on heuristics that may not generalise
- Future work could include fine-tuning models on HMLR-specific documents

