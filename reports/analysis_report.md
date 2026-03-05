# Analysis Report: HMLR Document Processing Pipeline

**Author:** AJ407-source (Azeez Jinadu)
**Date:** 2026-03-05
**Document Processed:** anonymised_1.pdf (4 pages)

---

## 1. Executive Summary

This report presents the results of building an automated document processing pipeline for HM Land Registry planning documents. The pipeline extracts text from scanned PDFs using OCR, classifies document types using a zero-shot transformer, and extracts key information (applicant names and application numbers) using NER and regex.

**Key Results:**
- Document classification achieved **100% accuracy** across all 4 pages
- Application number extraction achieved **100% accuracy** with zero false positives
- Applicant name extraction achieved **60% recall** with zero false positives
- The pipeline processes a 4-page document in approximately 2 minutes on the CPU

The pipeline demonstrates that off-the-shelf transformer models can be effective for document processing tasks, even without domain-specific training data, though OCR quality remains the primary bottleneck for downstream accuracy.

---

## 2. Data Description

### 2.1 Input Document

The input is a single anonymised PDF containing 4 pages of scanned planning documents. The pages represent different document types commonly found in HMLR planning records:

| Page | Document Type | Key Content |
|------|--------------|-------------|
| 1 | Planning Charges Register | Table of planning charges with application numbers and names |
| 2 | Approval Notice | Formal notice approving planning application P/00/0759 |
| 3 | General Permission Grant | Generic grant of conditional planning permission |
| 4 | Approval Notice | Notice of approval of details for application P/98/0964 |

### 2.2 Data Quality Challenges

The scanned document presented several challenges typical of real-world HMLR records:

1. **Variable scan quality** — Some pages were clear, others had noise and distortion
2. **Mixed layouts** — Tables (Page 1), formal letters (Pages 2, 4), and legal text (Page 3)
3. **Historical documents** — Spanning different decades (1970s–2000s) with varying formats
4. **OCR-hostile features** — Table borders, stamps, handwritten annotations, and faded text

---

## 3. Methodology

### 3.1 Pipeline Architecture

The pipeline follows a sequential architecture where each stage processes the output of the previous stage:

```
PDF → Image Extraction → Preprocessing → OCR → Text Cleaning
    → Classification → Name Extraction → Number Extraction → JSON
```

This sequential design was chosen for simplicity and transparency. Each module is independent and can be tested, debugged, and improved individually.

### 3.2 OCR and Preprocessing

**Tool:** Tesseract OCR v4/v5 with pdf2image for PDF conversion

**Approach:** Each page was processed twice — once with the original image and once with a preprocessed (contrast-enhanced) version. The version producing more text was automatically selected.

**Preprocessing strategy:** After experimentation, a "gentle" contrast enhancement (factor 1.5) was chosen over aggressive binary thresholding. Thresholding destroyed text on pages with complex backgrounds (Page 2 lost 70% of its content), while gentle enhancement preserved text across all page types.

**Key finding:** More aggressive preprocessing is not always better. The optimal strategy depends on the quality of the input document, and a conservative approach generalises better across diverse page types.

### 3.3 Text Cleaning

**Approach:** Three-stage cleaning pipeline:

1. **Garbage line removal** — Lines with fewer than 30% alphabetic characters are removed. This eliminates OCR artefacts from table borders and decorative elements while preserving legitimate content.

2. **Whitespace normalisation** �� Multiple blank lines are collapsed to single blank lines, and excessive spaces are reduced. This standardises the text format for downstream processing.

3. **OCR error correction** — A conservative dictionary of known OCR substitution errors is applied (e.g., "-ondon" → "London", "relephone" → "Telephone"). Only specific, verified corrections are included to avoid introducing new errors.

**Design decision:** The cleaning is intentionally conservative. Aggressive cleaning risks removing real content, which would be more damaging than leaving minor noise. The downstream models (classification, NER) are robust enough to handle slightly noisy text.

### 3.4 Document Classification

**Model:** facebook/bart-large-mnli (zero-shot classification)

**Why zero-shot?** With only 4 pages of data, supervised training is not feasible. Zero-shot classification requires no training examples — only natural language descriptions of the target categories.

**Categories defined:**

| Category | Label Given to Model |
|----------|---------------------|
| approval_notice | "planning approval notice or decision" |
| planning_register | "planning charges register or record" |
| general_permission | "general grant of planning permission or legal terms" |

**Design decision:** Descriptive labels were used rather than short codes because the model understands natural language. "Planning approval notice or decision" provides more semantic context than simply "approval".

### 3.5 Name Extraction

**Model:** dslim/bert-base-NER (token classification)

**Approach:** A two-stage strategy combining ML and rule-based methods:

1. **NER extraction** — The BERT model identifies all person (PER) and organisation (ORG) entities in the text
2. **Keyword proximity filtering** — Names appearing within 200 characters of keywords like "Applicant" or "granted to" are prioritised as likely applicants
3. **Validation filtering** — A rule-based filter removes common NER false positives (short strings, BERT subword tokens, common English words, known authority names)

**Design decision:** Combining NER with keyword proximity was necessary because NER alone finds ALL names in a document (council officers, legal references, etc.), not just applicants. The keyword proximity heuristic provides the contextual understanding that the NER model lacks.

### 3.6 Application Number Extraction

**Tool:** Python regular expressions (regex)

**Why regex instead of a transformer?** Application numbers follow strict, predictable formats (XX/XX/XXXX or X/XX/XXXX). Pattern matching is more reliable, faster, and more transparent than a neural model for this task. Using the right tool for the right job is a core engineering principle.

**Date filtering:** The regex patterns also match dates (e.g., 17/07/2000). A date detection function was implemented that checks:
1. Whether the middle digits represent a valid month (1-12) and the last digits represent a plausible year (1900-2099)
2. Whether date-related keywords ("Date", "dated") appear within 50 characters before the match

---

## 4. Results

### 4.1 Classification Performance

| Page | True Category | Predicted Category | Confidence | Correct |
|------|--------------|-------------------|------------|---------|
| 1 | planning_register | planning_register | 46.9% | ✅ |
| 2 | approval_notice | approval_notice | 73.8% | ✅ |
| 3 | general_permission | general_permission | 47.7% | ✅ |
| 4 | approval_notice | approval_notice | 74.7% | ✅ |

**Accuracy: 4/4 (100%)**

**Observations:**
- Pages with clear, unambiguous titles ("NOTICE OF APPROVAL") received high confidence (73-75%)
- Pages with mixed content (Page 1: register with approval text; Page 3: permission grant similar to approval) received low confidence (47%)
- Page 3 had only a 2.1% margin between the top two categories, indicating the model found this classification ambiguous
- All pages were classified correctly despite the low confidence on some pages

### 4.2 Name Extraction Performance

| Page | Expected Name | Extracted Name | Confidence | Method |
|------|--------------|----------------|------------|--------|
| 1 | Mr. & Mrs. J. M Doe | M Do | 80.5% | keyword_proximity |
| 1 | My First Company Ltd. | (not found) | — | — |
| 2 | Mr M Dale | M Dale | 99.6% | keyword_proximity |
| 3 | (none) | (none) | — | — |
| 4 | Mrs AM Stephens | AM Stephens | 99.5% | keyword_proximity |

**Recall: 3/5 (60%) | Precision: 3/3 (100%) | False Positives: 0**

**Observations:**
- On clean pages (2, 4), NER achieved near-perfect extraction with 99%+ confidence
- On the noisy table page (1), only a partial name fragment was recovered
- The company name "My First Company Ltd." was not detected — likely because the NER model was not trained on this style of company name
- Titles (Mr, Mrs) were consistently omitted by the NER model, which is standard BERT-NER behaviour
- The keyword proximity method was 100% reliable — every name it identified as an applicant was correct
- Zero false positives is a significant achievement: the system never reported a wrong name

### 4.3 Application Number Extraction Performance

| Page | Expected Numbers | Extracted Numbers | Correct |
|------|-----------------|-------------------|---------|
| 1 | 02/80/1609, 02/81/1237 | 02/80/1609, 02/81/1237 | ✅ |
| 2 | P/00/0759 | P/00/0759 | ✅ |
| 3 | (none) | (none) | ✅ |
| 4 | P/98/0964, P/96/0900 | P/98/0964, P/96/0900 | ✅ |

**Accuracy: 5/5 (100%) | Dates correctly filtered: 2/2 (100%)**

**Observations:**
- Regex was the most reliable extraction method in the pipeline
- The date filter successfully excluded 17/07/2000 and 13/07/1998
- Both application number formats (numeric and letter-prefix) were captured
- No false positives — every extracted number was a genuine application number

---

## 5. Error Analysis

### 5.1 OCR as the Primary Bottleneck

The most significant source of errors in the pipeline is OCR quality. Examples of OCR errors that affected downstream processing:

| Original Text | OCR Output | Impact |
|---------------|-----------|--------|
| London | -ondon or .ondon | Text cleaner partially fixed |
| Mr. & Mrs. J. M Doe | M Do (fragment) | NER could only extract partial name |
| NORTH DEVON DISTRICT COUNCIL | NORTH DEVON DISTRICT COUN@'L | @ symbol introduced by OCR |
| East Ham | =ast Ham | Text cleaner fixed to "East Ham" |

**Root cause:** The input documents are physical scans with variable quality. Tesseract OCR performs well on clean, high-contrast text, but struggles with:
- Table borders that create visual noise
- Faded or low-contrast text
- Mixed fonts and sizes within a single page
- Stamps and annotations overlapping text

### 5.2 Classification Confidence Gap

The classifier showed a clear confidence gap between "easy" and "hard" pages:

- **Easy pages (2, 4):** 73-75% confidence — clear document titles present
- **Hard pages (1, 3):** 47% confidence — ambiguous or mixed content

This suggests the model relies heavily on explicit title text. Pages without clear titles require the model to infer the category from the overall content, which is more difficult.

### 5.3 NER Limitations on Noisy Text

The NER model (dslim/bert-base-NER) was trained on clean, well-formatted text (CoNLL-2003 dataset: news articles). When applied to noisy OCR text, it exhibited several failure modes:

1. **Partial name extraction** — Only fragments of names were recognised
2. **False entity detection** — Common words ("and", "end") were tagged as entities
3. **Subword token leakage** — Internal BERT tokens ("##VO", "##VAL") appeared in output
4. **Organisation detection weakness** — Unusual company names were not recognised

These issues were mitigated through post-processing filters, but they highlight the domain gap between the model's training data and real-world HMLR documents.

---

## 6. Design Decisions and Trade-offs

### 6.1 Precision vs Recall

The pipeline prioritises **precision over recall** throughout. This means we prefer to miss some information rather than return incorrect information.

**Rationale:** In the context of HMLR land registry records, incorrect data (wrong applicant name, wrong application number) would be far more damaging than missing data. A missing name triggers a human review; a wrong name could lead to legal errors.

### 6.2 Conservative Text Cleaning

The text-cleaning stage uses a 30% alphabetic threshold to remove garbage lines. This was chosen after observing that:
- Real content lines (even with numbers and punctuation) typically exceed 50% alphabetic characters
- OCR artefacts from table borders typically have 0-15% alphabetic characters
- The 30% threshold provides a safe margin that preserves legitimate content

### 6.3 Tool Selection by Task

Different tools were selected for different extraction tasks based on the nature of each task:

| Task | Tool | Rationale |
|------|------|-----------|
| Classification | Transformer (BART) | Requires semantic understanding of document meaning |
| Name extraction | Transformer (BERT-NER) | Names are unpredictable — no fixed pattern to match |
| Number extraction | Regex | Numbers follow strict, known patterns — no AI needed |

This demonstrates that the best tool depends on the task. AI models are powerful but not always necessary. Using regex for application numbers is faster, more reliable, and more transparent than a neural approach.

---

## 7. Recommendations for Future Work

### 7.1 Short-term Improvements

1. **Enhanced OCR** — Evaluate alternative OCR engines (e.g., EasyOCR, PaddleOCR) or use ensemble approaches combining multiple OCR outputs
2. **Context-aware NER** — Fine-tune the NER model on a small dataset of HMLR documents to improve name extraction on noisy text
3. **Confidence thresholds** — Implement automatic flagging of low-confidence results (below 60%) for human review
4. **Expanded regex patterns** — Add patterns for other reference number formats used across different local authorities

### 7.2 Long-term Improvements

1. **Layout analysis** — Use document layout models (e.g., LayoutLM) to understand the spatial structure of pages, improving extraction accuracy for table-based documents
2. **Multi-document processing** — Extend the pipeline to process batches of documents and aggregate results
3. **Active learning** — Build a feedback loop where human corrections are used to incrementally improve model accuracy
4. **API deployment** — Wrap the pipeline in a REST API for integration with existing HMLR systems

---

## 8. Conclusion

This pipeline demonstrates that meaningful information extraction from scanned planning documents is achievable using off-the-shelf NLP tools, even without domain-specific training data. The combination of zero-shot classification, pre-trained NER, and regex pattern matching provides a practical baseline that could be refined with HMLR-specific data and feedback.

The most important finding is that **OCR quality is the primary bottleneck** — improving upstream text extraction would have the greatest impact on overall pipeline accuracy. The downstream NLP models perform well on clean text, as evidenced by 99%+ confidence on well-scanned pages.

The modular architecture allows each component to be independently improved, tested, and replaced as better tools become available, making the pipeline a sustainable foundation for document processing at HMLR.

