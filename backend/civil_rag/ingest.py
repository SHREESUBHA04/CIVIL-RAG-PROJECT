import os
import re
from pathlib import Path
from typing import List, Dict
import pytesseract
from PIL import Image
import io
import pdfplumber
import fitz  # PyMuPDF
from docx import Document
from dotenv import load_dotenv
import base64
from PIL import Image
import io
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data"
VECTORSTORE_DIR = Path(__file__).parent.parent / "vectorstore"
VECTORSTORE_DIR.mkdir(exist_ok=True)

WINDOW_SIZE = 2   # sentences on each side of the target sentence


load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def image_to_base64(pil_image: Image.Image) -> str:
    """Convert PIL image to base64 string for API"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
def extract_tables_as_sentences(filepath: str) -> str:
    """
    Extract tables from PDF using pdfplumber's table detection.
    Convert each table row into a complete descriptive sentence.
    
    Example:
    Table header: [Cement, w/c ratio, Slump, 28-day strength]
    Row: [440, 0.37, 25, 54]
    
    Becomes:
    "In this mix design table, Cement is 440 kg, w/c ratio is 0.37,
     Slump is 25 mm, and 28-day strength is 54 N/mm2."
    
    This is searchable. The raw "440 0.37 25 54" is not.
    """
    from groq import Groq
    import os
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    all_table_text = ""
    
    try:
        with pdfplumber.open(filepath) as pdf:
            filename = Path(filepath).name
            
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                
                if not tables:
                    continue
                
                for table_index, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    
                    # Clean the table — remove None and empty cells
                    cleaned = []
                    for row in table:
                        cleaned_row = [
                            str(cell).strip() if cell else ""
                            for cell in row
                        ]
                        # Skip completely empty rows
                        if any(cell for cell in cleaned_row):
                            cleaned.append(cleaned_row)
                    
                    if len(cleaned) < 2:
                        continue
                    
                    # Convert table to plain text for LLM
                    table_text = "\n".join(
                        " | ".join(row) for row in cleaned
                    )
                    
                    print(f"    Converting table on page {page_num+1}...")
                    
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[{
                                "role": "user",
                                "content": f"""Convert this civil engineering table into descriptive sentences.

Rules:
- Write one complete sentence per data row
- Include the column header meaning in every sentence
- Include all numerical values with context
- Make each sentence self-contained and searchable
- Example: "For Mix TM-1 with cement content 440 kg/m3, the water-cement ratio is 0.37, slump is 25mm, 7-day strength is 39 N/mm2, and 28-day strength is 54 N/mm2."

Table from page {page_num+1} of {filename}:
{table_text}

Write only the sentences, nothing else:"""
                            }],
                            temperature=0.1,
                            max_tokens=400,
                        )
                        
                        sentences = response.choices[0].message.content.strip()
                        all_table_text += (
                            f"\n[Table from page {page_num+1} of {filename}]\n"
                            f"{sentences}\n"
                        )
                        
                    except Exception as e:
                        # Fallback: just use raw table text
                        print(f"    LLM table conversion failed: {e}")
                        all_table_text += f"\n{table_text}\n"
                        
    except Exception as e:
        print(f"  Table extraction failed: {e}")
    
    return all_table_text

def describe_image_with_vision(
    pil_image: Image.Image,
    page_num: int,
    source_filename: str
) -> str:
    """
    Send an image to a Vision LLM and get a text description.
    
    The description is what gets indexed — not the raw image.
    This converts visual information into searchable text.
    
    We use a specific prompt that forces the model to extract
    every piece of technical information from the image —
    values, labels, axes, equations, tables, arrows, etc.
    """
    try:
        img_b64 = image_to_base64(pil_image)
        
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """You are analyzing a civil engineering document image.
Extract ALL technical information visible in this image.

Include:
- Any equations or formulas (write them out in words and symbols)
- All numerical values with their units
- Axis labels and their ranges (if a graph/chart)
- Table contents (all rows and columns)
- Diagram labels and annotations
- Any text visible in the image
- What the figure/diagram represents
- Key relationships shown (e.g. as X increases, Y decreases)

Be thorough and specific. This description will be used to answer 
technical questions, so include every number, label, and relationship.
Write as flowing technical text, not bullet points."""
                        }
                    ]
                }
            ],
            max_tokens=500,
        )
        
        description = response.choices[0].message.content.strip()
        
        # Wrap with context so retrieval knows where this came from
        return (
            f"[Figure from page {page_num} of {source_filename}]\n"
            f"{description}"
        )
        
    except Exception as e:
        print(f"    Vision LLM failed for image: {e}")
        return ""


def extract_and_describe_images(filepath: str) -> str:
    """
    Extract all meaningful images from a PDF and describe
    each one using a Vision LLM.
    
    Returns all descriptions concatenated as text —
    ready to be chunked and indexed alongside regular text.
    """
    all_descriptions = ""
    
    try:
        doc = fitz.open(filepath)
        filename = Path(filepath).name
        total_described = 0
        
        for page_num, page in enumerate(doc):
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue
            
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    width, height = pil_image.size
                    
                    # Skip tiny images (icons, decorations, logos)
                    # Only process images large enough to contain content
                    if width < 200 or height < 200:
                        continue
                    
                    # Convert to RGB
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    
                    print(f"    Describing image {img_index+1} "
                          f"on page {page_num+1} ({width}×{height})...")
                    
                    description = describe_image_with_vision(
                        pil_image, page_num + 1, filename
                    )
                    
                    if description:
                        all_descriptions += description + "\n\n"
                        total_described += 1
                        
                except Exception as e:
                    print(f"    Skipping image: {e}")
                    continue
        
        doc.close()
        
        if total_described > 0:
            print(f"  ✓ Described {total_described} images using Vision LLM")
        else:
            print(f"  No meaningful images found")
            
    except Exception as e:
        print(f"  Image description failed: {e}")
    
    return all_descriptions

def extract_images_text_from_pdf(filepath: str) -> str:
    """
    Extract text from images embedded WITHIN a PDF.
    
    Works alongside digital text extraction — not instead of it.
    Finds every image on every page, runs OCR on each,
    and appends the results to the page text.
    
    This catches:
    - Tables saved as images
    - Diagrams with text labels
    - Figures with captions embedded in the image
    - Screenshots pasted into documents
    """
    print(f"  Extracting text from embedded images...")
    all_image_text = ""
    
    try:
        doc = fitz.open(filepath)
        
        for page_num, page in enumerate(doc):
            # Get all images on this page
            image_list = page.get_images(full=True)
            
            if not image_list:
                continue
                
            page_image_text = ""
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract the image bytes
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    
                    # Skip tiny images — likely icons or decorations
                    width, height = pil_image.size
                    if width < 100 or height < 100:
                        continue
                    
                    # Convert to RGB if needed
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    
                    # Run OCR
                    img_text = pytesseract.image_to_string(
                        pil_image,
                        lang="eng",
                        config="--psm 6"
                    ).strip()
                    
                    # Only keep if meaningful text found
                    if len(img_text) > 20:
                        page_image_text += img_text + "\n"
                        
                except Exception:
                    continue
            
            if page_image_text:
                all_image_text += f"\n[Page {page_num+1} image text]\n"
                all_image_text += page_image_text
        
        doc.close()
        
        if all_image_text.strip():
            print(f"  ✓ Extracted text from embedded images")
        else:
            print(f"  No text found in embedded images")
            
    except Exception as e:
        print(f"  Image text extraction failed: {e}")
    
    return all_image_text


def extract_text_from_pdf(filepath: str) -> str:
    """
    Complete PDF extraction:
    Layer 1: Digital text (pdfplumber)
    Layer 2: Full page OCR (scanned pages)
    Layer 3: Table-aware extraction (converts tables to sentences)
    Layer 4: Vision LLM for non-table images
    """
    n_pages = 1
    digital_text = ""

    # Layer 1: Digital text
    try:
        with pdfplumber.open(filepath) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    digital_text += page_text + "\n"
    except Exception:
        pass

    if len(digital_text.strip()) < 100:
        try:
            doc = fitz.open(filepath)
            n_pages = len(doc)
            for page in doc:
                digital_text += page.get_text() + "\n"
            doc.close()
        except Exception:
            pass

    chars_per_page = len(digital_text.strip()) / max(n_pages, 1)
    print(f"  Digital text: {len(digital_text.strip())} chars "
          f"({chars_per_page:.0f}/page)")

    # Layer 2: Full page OCR if needed
    scanned_text = ""
    if chars_per_page < 100:
        print(f"  Low text density — running full page OCR")
        scanned_text = extract_text_from_pdf(filepath)

    # Layer 3: Table-aware extraction
    print(f"  Extracting and converting tables...")
    table_text = extract_tables_as_sentences(filepath)
    if table_text.strip():
        print(f"  ✓ Tables converted to {len(table_text)} chars of sentences")

    # Layer 4: Vision LLM for images
    print(f"  Running vision extraction on images...")
    image_descriptions = extract_and_describe_images(filepath)

    # Combine all layers
    combined = "\n\n".join(filter(None, [
        digital_text,
        scanned_text,
        table_text,
        image_descriptions
    ]))

    print(f"  Total extracted: {len(combined.strip())} chars")
    return combined
# ─────────────────────────────────────────
# TEXT EXTRACTORS
# ─────────────────────────────────────────

def extract_text_from_pdf(filepath: str) -> str:
    """
    Complete PDF extraction — text + vision:
    
    Layer 1: Digital text extraction
    Layer 2: Full page OCR (scanned pages)  
    Layer 3: Vision LLM image descriptions ← NEW
    """
    n_pages = 1
    digital_text = ""

    # Layer 1: Digital text
    try:
        with pdfplumber.open(filepath) as pdf:
            n_pages = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    digital_text += page_text + "\n"
    except Exception:
        pass

    if len(digital_text.strip()) < 100:
        try:
            doc = fitz.open(filepath)
            n_pages = len(doc)
            for page in doc:
                digital_text += page.get_text() + "\n"
            doc.close()
        except Exception:
            pass

    chars_per_page = len(digital_text.strip()) / max(n_pages, 1)
    print(f"  Digital text: {len(digital_text.strip())} chars "
          f"({chars_per_page:.0f}/page)")

    # Layer 2: Full page OCR if needed
    scanned_text = ""
    if chars_per_page < 100:
        print(f"  Low text density — running full page OCR")
        scanned_text = extract_text_from_pdf(filepath)

    # Layer 3: Vision LLM for images
    print(f"  Running vision extraction on images...")
    image_descriptions = extract_and_describe_images(filepath)

    # Combine all layers
    combined = "\n\n".join(filter(None, [
        digital_text,
        scanned_text,
        image_descriptions
    ]))

    print(f"  Total extracted: {len(combined.strip())} chars")
    return combined


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from Word documents"""
    doc = Document(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text.strip())
    return "\n".join(paragraphs)


def extract_text_from_txt(filepath: str) -> str:
    """Extract text from plain text files"""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text(filepath: str) -> str:
    """Route to the correct extractor based on file extension"""
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    elif ext == ".txt":
        return extract_text_from_txt(filepath)
    else:
        print(f"Unsupported file type: {ext}")
        return ""


# ─────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean extracted text:
    - Remove page numbers
    - Fix broken hyphenated words
    - Collapse whitespace
    """
    # Remove lines that are just numbers (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Fix hyphenated line breaks (bro-\nken → broken)
    text = re.sub(r'-\n', '', text)

    # Collapse multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


# ─────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into individual sentences.

    Handles common civil engineering edge cases:
    - "IS 456:2000" — colon after number, not a sentence end
    - "w/c ratio of 0.45." — decimal numbers
    - "Cl. 8.2.1" — clause references
    - "Fig. 3" — figure references
    """
    # Protect abbreviations from being split
    # These patterns look like sentence ends but aren't
    protections = [
        (r'(\bIS)\.',        r'\1<DOT>'),       # IS codes
        (r'(\bCl)\.',        r'\1<DOT>'),       # Clause
        (r'(\bFig)\.',       r'\1<DOT>'),       # Figure
        (r'(\bSec)\.',       r'\1<DOT>'),       # Section
        (r'(\bVol)\.',       r'\1<DOT>'),       # Volume
        (r'(\bNo)\.',        r'\1<DOT>'),       # Number
        (r'(\bvs)\.',        r'\1<DOT>'),       # versus
        (r'(\d+)\.(\d+)',    r'\1<DOT>\2'),     # decimal numbers like 0.45
        (r'([A-Z])\.',       r'\1<DOT>'),       # single capital letters (e.g. "N.A.")
    ]

    for pattern, replacement in protections:
        text = re.sub(pattern, replacement, text)

    # Now split on real sentence endings: . ! ?
    # Must be followed by space + capital letter OR end of string
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore protected dots
    sentences = [s.replace('<DOT>', '.') for s in sentences]

    # Clean and filter
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    return sentences


# ─────────────────────────────────────────
# SENTENCE WINDOW CHUNKING
# ─────────────────────────────────────────

def create_sentence_windows(
    sentences: List[str],
    filename: str,
    window_size: int = WINDOW_SIZE
) -> List[Dict]:
    """
    For each sentence, create a chunk that contains:
    - 'text'    : just this sentence (used for FAISS retrieval)
    - 'context' : this sentence + surrounding window (sent to LLM)
    - 'source'  : which document it came from
    - 'chunk_id': unique identifier

    Example with window_size=2:
    sentences = [S0, S1, S2, S3, S4]
    For S2:
        text    = "S2"
        context = "S1 S2 S3"  (1 before + target + 1 after, 
                                capped at document boundaries)
    """
    chunks = []

    for i, sentence in enumerate(sentences):
        # Calculate window boundaries (don't go out of bounds)
        start = max(0, i - window_size)
        end = min(len(sentences), i + window_size + 1)

        # Build context from surrounding sentences
        context_sentences = sentences[start:end]
        context = " ".join(context_sentences)

        chunks.append({
            "text": sentence,          # precise — for retrieval
            "context": context,        # rich — for LLM answer generation
            "source": filename,
            "chunk_id": f"{filename}_sent_{i}",
            "sentence_index": i,
        })

    return chunks


# ─────────────────────────────────────────
# MAIN INGESTION FUNCTION
# ─────────────────────────────────────────

def ingest_documents(data_dir: str = None) -> List[Dict]:
    """
    Load all documents from data/ folder.
    Returns a list of sentence-window chunks.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    all_chunks = []
    supported = {".pdf", ".docx", ".txt"}

    files = [f for f in data_dir.iterdir() if f.suffix.lower() in supported]

    if not files:
        print(f"⚠ No documents found in {data_dir}")
        return []

    for filepath in files:
        print(f"\nProcessing: {filepath.name}")

        # Extract
        raw_text = extract_text(str(filepath))
        if not raw_text.strip():
            print(f"  ⚠ No text extracted from {filepath.name}")
            continue

        # Clean
        cleaned = clean_text(raw_text)

        # Split into sentences
        sentences = split_into_sentences(cleaned)
        print(f"  → {len(sentences)} sentences found")

        # Create sentence windows
        chunks = create_sentence_windows(sentences, filepath.name)
        all_chunks.extend(chunks)
        print(f"  ✓ {len(chunks)} sentence-window chunks created")

    print(f"\n{'─'*40}")
    print(f"Total chunks ready for embedding: {len(all_chunks)}")
    return all_chunks


# ─────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    chunks = ingest_documents()

    if chunks:
        # Safely pick a sample index
        sample_index = min(5, len(chunks) - 1)
        print(f"\n--- Sample Chunk (index {sample_index}) ---")
        print(f"TEXT (for retrieval):\n{chunks[sample_index]['text']}")
        print(f"\nCONTEXT (for LLM):\n{chunks[sample_index]['context']}")
        print(f"\nSource: {chunks[sample_index]['source']}")
        print(f"Chunk ID: {chunks[sample_index]['chunk_id']}")