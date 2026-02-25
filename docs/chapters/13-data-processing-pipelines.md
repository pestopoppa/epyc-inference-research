# Chapter 13: Data Processing Pipelines

## Introduction

The orchestration system provides two integrated data processing pipelines: **document processing** (OCR, chunking, figure extraction) and **vision processing** (face detection, embeddings, VL description). Both pipelines are designed for CPU-only inference on the EPYC 9655 with emphasis on throughput and memory efficiency.

Document processing achieved a **19x OCR speedup** by routing born-digital PDFs to pdftotext (fast path) while scanned documents use LightOnOCR-2-1B. Vision processing provides face recognition, semantic image search via CLIP embeddings, and VL model integration on ports 8086 (worker) and 8087 (escalation).

## Document Processing Pipeline

The document pipeline handles everything from PDF ingestion to chunked, figure-annotated output. It starts with a smart router that checks whether a PDF is born-digital or scanned, then fans out to OCR, chunking, figure analysis, and archive extraction as needed.

<details>
<summary>Pipeline architecture and components</summary>

### Architecture Overview

```
PDF Input
    ↓
[pdf_router] → Quality check (entropy, char ratio)
    │
    ├─ Born-digital (>90% quality): pdftotext + PyMuPDF figures
    │   ├─ Text: pdftotext output (~100ms)
    │   └─ Figures: PyMuPDF bbox extraction
    │
    └─ Scanned/image (<90% quality): LightOnOCR
        ├─ Text: OCR with bounding boxes
        └─ Figures: Detected regions from OCR
            ↓
[document_chunker] → Split by markdown headers
    ↓
[figure_analyzer] → VL model descriptions (optional)
    ↓
[archive_extractor] → Handle .zip/.tar/.7z archives
```

### PDF Router (Fast Path Detection)

**Purpose**: Intelligently route PDFs to optimal extraction method
**Implementation**: `src/services/pdf_router.py`
**Speedup**: 19x for born-digital PDFs

**Quality Assessment Metrics**:

| Metric | Threshold | Purpose |
|--------|-----------|---------|
| Shannon entropy | > 3.5 | Detect readable text vs garbage |
| Garbage ratio | < 0.15 | Max non-printable characters |
| Avg word length | > 2.5 chars | Verify word-like tokens |
| Min text length | 100 chars | Sufficient sample for assessment |

**Fast Path Decision**:

<details>
<summary>Code: PDF routing and fast-path extraction</summary>

```python
from src.services.pdf_router import PDFRouter

router = PDFRouter()
result = await router.extract(pdf_path)

if result.method == "pdftotext":
    # Fast path: ~100ms for born-digital PDF
    # Quality score: 0.92 (high confidence)
    text = result.text
    figures = result.figures  # From PyMuPDF
elif result.method == "lightonocr":
    # OCR path: ~2-5s for scanned PDF
    # Quality score: 0.68 (OCR uncertainty)
    text = result.text
    figures = result.figures  # From OCR bboxes
```

</details>

**Performance** (50-page technical document):

| Method | Latency | Quality Score |
|--------|---------|---------------|
| pdftotext | 120ms | 0.94 |
| LightOnOCR | 2.3s | 0.71 |
| Speedup | **19x** | — |

### LightOnOCR Server (Port 9001)

**Model**: LightOnOCR-2-1B
**Endpoint**: `http://localhost:9001/ocr/pdf`
**Output Format**: Text + bounding boxes (bbox coordinates normalized 0-1000)

**Status Codes**:
- `200`: Success
- `202`: Async processing (job_id returned)
- `400`: Invalid request
- `503`: Server unavailable

<details>
<summary>Code: LightOnOCR client usage</summary>

```python
from src.services.document_client import get_document_client, process_document
from src.models.document import DocumentProcessRequest

request = DocumentProcessRequest(
    file_path="/path/to/scan.pdf",
    output_format="bbox",
    max_pages=100,
    dpi=200,
    extract_figures=True,
)

result = await process_document(request)

for page in result.pages:
    print(f"Page {page.page_num}: {len(page.text)} chars")
    for bbox in page.bboxes:
        print(f"  Figure at ({bbox.x1}, {bbox.y1})")
```

</details>

### Document Chunker (Markdown Structure)

**Purpose**: Split documents by headers for context-aware retrieval
**Implementation**: `src/services/document_chunker.py`

<details>
<summary>Code: Chunking configuration and output structure</summary>

**Chunking Strategy**:

```python
from src.services.document_chunker import chunk_document, ChunkingConfig

config = ChunkingConfig(
    max_header_level=3,      # Split at ###, ##, #
    min_section_length=10,   # Skip tiny sections
    max_section_length=10000,  # Split long sections
    min_figure_area=5000,    # Min pixels for figure detection
)

document = chunk_document(ocr_result, file_path, config)

for section in document.sections:
    print(f"{section.title} (pages {section.page_start}-{section.page_end})")
    print(f"  {len(section.content)} chars, {len(section.figure_ids)} figures")
```

**Output Structure**:

```
Section(
    id="s1",
    title="Abstract",
    level=1,
    content="...",
    page_start=1,
    page_end=1,
    figure_ids=["fig_1", "fig_2"],
)
```

</details>

### Figure Analyzer (Vision Model)

**Purpose**: Generate descriptions for extracted figures
**Implementation**: `src/services/figure_analyzer.py`
**Endpoint**: `http://localhost:8000/v1/vision/analyze`

**Concurrency Control**: Max 4 concurrent VL API calls (semaphore-based)

**Workflow**:

1. Render PDF pages to images (pypdfium2, 200 DPI)
2. Crop figure regions using bounding boxes
3. Convert crops to base64
4. Send to VL model with document context
5. Store descriptions in FigureRef objects

**Performance** (10 figures, 4 concurrent):

- Total time: 12.4s
- Per-figure latency: 3.1s avg
- Throughput: 0.81 figures/s

<details>
<summary>Code: Document context injection for figure analysis</summary>

```python
# Extract summary from Abstract/Introduction for context
summary = preprocessor._extract_summary_context(
    document_result,
    max_chars=8000,
)

# Build VL prompt with context
prompt = f"""You are analyzing a figure from a technical document.
Below is a summary of the document for context.

=== DOCUMENT SUMMARY ===
{summary}
=== END SUMMARY ===

Now analyze this figure:
1. What type of visualization is this?
2. What parameters/data are shown?
3. What is the main insight?"""

figures = await analyze_figures(pdf_path, figures, vl_prompt=prompt)
```

</details>

### Archive Extractor (Multi-Document)

**Purpose**: Process .zip/.tar/.7z archives containing multiple documents
**Implementation**: `src/services/archive_extractor.py`

**Security Features**:
- Zip bomb detection (compression ratio > 100:1)
- Path traversal prevention
- Size limits (500MB archive, 1GB extracted, 100MB per file)
- File count limits (1000 files max)

**Extraction Strategies**:

| Strategy | Condition | Behavior |
|----------|-----------|----------|
| `auto_all` | < 20 files, < 5MB | Extract everything automatically |
| `manifest_then_ask` | 20-100 files | Show manifest, let user select |
| `summary_with_recommendations` | > 100 files | Show summary + file type stats |

<details>
<summary>Code: Archive extraction and validation</summary>

```python
from src.services.archive_extractor import ArchiveExtractor

extractor = ArchiveExtractor()

# Get manifest without extracting
manifest = extractor.list_contents(Path("docs.zip"))
print(f"{manifest.total_files} files, {manifest.compression_ratio:.1f}:1 ratio")
print(f"Types: {manifest.type_summary}")  # {'.pdf': 12, '.md': 5}

# Validate for security issues
validation = extractor.validate(Path("docs.zip"))
if not validation.is_safe:
    print(f"Issues: {validation.issues}")

# Extract specific files
result = extractor.extract_files(
    Path("docs.zip"),
    files=["report.pdf", "readme.md"],
    dest=Path("/mnt/raid0/llm/tmp/archives/session123/"),
)

for filename, path in result.extracted_files.items():
    print(f"Extracted {filename} to {path}")
```

</details>

</details>

## Vision Processing Pipeline

The vision pipeline chains seven analyzers in sequence -- from EXIF metadata through face detection, VL descriptions, and CLIP embeddings. It stores everything in SQLite for structured queries and ChromaDB for semantic image search.

<details>
<summary>Pipeline architecture, analyzers, and storage</summary>

### Architecture Overview

The vision pipeline orchestrates 7 analyzers in optimal order:

```
Image Input
    ↓
1. [EXIF Extract] → Metadata (GPS, camera, timestamp)
    ↓
2. [Face Detect] → Bounding boxes + confidence (InsightFace)
    ↓
3. [Face Embed] → 512-dim embeddings (ArcFace)
    ↓
4. [VL Describe] → Natural language description (Qwen2.5-VL-7B)
    ↓
5. [VL OCR] → Text extraction from image
    ↓
6. [VL Structured] → JSON extraction (tables, forms)
    ↓
7. [CLIP Embed] → 512-dim visual similarity vector
    ↓
[SQLite Storage] → Photo + Face records
    ↓
[ChromaDB Storage] → CLIP embeddings for semantic search
```

### Pipeline Initialization

**Implementation**: `src/vision/pipeline.py`

<details>
<summary>Code: Pipeline initialization</summary>

```python
from src.vision.pipeline import get_pipeline
from src.vision.models import AnalyzerType

pipeline = get_pipeline()
pipeline.initialize([
    AnalyzerType.FACE_DETECT,
    AnalyzerType.FACE_EMBED,
    AnalyzerType.VL_DESCRIBE,
    AnalyzerType.CLIP_EMBED,
])
```

</details>

### Analyzer Types

**Defined in** `src/vision/models.py`:

<details>
<summary>Code: AnalyzerType enum definition</summary>

```python
class AnalyzerType(str, Enum):
    FACE_DETECT = "face_detect"        # InsightFace detection
    FACE_EMBED = "face_embed"          # ArcFace 512-dim embeddings
    VL_DESCRIBE = "vl_describe"        # Qwen2.5-VL-7B description
    VL_OCR = "vl_ocr"                  # Text extraction
    VL_STRUCTURED = "vl_structured"    # JSON extraction
    EXIF_EXTRACT = "exif_extract"      # EXIF metadata
    CLIP_EMBED = "clip_embed"          # CLIP visual embeddings
```

</details>

### Single Image Analysis

<details>
<summary>Code: Analyzing a single image with all analyzers</summary>

```python
from PIL import Image

# Analyze with all initialized analyzers
result = pipeline.analyze(
    image=Image.open("/path/to/photo.jpg"),
    vl_prompt="Describe this photo in detail.",
    store_results=True,  # Save to SQLite/ChromaDB
    return_crops=False,  # Optionally return face crops
)

print(f"Image ID: {result.image_id}")
print(f"Description: {result.description}")
print(f"Faces: {len(result.faces)}")
for face in result.faces:
    print(f"  Face {face.face_id}: {face.confidence:.2f} confidence")
    if face.person_id:
        print(f"    Identified as: {face.person_id}")

if result.exif:
    print(f"Taken at: {result.exif.taken_at}")
    print(f"Camera: {result.exif.camera_model}")
    if result.exif.gps_lat:
        print(f"Location: {result.exif.gps_lat}, {result.exif.gps_lon}")
```

</details>

### Batch Processing

<details>
<summary>Code: Batch processing a directory of photos</summary>

```python
from pathlib import Path

photos_dir = Path("/mnt/raid0/llm/photos")

for photo_path in photos_dir.glob("*.jpg"):
    result = pipeline.analyze(
        image=photo_path,
        analyzers=[
            AnalyzerType.FACE_DETECT,
            AnalyzerType.EXIF_EXTRACT,
            AnalyzerType.CLIP_EMBED,
        ],
        store_results=True,
    )
    print(f"Processed {photo_path.name}: {len(result.faces)} faces")
```

</details>

### Vision Model Servers

| Port | Model | Analyzer | Use Case |
|------|-------|----------|----------|
| 8086 | Qwen2.5-VL-7B-Q4_K_M | `vl_describe`, `vl_ocr` | Worker-level vision tasks |
| 8087 | Qwen3-VL-30B-A3B-Q4_K_M | `vl_describe` (escalation) | Complex image analysis |

<details>
<summary>Code: Direct vision API usage</summary>

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8086/v1/vision/analyze",
        json={
            "image_base64": encoded_image,
            "analyzers": ["vl_describe"],
            "vl_prompt": "Analyze this diagram.",
            "store_results": False,
        }
    )
    data = response.json()
    print(data["description"])
```

</details>

### Storage & Retrieval

**SQLite Schema** (`src/db/models/vision.py`):

<details>
<summary>Data: SQLite schema and semantic search queries</summary>

```sql
CREATE TABLE photos (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    hash TEXT UNIQUE,
    description TEXT,
    width INTEGER,
    height INTEGER,
    taken_at TIMESTAMP,
    camera TEXT,
    location_lat REAL,
    location_lon REAL
);

CREATE TABLE faces (
    id TEXT PRIMARY KEY,
    photo_id TEXT REFERENCES photos(id),
    person_id TEXT,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    confidence REAL,
    embedding_id TEXT  -- Reference to ChromaDB
);
```

**Semantic Search** (CLIP embeddings in ChromaDB):

```python
from src.vision.search import search_similar_images

results = search_similar_images(
    query="sunset over mountains",
    limit=20,
)

for result in results:
    print(f"{result.path}: {result.score:.3f} similarity")
```

</details>

</details>

## Performance Benchmarks

Both pipelines have been profiled on the EPYC 9655. The document pipeline's headline number is the 19x speedup on born-digital PDFs via the fast-path router. Vision processing tops out at about 1.2 images/s end-to-end with 4 workers.

<details>
<summary>Detailed benchmark tables</summary>

### Document Processing

| Task | Method | Latency | Speedup |
|------|--------|---------|---------|
| Born-digital PDF (50 pages) | pdftotext | 120ms | 19x |
| Scanned PDF (50 pages) | LightOnOCR | 2.3s | 1.0x |
| Figure extraction (10 figures) | PyMuPDF | 850ms | — |
| Figure description (10 figures) | VL model | 12.4s | — |

### Vision Processing

| Task | Model | Latency | Throughput |
|------|-------|---------|------------|
| Face detection | InsightFace | 42ms | 24 faces/s |
| Face embedding | ArcFace | 18ms | 56 faces/s |
| VL description | Qwen2.5-VL-7B | 3.1s | 0.32 images/s |
| CLIP embedding | CLIP-ViT-B/32 | 65ms | 15 images/s |

**Batch throughput** (4 workers): 1.2 images/s end-to-end

</details>

## Operational Notes (2026-02-08)

- Vision preprocessing uses `DocumentPreprocessor` for both documents and images.
- Figure-description analysis (`FigureAnalyzer`) renders pages via PDFium and therefore applies only to PDF inputs.
- Non-PDF image inputs (png/jpg/etc.) skip figure-render analysis to avoid PDFium decode errors and continue through OCR/chunking flow.

## References

<details>
<summary>Source file references</summary>

- `src/services/document_preprocessor.py` - Main document pipeline
- `src/services/pdf_router.py` - Fast-path PDF routing
- `src/services/figure_analyzer.py` - VL figure descriptions
- `src/services/archive_extractor.py` - Multi-document archives
- `src/vision/pipeline.py` - Vision analyzer orchestration
- `src/vision/models.py` - Pydantic models for vision API

</details>

---

*Previous: [Chapter 12: Production Server Stack](12-production-server-stack.md)* | *Next: [Chapter 14: TOON Encoding](14-toon-encoding.md)*
