# PDF Processor

This is an example of a PDF processor that uses the `pdf_processor` package.

## Installation

```bash
cd examples/pdf_processor
pip install requirements.txt
```

## Usage

### Run mock tests:

```bash
python -m pytest tests/unit/test_mock_pdf_downloader.py -v
```

### Run real tests:

```bash
python -m pytest tests/unit/test_pdf_downloader.py -v
python -m pytest tests/unit/test_real_downloads.py -v
```

## pdf -> vector store pipeline:

```bash
python tools/pdf_to_text.py
python tools/json_to_index.py
python tools/search.py
```

