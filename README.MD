# MRC Question Answering System

This repository implements a Machine Reading Comprehension (MRC) system using Transformer-based NLP models. The `MRCSystem` class processes text inputs to answer specific questions about the context, returning extracted details like subject, object, action, time, and location based on a predefined confidence threshold.

## Features
- **Text Alignment & Preprocessing:** Automatically aligns and formats text for analysis, including custom character mapping.
- **Tokenization & Collation:** Efficient tokenization and collation of inputs with support for padding and attention masks.
- **Extractive Question Answering (QA):** Extracts the most relevant answers from a provided context using start and end logits.
- **Configurable Confidence Threshold:** Filters answers based on confidence score to improve result reliability.

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/Jester6136/NewsScope.git
    cd NewsScope
    ```

2. **Install required packages:**
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

3. **Download Pretrained Models**  
   Place your pretrained tokenizer and model in the `models` folder and adjust the paths in the code:
    - Tokenizer path: `jester6136/NewsScope`
    - Model checkpoint: `jester6136/NewsScope`

## Usage
```
python infer_batch.py
```

## Limitation

Sorry about the messy source code. I only cleaned up infer_batch.py to ensure it runs properly.