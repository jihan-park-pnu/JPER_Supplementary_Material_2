# Supplementary Material 2

- This repository contains the Python code (RAG_pipeline.py) and the pre-built vector data stored in Chroma database (see ChromaDB) for analysis reproducibility. 
- It also provides prompts, queries, output examples, and documentation on operational workflow.

## 1. Technical setup and configuration 

- This section describes the system requirements, environment setup, directory structure, and model parameters required to reproduce the LLM–RAG pipeline used in this study.

### 1.1. Required python libraries

- The pipeline uses the following key libraries: 

- `openai` — LLM invocation (GPT-5-mini)
- `requests` — Upstage Document Parse API call
- `langchain_community.vectorstores.Chroma` — ChromaDB client
- `langchain_community.embeddings.HuggingFaceEmbeddings` — bge-m3 embedding model
- `FlagEmbedding` — bge-reranker-v2-m3 reranker
- `PyPDF2` — PDF splitting and handling
- `pathlib` — file management
- `re` — regular expression parsing

- Recommended installation:

```bash
pip install openai langchain-community chromadb FlagEmbedding PyPDF2 requests
```

### 1.2. Directory structure

- The following folders must exist befor running the pipeline:

```bash
BASE/
│── CCAP_Action_Plan/        # Original PDFs (local government plans)
│── ChromaDB/                # Pre-built vector store persisted by Chroma
│── Output/                  # LLM outputs: extracted text, HTML, inference results
│── RAG_pipeline.py          # Main pipeline script
```

- In the script:

```python
BASE = Path(r"C:\path\to\your\project")
PDF_DIR = BASE / "CCAP_Action_Plan"
OUT_DIR = BASE / "Output"
CHROMA_DIR = BASE / "ChromaDB"
OUT_DIR.mkdir(parents=True, exist_ok=True)
```

### 1.3. API keys and model versions

- Require `UPSTAGE_API_KEY` and `OPENAI_API_KEY` environment variable. 
- Following APIs are required:
  - Upstage Document Parse (document-parse-250618)
  - OpenAI GPT-5-mini (2025-08-07)

- Re-ranker configuration:

```python
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)
```

- Model names used:

```python
LLM_MODEL = "gpt-5-mini-2025-08-07"
EMB_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
```

- Embedding initialization:

```python
emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)
```

## 2. Prompt and query
### 2.1. Information extraction 

```text
# Extract the text exactly as written in the document. Do not paraphrase, rewrite, or modify wording.
# Do not infer, assume, or supplement any information that is not explicitly stated in the document.
# Exclude all labels, numbering, bracketed codes, and formatting markers from outputs; return only the descriptive text.

# Objective/Action classification rules:
## Only the sections in the document’s tables or main text that are explicitly marked as “추진전략” should be identified as {objective}.
## Only the sections in the document’s tables or main text that are explicitly marked as “실천과제” should be identified as {action}.
## Only structural labels (e.g., 추진전략, 실천과제) determine classification; Do not classify objectives and actions based on semantic meaning, wording style, and phrasing.

# Maladaptation definition:
## Maladaptation arises from unintended trade-offs created by implementing an action to achieve its objective—such as harms imposed on other policy goals, social groups, or spatial areas.
## Do not classify background problems, general negative conditions, or implementation challenges (e.g., costs, burdens, resource shortages) as maladaptation.
## Do not infer maladaptation unless explicitly stated; if no maladaptation is explicitly mentioned for an action, output “(Missing)”.

# Extract maladaptation risks only for each {action} in relation to its corresponding {objective}.
# Attach maladaptation output under each action, but treat maladaptation as occurring at the objective–action pair level.

Output Format (must follow this structure strictly):
# Objective: {objective}
## Action: {action}
## Maladaptation risks: ...
(Repeat this block for all objectives)

=== DOCUMENT START ===
{document_text}
=== DOCUMENT END ===
```
### 2.2. Retrieval query 

```text
Maladaptation from implementing '{objective}' via measure '{action}'
```

### 2.3. Inference 

```text
# Your task is to infer maladaptation that may arise when achieving the given {objective} through its {action}.

# Maladaptation definition: 
## Maladaptation arises from unintended trade-offs created by implementing an action to achieve its objective—such as harms imposed on other policy goals, social groups, or spatial areas.
## Do not classify background problems, general negative conditions, or implementation challenges (e.g., costs, burdens, resource shortages) as maladaptation.
                 
# Using only the contextual evidence provided below, infer maladaptation risks for each objective–action pair.
# If no evidence supports a maladaptation, write: "(No evidence-based maladaptation found)"
                 
---
Objective: {objective}
                    
Actions: {action}

Contextual Evidence: {context_text}
---
                 
# Instructions:
## 1. Write ONE concise paragraph describing an evidence-supported maladaptation.
## 2. End with an inline citation: (Evidence: Author, Year).
## 3. If no evidence supports a maladaptation risk, output only: "(No evidence-based maladaptation found)"
## 4. Respond in English.
                        
Output format:
# Inferred risk for: {objective} – {action}
[Paragraph OR (No evidence-based maladaptation found)]
```
