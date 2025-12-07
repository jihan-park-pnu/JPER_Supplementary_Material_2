# Supplementary Material 2

- This repository contains the Python code (RAG_pipeline.py) and the pre-built vector data stored in Chroma database (see ChromaDB) for analysis reproducibility. 
- It also provides prompts, queries, output examples, and documentation on operational workflow.

## A. Technical setup and configuration 

- This section describes the system requirements, environment setup, directory structure, and model parameters required to reproduce the LLM–RAG pipeline used in this study.

### A.1. Required python libraries

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

### A.2. Directory structure

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

### A.3. API keys and model versions

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

## B. Prompt and query
### B.1. Information extraction 

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
### B.2. Retrieval query 

```text
Maladaptation from implementing '{objective}' via measure '{action}'
```

### B.3. Inference 

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

## C. Output
### C.1. Output format instruction in prompt

- In this study, we instructed the model (prompt) to produce a hierarchical output structure as follows.
- Information extraction:

```text
Output Format (must follow this structure strictly):
# Objective: {objective}
## Action: {action}
## Maladaptation risks: ...
(Repeat this block for all objectives)
```

- Inference:

```text
Output format:
# Inferred risk for: {objective} – {action}
[Paragraph OR (No evidence-based maladaptation found)]
```

### C.2. Output example: Geumjeong-gu, Busan

- The full results for the Geumjeong-gu, Busan case presented in Section 4.1 are shown below.
- This serves as a sample illustrating that the same output structure is consistently produced across other local governments.
- Extraction: 
<details>
<summary><strong>Click to view full prompt</strong></summary>

```text
# Objective: 기후변화로부터 구민 건강 보호
## Action: 취약계층 중심 건강관리 강화
## Maladaptation risks: (Missing)

## Action: 감염병 예방 및 신속 대응체계 강화
## Maladaptation risks: (Missing)

# Objective: 구민 안전 확보 및 피해 최소화
## Action: 폭염으로부터 안전한 생활환경 조성
## Maladaptation risks: (Missing)

## Action: 체계적 풍수해 대응 관리
## Maladaptation risks: (Missing)

## Action: 미세먼지 대응 강화
## Maladaptation risks: (Missing)

# Objective: 재해로부터 안전한 산림환경 구축
## Action: 산림종합방제 시스템 구축
## Maladaptation risks: (Missing)

## Action: 기후변화 적응을 위한 산림 확대
## Maladaptation risks: (Missing)

# Objective: 안정적 물이용 체계 확보
## Action: 안전한 물 공급 및 깨끗한 수자원 관리
## Maladaptation risks: (Missing)

# Objective: 기후변화 대응 역량 강화
## Action: 저탄소 생활 실천 활성화
## Maladaptation risks: (Missing)
```
</details>

- Inference: 
<details>
<summary><strong>Click to view full prompt</strong></summary>

```text
# Inferred risk for: 시민 건강보호 – 전염병 관리대책 강화
Strengthening infectious-disease control as a narrowly framed, technical response risks maladaptation by diverting limited funding, staff and political attention from broader climate–health integration and other public-health priorities, reinforcing path-dependent, technocratic interventions, undermining intersectoral governance and local capacity, and disproportionately burdening vulnerable groups—thereby increasing future vulnerability and eroding equitable, sustainable health outcomes [Evidence: Juhola, 2016; Findlater, 2021; Turner, 2024].

# Inferred risk for: 시민 건강보호 – 전염병 관리대책 강화
Strengthening infectious-disease control as a narrowly framed, technical response risks maladaptation by diverting limited funding, staff and political attention from broader climate–health integration and other public-health priorities, reinforcing path-dependent, technocratic interventions, undermining intersectoral governance and local capacity, and disproportionately burdening vulnerable groups—thereby increasing future vulnerability and eroding equitable, sustainable health outcomes [Evidence: Juhola, 2016; Findlater, 2021; Turner, 2024].

# Inferred risk for: 시민 건강보호 – 전염병 관리대책 강화
Strengthening infectious-disease control as a narrowly framed, technical response risks maladaptation by diverting limited funding, staff and political attention from broader climate–health integration and other public-health priorities, reinforcing path-dependent, technocratic interventions, undermining intersectoral governance and local capacity, and disproportionately burdening vulnerable groups—thereby increasing future vulnerability and eroding equitable, sustainable health outcomes [Evidence: Juhola, 2016; Findlater, 2021; Turner, 2024].
```
</details>

## D. Operational workflow: End-user configuration 
### D.1. Section extraction parameters
- Users must know beforehand which page the search should start from and which keywords appear in the document. 
- The model begins scanning after the specified page and extracts the section once the keyword is detected, meaning that some prior knowledge is required.
- Used for target-section retrieval:

```python
min_page_threshold = 150
```

- Korean keywords: 

```python
start_kw = "부문별세부시행계획"
end_kw   = "계획의집행및관리"
```

- When the planning document changes, the keywords and the search starting point must be adjusted manually.

### D.2. Section extraction parameters
- The labels that correspond to {objective} and {action} must be specified for each document.
- In this study, the prompt explicitly specified which keywords in the document should be interpreted as {objective} and {action}:

```text
# Objective/Action classification rules:
## Only the sections in the document’s tables or main text that are explicitly marked as “추진전략” should be identified as {objective}.
## Only the sections in the document’s tables or main text that are explicitly marked as “실천과제” should be identified as {action}.
## Only structural labels (e.g., 추진전략, 실천과제) determine classification; Do not classify objectives and actions based on semantic meaning, wording style, and phrasing.
```
- Since local governments frequently employ different expressions for equivalent hierarchical concepts, i.e. {objective} and {action}, these labels were revised for each document to reflect its specific wording.
