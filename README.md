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

- Inference: 

```text
# Inferred risk for: 기후변화로부터 구민 건강 보호 – 취약계층 중심 건강관리 강화
Focusing adaptation resources on fortifying health services for designated vulnerable groups can be maladaptive if implemented narrowly or as short‑term clinical fixes: it may substitute or transfer risks to other social groups or sectors, impose opportunity costs and path dependence by locking funding and capacity into immediate care rather than cross‑sectoral prevention and resilience, and thereby increase long‑term vulnerability or inequities instead of reducing them (Evidence: Chi et al., 2021).

# Inferred risk for: 기후변화로부터 구민 건강 보호 – 감염병 예방 및 신속 대응체계 강화
Focusing investment on technocratic, rapid-response infectious disease systems (surveillance, stockpiles, emergency teams) without integrated, long-term, multisectoral planning can produce maladaptation by creating path-dependencies and shifting or substituting risk—diverting resources from upstream determinants (water, housing, equitable services), transferring vulnerabilities onto marginalized groups or other places, and locking in high-impact practices that reduce flexibility under uncertain future climates; this risk of shifting/creating new vulnerabilities and exacerbating inequities is highlighted as a common maladaptive outcome when adaptation is narrowly framed or not assessed across spatiotemporal scales (Evidence: Chi et al., 2021; Turner et al., 2024; Pourzand et al., 2023).

# Inferred risk for: 구민 안전 확보 및 피해 최소화 – 폭염으로부터 안전한 생활환경 조성
Creating heat-safe environments (e.g., large-scale provision of air-conditioning, hard cooling infrastructure, or top-down retrofit programmes) can unintentionally erode sustainable development by increasing energy use and GHG emissions, generate path dependencies and sunk costs in centralized solutions, and disproportionately burden or shift vulnerability onto low-income or less mobile residents who cannot afford operation/maintenance or are excluded from benefits; such measures can also create a false sense of security that encourages risky settlement or investment patterns and reduce incentives to build broader adaptive capacity (Evidence: Juhola et al., 2016; Chi et al., 2020; Neset et al., 2019).

# Inferred risk for: 구민 안전 확보 및 피해 최소화 – 체계적 풍수해 대응 관리
A systematic, government-led flood/wind disaster response that emphasizes hard infrastructure (elevated dikes, pumps, raised roads), emergency relief and compensation can create a false sense of security and path dependency that reduces residents' incentives to pursue independent adaptive actions or evacuate, encourages development and exposure behind defenses, shifts vulnerability onto poorer households who cannot afford private measures (e.g., foundation elevation), risks sunk costs if future conditions differ, and may erode sustainable development via increased emissions from repeated hard-structure works—thereby increasing long-term vulnerability despite short-term risk reduction (Evidence: Chi et al., 2020).

# Inferred risk for: 구민 안전 확보 및 피해 최소화 – 미세먼지 대응 강화
Strengthening fine‑dust responses via measures such as subsidized masks or air purifiers, centralized filtration infrastructure, and internet‑based alert systems may inadvertently create maladaptation by (1) shifting responsibility and reducing individual and community incentives for longer‑term emission reductions (dependency on government relief/subsidies), (2) disproportionately benefiting wealthier or digitally connected residents who can access or afford home filtration and online warnings while leaving the most vulnerable exposed, (3) locking public budgets into energy‑intensive, capital‑heavy technologies that raise operating emissions and risk becoming sunk costs or path‑dependent investments if pollution patterns change, and (4) excluding groups (e.g., elderly, low‑income) from web‑based early warnings—outcomes consistent with documented pathways of rebounding/shifting vulnerability, eroding sustainability, opportunity costs, and path dependence (Evidence: Chi et al., 2020).

# Inferred risk for: 재해로부터 안전한 산림환경 구축 – 산림종합방제 시스템 구축
Establishing a centralized, techno‑centric 산림종합방제 system risks maladaptation by reinforcing path dependence on suppression-oriented infrastructure and expertise, incurring high opportunity costs that divert resources from landscape‑level, community‑led measures (e.g., extensive grazing, traditional burning, local stewardship) and by narrowly framing adaptation around hazard control rather than broader forest values—outcomes that can shift or increase vulnerability for local groups and erode long‑term sustainable development and adaptive capacity (Evidence: Findlater et al., 2021).

# Inferred risk for: 재해로부터 안전한 산림환경 구축 – 기후변화 적응을 위한 산림 확대
Expanding forests as a climate-adaptation measure can be maladaptive if it creates path dependence and opportunity costs—for example, heavy investment in species-specific or genomics-based reforestation can lock governance and management into narrow, status-quo trajectories that foreclose alternative land uses and broader forest values, reduce future flexibility to respond to changing ecological, social or market conditions, shift vulnerabilities onto other groups or ecosystem components, and even produce technical failure when planted stock is mismatched to future conditions, thereby increasing rather than reducing risk (Evidence: Findlater, 2021).

# Inferred risk for: 안정적 물이용 체계 확보 – 안전한 물 공급 및 깨끗한 수자원 관리
Implementing actions to secure safe, clean water (e.g., environmental flows, environmental works/measures, water buybacks or large supply infrastructure) can be maladaptive by creating trade‑offs that shift or increase vulnerability: environmental flows and connectivity measures can spread invasive species and cause cold‑water pollution or mobilize saline groundwater; water buybacks and reallocations can place disproportionate socio‑economic burdens on irrigation communities; and investment in certain engineered or single‑source solutions can lock in path‑dependencies, reduce incentives for demand management, concentrate scarce restoration funds in already intact habitats, and even increase emissions via energy‑intensive options—thereby undermining broader resilience and sustainable development (Evidence: Lukasiewicz et al., 2016).

# Inferred risk for: 기후변화 대응 역량 강화 – 저탄소 생활 실천 활성화
Promoting low‑carbon lifestyle measures in a top‑down, non‑contextual way can be maladaptive by shifting or rebounding vulnerability onto low‑income or marginalized groups through higher opportunity and implementation costs, by creating path‑dependency and high opportunity costs from lock‑in to particular technologies or practices, and by producing negative externalities (e.g., environmental impacts or upstream GHG increases) that erode sustainable development when social contexts and multiple scales are ignored. (Evidence: Juhola et al., 2016; Zango‑Palau et al., 2024; Neset et al., 2019).
```

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
- Since local governments frequently employ different expressions for equivalent hierarchical concepts, {objective} and {action}, these labels were revised for each document to reflect its specific wording.

# Supplementary Material 3

| No. | Author(s) | Article Title | Journal | DOI |
|-----|---------|----------------|---------------|-----------|
| 1 | Akamani, K | The Roles of Adaptive Water Governance in Enhancing the Transition towards Ecosystem-Based Adaptation | WATER | http://dx.doi.org/10.3390/w15132341 |
| 2 | Althor, G; Mahood, S; Witt, B; Colvin, RM; Watson, JEM | Large-scale environmental degradation results in inequitable impacts to already impoverished communities: A case study from the floating villages of Cambodia | AMBIO | http://dx.doi.org/10.1007/s13280-018-1022-2 |
| 3 | Alvarez, S; Larkin, SL; Ropicki, A | Optimizing provision of ecosystem services using modern portfolio theory | ECOSYSTEM SERVICES | http://dx.doi.org/10.1016/j.ecoser.2017.07.016 |
| 4 | Bisaro, A | Coastal adaptation through urban land reclamation: Exploring the distributional effects | Journal of the Geographical Society of Berlin | http://dx.doi.org/10.12854/erde-2019-453 |
| 5 | Ciplet, D; Roberts, JT; Khan, M | The Politics of International Climate Adaptation Funding: Justice and Divisions in the Greenhouse | GLOBAL ENVIRONMENTAL POLITICS | http://dx.doi.org/10.1162/GLEP_a_00153 |
| 6 | Hurlimann, A; Barnett, J; Fincher, R; Osbaldiston, N; Mortreux, C; Graham, S | Urban planning and sustainable adaptation to sea-level rise | LANDSCAPE AND URBAN PLANNING | http://dx.doi.org/10.1016/j.landurbplan.2013.12.013 |
| 7 | Jesdale, BM; Morello-Frosch, R; Cushing, L | The Racial/Ethnic Distribution of Heat Risk-Related Land Cover in Relation to Residential Segregation | ENVIRONMENTAL HEALTH PERSPECTIVES | http://dx.doi.org/10.1289/ehp.1205919 |
| 8 | Rudge, K | Leveraging critical race theory to produce equitable climate change adaptation | NATURE CLIMATE CHANGE | http://dx.doi.org/10.1038/s41558-023-01690-9 |
| 9 | Silao, ZH; Ocloo, XS; Chapman, M; Hunt, L; Stenger, K | Trends in agroforestry research over 4 decades | ELEMENTA-SCIENCE OF THE ANTHROPOCENE | http://dx.doi.org/10.1525/elementa.2022.00151 |
| 10 | Ziaja, S | How Algorithm-Assisted Decision Making Is Influencing Environmental Law and Climate Adaptation | ECOLOGY LAW QUARTERLY | http://dx.doi.org/10.15779/Z38086363B |
