# ==========================================================
# 1. 기본 설정
# ==========================================================

import os
import re
from pathlib import Path
import requests
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from FlagEmbedding import FlagReranker
from PyPDF2 import PdfReader, PdfWriter

# ====== 경로 설정 ======
BASE = Path(r"C:\Users\RDPL-005\Desktop")
PDF_DIR = BASE / "CCAP"              # 원본 PDF 폴더
OUT_DIR = BASE / "Parsed_Preview"    # Passer 결과 저장 폴더
CHROMA_DIR = BASE / "Chroma_Index"   # RAG용 Chroma 인덱스 폴더
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== API 키 ======
UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY", "up_**********tslL") 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-_1**********A")
LLM_MODEL = "gpt-5-mini-2025-08-07"

# ====== Embedding & DB 세팅 ======
EMB_MODEL = "BAAI/bge-m3"
emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
db = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=emb)

# ====== Reranker (업그레이드 버전) ======
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=False)

print("[환경 설정 완료]")
print(f"PDF 폴더: {PDF_DIR}")
print(f"Chroma DB: {CHROMA_DIR}")
print(f"모델: {LLM_MODEL}")
print(f"임베딩: {EMB_MODEL}")
print("리랭커: BAAI/bge-reranker-v2-m3")

# ==========================================================
# 2. Passer API 직접 호출 (Upstage 문서 파싱)
# ==========================================================
def parse_with_upstage(pdf_path: Path) -> str:
    url = "https://api.upstage.ai/v1/document-digitization"
    headers = {"Authorization": f"Bearer {UPSTAGE_API_KEY}"}

    with open(pdf_path, "rb") as f:
        files = {"document": f}
        payload = {
            "model": "document-parse",         
            "ocr": "auto",
            "output_formats": ["html"],
            "merge_multipage_tables": True,
            "chart_recognition": True
        }

        print(f"[UPSTAGE] {pdf_path.name} 업로드 중...")
        res = requests.post(url, headers=headers, files=files, data=payload)

    if res.status_code != 200:
        raise RuntimeError(
            f"❌ Passer API 오류 ({res.status_code})\n"
            f"응답: {res.text[:500]}..."
        )

    data = res.json()
    html = data.get("content", {}).get("html", "")
    if not html.strip():
        raise ValueError("⚠️ 결과 HTML이 비어 있음")

    out_path = OUT_DIR / f"{pdf_path.stem}_parsed.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"✅ [Passer 파싱 완료] {out_path.name} ({len(html):,} chars)")

    return html

# ==========================================================
# 3. 대용량 PDF 자동 분할 + Passer 파싱 + HTML 병합
# ==========================================================
def list_pdfs(max_n: int = 10):
    """📄 PDF 폴더 내 PDF 파일 목록 출력"""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"⚠️ PDF 없음: {PDF_DIR}")

    print(f"📄 {len(pdf_files)}개 PDF 발견:")
    for i, p in enumerate(pdf_files[:max_n], start=1):
        print(f"{i:>3}. {p.name}")
    return pdf_files

def pick_pdf_by_fragment(fragment: str) -> Path:
    """🎯 파일명 일부(fragment)로 특정 PDF 선택"""
    frag = fragment.lower()
    matches = [p for p in PDF_DIR.glob("*.pdf") if frag in p.name.lower()]
    if not matches:
        raise FileNotFoundError(f"'{fragment}'에 해당하는 PDF를 찾을 수 없습니다.")
    matches.sort(key=lambda p: len(p.name))  # 가장 짧은 이름 우선
    target = matches[0]
    print(f"🎯 선택된 PDF: {target.name}")
    return target

def split_pdf(input_path: Path, output_dir: Path, max_pages: int = 90):
    reader = PdfReader(str(input_path))
    total_pages = len(reader.pages)
    parts = (total_pages // max_pages) + (1 if total_pages % max_pages else 0)
    output_paths = []

    print(f"📘 {input_path.name} ({total_pages} pages) → {parts}개로 분할 예정")

    for i in range(parts):
        writer = PdfWriter()
        start, end = i * max_pages, min((i + 1) * max_pages, total_pages)
        for j in range(start, end):
            writer.add_page(reader.pages[j])

        out_path = OUT_DIR / f"{input_path.stem}_part{i+1}.pdf"
        with open(out_path, "wb") as f:
            writer.write(f)
        output_paths.append(out_path)
        print(f"✅ {out_path.name} 저장 ({end - start}p)")

    return output_paths

def parse_large_pdf_with_upstage(pdf_path: Path, max_pages: int = 90) -> str:
    split_paths = split_pdf(pdf_path, OUT_DIR, max_pages=max_pages)
    merged_html = ""

    for i, part_path in enumerate(split_paths, start=1):
        print(f"\n🚀 [{i}/{len(split_paths)}] {part_path.name} 파싱 중...")
        try:
            html_chunk = parse_with_upstage(part_path)
            merged_html += f"\n<!-- PART {i} START -->\n" + html_chunk + f"\n<!-- PART {i} END -->\n"
        except Exception as e:
            print(f"⚠️ {part_path.name} 파싱 실패: {e}")
            continue

    merged_path = OUT_DIR / f"{pdf_path.stem}_merged.html"
    merged_path.write_text(merged_html, encoding="utf-8")

    print(f"\n🎯 전체 병합 완료 → {merged_path.name} ({len(merged_html):,} chars)")
    return merged_html

def extract_section_by_fixed_keywords(pdf_path: Path, min_page_threshold: int = 10, max_section_pages: int = 90) -> list[Path]:
    reader = PdfReader(pdf_path)
    start_page, end_page = None, None

    start_kw = "부문별세부시행계획"
    end_kw = "계획의집행및관리"

    print(f"🔍 '{pdf_path.name}'에서 '{start_kw}' ~ '{end_kw}' 구간 탐색 중...")

    # ==== 1. 시작/끝 페이지 탐색 =====
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text_clean = re.sub(r"\s+", "", text)

        if start_page is None and i > min_page_threshold and start_kw in text_clean:
            start_page = i
            print(f"✅ 본문 시작 키워드 발견 (p.{i+1})")
        elif start_page is not None and end_kw in text_clean:
            end_page = i
            print(f"✅ 종료 키워드 발견 (p.{i+1})")
            break

    if start_page is None:
        raise ValueError(f"❌ '{start_kw}' 키워드를 {min_page_threshold+1}페이지 이후에서 찾을 수 없습니다.")
    if end_page is None:
        end_page = len(reader.pages)

    total_pages = end_page - start_page
    print(f"📄 추출 구간: p.{start_page+1}–{end_page} ({total_pages} pages)")

    # ==== 2. 90페이지 초과 시 자동 분할 =====
    section_parts = []
    for idx in range(0, total_pages, max_section_pages):
        writer = PdfWriter()
        part_start = start_page + idx
        part_end = min(start_page + idx + max_section_pages, end_page)
        for j in range(part_start, part_end):
            writer.add_page(reader.pages[j])

        part_path = OUT_DIR / f"{pdf_path.stem}_section_part{len(section_parts)+1}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)

        section_parts.append(part_path)
        print(f"✅ 섹션 부분 저장: {part_path.name} ({part_end - part_start}p)")

    print(f"📘 '부문별 세부시행계획' 섹션 {len(section_parts)}개로 분할 완료")
    return section_parts

files = list_pdfs()
target_pdf = pick_pdf_by_fragment("순천시")

# ‘부문별 세부시행계획’ 섹션만 추출 (자동 90p 단위 분할)
section_parts = extract_section_by_fixed_keywords(target_pdf, min_page_threshold=150, max_section_pages=90)

# 각 부분을 순차적으로 Passer로 파싱 및 병합
merged_html = ""
for i, section_pdf in enumerate(section_parts, start=1):
    print(f"\n🚀 [섹션 {i}/{len(section_parts)}] Passer 파싱 중...")
    html_chunk = parse_with_upstage(section_pdf)
    merged_html += f"\n<!-- SECTION {i} START -->\n" + html_chunk + f"\n<!-- SECTION {i} END -->\n"

merged_path = OUT_DIR / f"{target_pdf.stem}_section_merged.html"
merged_path.write_text(merged_html, encoding="utf-8")

print(f"\n✅ 전체 섹션 병합 완료 → {merged_path.name}")
print(f"📑 총 {len(section_parts)}개 섹션 병합 완료 ({len(merged_html):,} chars)")

# ==========================================================
# 4. LLM 기반 문서 분석 (정보 추출 단계)
# ==========================================================
def ask_llm_on_document(html_path: Path, model: str = LLM_MODEL) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)
    document_text = html_path.read_text(encoding="utf-8")
    print(f"문서 길이: {len(document_text):,} chars → 단일 분석 시작")

    # Prompt
    user_prompt = f"""
        # Extract the text exactly as written in the document. Do not paraphrase, rewrite, or modify wording.
        # Do not infer, assume, or supplement any information that is not explicitly stated in the document.
        # Exclude all labels, numbering, bracketed codes, and formatting markers from outputs; return only the descriptive text.
        
        # Objective/Action classification rules:       
        ## Only the sections in the document’s tables or main text that are explicitly marked as “추진전략” should be identified as {{objective}}.
        ## Only the sections in the document’s tables or main text that are explicitly marked as “실천과제” should be identified as {{action}}.
        ## Only structural labels (e.g., 추진전략, 실천과제) determine classification; Do not classify objectives and actions based on semantic meaning, wording style, and phrasing.
        
        # Maladaptation definition: 
        ## Maladaptation arises from unintended trade-offs created by implementing an action to achieve its objective—such as harms imposed on other policy goals, social groups, or spatial areas.
        ## Do not classify background problems, general negative conditions, or implementation challenges (e.g., costs, burdens, resource shortages) as maladaptation.
        ## Do not infer maladaptation unless explicitly stated; if no maladaptation is explicitly mentioned for an action, output “(Missing)”.
        
        # Extract maladaptation risks only for each {{action}} in relation to its corresponding {{objective}}.
        # Attach maladaptation output under each action, but treat maladaptation as occurring at the objective–action pair level.
        
        Output Format (must follow this structure strictly):
        # Objective: {{objective}}
        ## Action: {{action}}
        ## Maladaptation risks: ...
        (Repeat this block for all objectives)
        
        === DOCUMENT START ===
        {document_text}
        === DOCUMENT END ===
    """

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": user_prompt}],
    )

    final_text = resp.output_text.strip() if hasattr(resp, "output_text") else ""
    clean_stem = html_path.stem.replace("_section_merged", "")
    out_path = OUT_DIR / f"LLM_Extract_{clean_stem}.txt"
    out_path.write_text(final_text, encoding="utf-8")    

    print(f"[분석 완료] {out_path.name}")
    return final_text

html_path = OUT_DIR / f"{target_pdf.stem}_section_merged.html"
ask_llm_on_document(html_path)

# ==========================================================
# 5. Inference stage: Objective-action pair inference
# ==========================================================
def infer_missing_impacts(extracted_txt_path: Path, model: str = LLM_MODEL, top_k: int = 5):
    client = OpenAI(api_key=OPENAI_API_KEY)
    extracted_text = extracted_txt_path.read_text(encoding="utf-8")

    print(f"추론 대상 텍스트 불러옴: {extracted_txt_path.name}")

    # === 1. Objective 블록 분리 ===
    sections = re.split(r"(?=^# Objective:)", extracted_text, flags=re.MULTILINE)
    sections = [s.strip() for s in sections if s.strip()]
    print(f"총 {len(sections)}개 Objective 블록 탐지")

    total_results = []

    for sidx, sec in enumerate(sections, start=1):

        # --- Objective 추출 ---
        obj_match = re.search(r"^# Objective:\s*(.+)", sec, flags=re.MULTILINE)
        if not obj_match:
            continue
        objective = obj_match.group(1).strip()

        # --- Action + Risk 추출 ---
        pairs = re.findall(
            r"## Action:\s*(.+?)\s*## Maladaptation risks:\s*(.+?)(?=^## Action:|$)",
            sec,
            flags=re.MULTILINE | re.DOTALL,
        )

        if not pairs:
            print(f"[경고] Action–Risk 쌍 없음 → {objective} 건너뜀")
            continue

        # Missing인 action만 모으기
        missing_actions = []
        for action, risk in pairs:
            action = action.strip()
            risk = risk.strip()
            if risk == "(Missing)":
                missing_actions.append(action)

        if not missing_actions:
            continue

        # === 2. Action별 inference ===
        for aidx, action in enumerate(missing_actions, start=1):

            print(f"\n Objective 추론 시작 ({sidx}/{len(sections)}) — {objective}")
            print(f"   Action {aidx}/{len(missing_actions)}: {action}")

            # --- Query 생성 (Objective–Measure pair) ---
            query = f"Maladaptation from implementing '{objective}' via measure '{action}'."

            # --- Retrieval ---
            docs = db.similarity_search(query, k=top_k)

            # --- Rerank ---
            scores = [reranker.compute_score([query, d.page_content]) for d in docs]
            reranked_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
            top_docs = [doc for _, doc in reranked_docs[:3]]

            # --- Evidence 구성 ---
            if top_docs:
                context_chunks = []
                for i, d in enumerate(top_docs):
                    src = d.metadata.get("source", "unknown")
                    title = d.metadata.get("title", "")
                    context_chunks.append(
                        f"[{i+1}] {src if src else title}:\n{d.page_content.strip()}"
                    )
                context_text = "\n\n".join(context_chunks)
            else:
                context_text = "(No evidence)"
            
            # --- LLM 프롬프트 ---
            prompt = f"""
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
                     """

            # --- LLM 호출 ---
            resp = client.responses.create(
                model=model,
                input=[{"role": "user", "content": prompt}],
            )

            inference = resp.output_text.strip() if hasattr(resp, "output_text") else "(Inference failed)"
            total_results.append(inference + "\n")

    # === 3. 결과 저장 ===
    if not total_results:
        print("모든 항목에 Adverse Impacts가 존재합니다. 추론 불필요.")
        return None

    final_output = "\n\n".join(total_results)

    clean_stem = extracted_txt_path.stem.replace("_SINGLE", "").replace("_section_merged", "").replace("LLM_Extract_", "")
    out_path = OUT_DIR / f"LLM_Inferred_{clean_stem}.txt"
    out_path.write_text(final_output, encoding="utf-8")

    print(f"\n추론 결과 저장 완료 → {out_path.name}")
    return final_output

extracted_txt_path = OUT_DIR / f"LLM_Extract_{html_path.stem.replace('_section_merged','')}.txt"
infer_missing_impacts(extracted_txt_path)