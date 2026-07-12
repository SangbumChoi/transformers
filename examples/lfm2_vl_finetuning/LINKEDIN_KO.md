<!--- LinkedIn post (Korean), ordered to match an image carousel. -->

# LinkedIn 게시글 (순서대로) — 이미지는 각 섹션의 🖼️ 위치에 첨부

추천 이미지 순서:
1. `linkedin_1_setup.png`  2. `linkedin_3_vision_vs_language.png`  3. `data_size_results.png`
4. `rank_sweep_5000_results.png`  5. `error_analysis_results.png`  6. `color_confusion.png`
7. `linkedin_2_results_tables.png` (부록: 전체 수치)

---

브라우저에서도 돌아가는 450M짜리 VLM을 이미지 50장도 안 되는 데이터로 파인튜닝해 봤습니다.
그런데 한 가지 질문이 머릿속을 떠나지 않았습니다 —
**"VLM이 좋아질 때, 실제로 학습되는 건 비전 인코더일까, 언어 모델일까?"**
그래서 데이터를 *그 질문에 답하도록* 설계하고, 끝까지 파고들어 봤습니다. 🧵

🖼️ [이미지 1 — 셋업: 모델 + 데이터]
• 모델: LiquidAI/LFM2-VL-450M (SigLIP2 비전 인코더 86M + LFM2 언어 모델 350M, WebGPU로 브라우저 구동).
• 데이터: Pillow로 직접 *생성*한 합성 데이터. 색·도형·공간관계를 정확히 읽어야 푸는 **"시각 이해(perception)" 태스크**로 일부러 설계했습니다 (62 colors × 17 shapes(2D+3D) × 5 relations).
• 핵심 장치: **학습에서 본 적 없는 '조합'만**으로 test set을 구성 → 암기(memorization)가 아니라 일반화(generalization)를 측정.

🖼️ [이미지 2 — 비전 vs 언어, 그리고 가중치 변화]
LoRA를 비전에만 / 언어에만 / 전체에 각각 걸어 비교했습니다.
• 비전만 학습하면 LoRA rank를 키울수록 성능이 쭉 올라갑니다 (capacity-limited).
• 언어만 학습하면 rank를 64배 키워도 거의 평평합니다 (이미 saturated).
• **가중치 변화량(‖ΔW‖/‖W‖)도 비전 쪽이 가장 크고, 정확도 향상과의 상관이 +0.95.**
→ 이 시각 태스크에서 "배우는 주체"는 비전 인코더라는 첫 신호.

🖼️ [이미지 3 — 데이터를 늘리면 천장이 올라간다]
같은 모델·같은 100장 held-out test로 학습 데이터만 늘려봤습니다.
• 50장 → 73%, 500장 → 80%, 1,500장 → 86%, 5,000장 → **93%**.
• 처음엔 train 100% / test 낮음(암기), 5,000장에서는 더 이상 외우지 못하고 **규칙을 학습**(test가 train을 추월).
→ 작은 데이터에서의 천장은 모델 한계가 아니라 **데이터 부족**이었습니다.

🖼️ [이미지 4 — 데이터가 많아지면 역할이 뒤집힌다 (rank sweep: 30장 vs 5,000장)]
• 30장(데이터 기근): 저-rank language-only가 가장 효율적, 비전은 capacity 부족으로 67%에서 막힘.
• 5,000장(데이터 풍부): **비전이 rank를 키우며 76%→94%로 1위로 올라오고, language-only는 rank를 키워도 86%에서 정체** → 최하위.
→ "어디에 LoRA 예산을 쓸지"는 **무엇이 병목이냐(데이터 vs capacity)**에 달려 있습니다.

🖼️ [이미지 5 — 왜 language-only는 정체될까? (perception 병목)]
오답을 유형별로 분해했더니:
• 세 설정 모두 **공간관계(spatial) 100% 정답** → 추론(reasoning)은 문제 아님.
• 오답은 거의 전부 **미세한 색 혼동**. 순수 지각 문제(single-shape)에서 language-only 60% < vision 71% < all 80%, 색 오류 13→10→7.
→ 인코더가 frozen이면 언어 모델은 인코더가 버린 정보를 복원할 수 없습니다. **비전을 건드려야** 풀립니다.

🖼️ [이미지 6 — 그럼 '왜 그 색들'을 못 맞췄나]
틀린 색은 전부 **RGB 공간의 최근접 이웃**이었습니다 (sapphire→cobalt, brown→chocolate, cyan↔turquoise …; Δ 22~63, 팔레트 평균 최근접 ≈33).
원인 4가지: ① 팔레트가 조밀 ② 우리가 넣은 색 jitter(≈24 RGB)가 이웃 분포를 겹치게 함(일부는 원천적으로 모호) ③ frozen SigLIP + pixel-unshuffle가 미세 색차를 압축해 버림 ④ 언어 헤드가 흔한/대표 색 이름으로 회귀(forestgreen→"green").
→ 모델이 "색을 못 배운" 게 아니라, **frozen된 눈의 해상도와 겹친 라벨이 허용하는 만큼만** 배운 것.

🖼️ [이미지 7 — 부록: 전체 수치 표]

**한 줄 결론**
시각 이해 태스크에서 병목은 **비전 인코더의 지각 해상도**였습니다. 가중치 변화·rank·데이터 스케일링이 모두 비전 쪽을 가리켰습니다. → 스케일할수록 예산은 **언어가 아니라 비전(capacity + data)** 에 쓰세요. 더 일반적으로는, *무엇이 병목인지부터 진단*하는 게 먼저입니다.

⚙️ 작은 실험은 CPU로, 큰 sweep은 Hugging Face Jobs(A10G GPU)로 — 총 GPU 비용 약 $5. 작은 모델 + 합성 데이터 = 끝까지 직접 돌려볼 수 있는, 감당 가능한 ML 실험.

#MachineLearning #LLM #VisionLanguageModels #LoRA #FineTuning #HuggingFace #LiquidAI #AI
