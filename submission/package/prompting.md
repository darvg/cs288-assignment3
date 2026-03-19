# CS288 Assignment 3 — Recommended Prompting & Formatting Steps Addendum

This addendum is written to plug directly into the existing implementation plan for an empty repo. It focuses on **prompting and evidence-formatting choices** that are worth adding based on the assignment constraints and the useful parts of the FRESHLLMS paper.

---

## Why add these steps

These additions are worth implementing because the assignment is a **short-answer, single-page, closed-domain QA task** evaluated with **Exact Match** and **token-level F1**, and the paper finds that **concise/direct answers**, **evidence formatting**, and **evidence count/order** materially affect answer quality and hallucination.

The main adaptation is that your repo should use these ideas in a **lighter, CPU-friendly, closed-domain way**:

- keep answers very short;
- pass structured evidence rather than raw chunk dumps;
- ablate evidence count and ordering;
- avoid long chain-of-thought or verbose demonstrations by default.

---

## Where to insert this in the implementation plan

Add the following items to the plan under:

1. **Generation stack**
2. **Evaluation / ablations**
3. **Error analysis**
4. **Report-support artifacts**

---

## New implementation steps to add

### Step A — Add a strict short-answer prompt contract

Add a new sub-step in the generation pipeline that defines a **hard output contract** for the model.

#### What to implement

Create a prompt template that always instructs the model to:

- answer using only the provided evidence;
- return **only the final answer**;
- return **no explanation**;
- return **no newline characters**;
- keep the answer **as short as possible**;
- return exactly `Yes` or `No` for yes/no questions.

#### Why this is worth adding

This aligns with the assignment’s short-answer evaluation format and reduces the risk of verbose outputs hurting EM/F1.

#### Suggested task for Codex

- Add `src/inference/prompting.py`
- Add a `build_prompt(question, evidence_blocks, question_type=None)` function
- Add a `postprocess_answer(raw_text)` function that:
  - strips whitespace;
  - keeps only the first line;
  - collapses repeated spaces;
  - removes obvious answer prefixes like `Answer:`;
  - normalizes yes/no outputs to exactly `Yes` or `No`.

#### Recommended default instruction block

```text
You are answering factoid questions about UC Berkeley EECS using only the provided evidence.

Return only the final answer.
Do not explain.
Do not include extra words.
Do not include newline characters.
If multiple evidence chunks disagree, prefer the most directly supported answer from the most relevant chunk.
If the answer is yes/no, return exactly "Yes" or "No".
```

---

### Step B — Format retrieved evidence as structured blocks

Add a new step so retrieved chunks are not passed to the LLM as an unstructured text blob.

#### What to implement

For each retrieved chunk, format it like this:

```text
[Evidence 1]
Title: <page title>
URL: <page url>
Section: <section heading path>
Text: <chunk text>
```

#### Why this is worth adding

This gives the model lightweight structure similar to the evidence formatting strategy from the paper, but adapted for a closed EECS corpus.

#### Suggested task for Codex

- Add `format_evidence_block(chunk)` in `src/inference/prompting.py`
- Ensure every chunk object includes:
  - `title`
  - `url`
  - `section_path`
  - `text`
- Keep evidence formatting lightweight; do **not** add long metadata fields that increase prompt length without obvious value.

#### Recommendation

Use:

- title
- URL
- section heading
- chunk text

Do **not** include:

- verbose source descriptions
- redundant metadata
- long boilerplate around each evidence block

---

### Step C — Add evidence ordering as a configurable retrieval-to-prompt stage

Add a step between retrieval and generation that explicitly controls **which order** retrieved chunks appear in the prompt.

#### What to implement

Support at least these prompt evidence orderings:

1. retrieval-score descending
2. retrieval-score ascending
3. page order / section order when available
4. random order (ablation only)

#### Why this is worth adding

The paper reports that evidence order matters. In your task, score-descending is the best default, but testing ordering is an easy and worthwhile ablation.

#### Suggested task for Codex

- Add `order_evidence(chunks, strategy="score_desc")`
- Store retrieval scores and section order metadata
- Default to `score_desc`

---

### Step D — Add evidence-count control and ablations

Add a configurable cap on how many retrieved chunks are passed into the final prompt.

#### What to implement

Add config values for:

- `num_bm25_candidates`
- `num_dense_candidates`
- `num_fused_candidates`
- `num_generator_evidence`

Recommended initial values:

- BM25 candidates: 8
- dense candidates: 8
- fused candidates: 6
- generator evidence: 4

#### Why this is worth adding

The paper finds that number of evidence items matters. In your assignment, this should be treated as a first-class ablation because too much context can hurt answer extraction and increase latency.

#### Suggested task for Codex

Add an ablation runner over:

- top 2 evidence blocks
- top 4 evidence blocks
- top 6 evidence blocks

Record:

- EM
- F1
- retrieval recall@k
- average latency per question

---

### Step E — Add a “shortest-supported-answer” fallback rule

Add a deterministic fallback pass after generation.

#### What to implement

When the generated answer is suspiciously long or malformed, apply a fallback policy:

1. if the answer contains multiple sentences, keep the first line only;
2. if it still looks verbose, try extracting the shortest high-overlap span from the top evidence chunk;
3. if timeout/error occurs, return a placeholder fallback answer.

#### Why this is worth adding

This reduces the chance that a mostly-correct answer is ruined by extra words or formatting noise.

#### Suggested task for Codex

- Add `is_verbose_answer(text)` heuristic
- Add `extractive_fallback(question, top_chunk)` heuristic
- Keep this lightweight and deterministic

---

### Step F — Avoid chain-of-thought by default

Add an explicit implementation rule that the default prompt should **not** request reasoning traces.

#### What to implement

Do **not** ask for:

- “think step by step”
- visible reasoning
- long explanations
- multi-example demonstrations by default

#### Why this is worth adding

The assignment rewards short exact answers, and long generated reasoning increases token cost, latency, and the chance of answer-format violations.

#### Suggested task for Codex

- Add one compact default prompt with **no reasoning request**
- Optionally add one ablation prompt with minimal demonstration examples, but do not make it the default

---

### Step G — Add a tiny prompt-ablation suite

Add prompt configuration variants so prompting choices can feed directly into the report.

#### What to implement

Create these prompt variants:

1. `strict_short`
   - only concise direct answer instructions
2. `strict_short_structured_evidence`
   - concise answer + structured evidence blocks
3. `strict_short_plus_shortest_span_rule`
   - concise answer + instruction to prefer shortest directly supported span
4. `strict_short_with_demo`
   - same as above, but with one very compact demonstration example

#### Recommended default

Use `strict_short_structured_evidence` as the default.

#### What to log

For each variant, log:

- EM
- F1
- retrieval recall@k
- average latency
- output-format violation count

---

### Step H — Add output-format validation before writing predictions

Add a final output-safety check before each prediction is written to the output file.

#### What to implement

Before writing each prediction:

- ensure it is a single line;
- strip tabs/newlines;
- replace empty output with fallback placeholder;
- optionally clip extremely long outputs.

#### Suggested task for Codex

Add `sanitize_prediction_for_output(answer)` in `src/inference/answer_question.py` or `src/utils/io.py`.

This function should guarantee:

- one prediction per line
- no embedded newlines
- non-empty output

---

### Step I — Add prompt-related error analysis categories

Extend the error analysis step so it can diagnose prompt-formatting failures, not just retrieval failures.

#### Add these categories

- retrieval miss
- answer extraction miss
- verbose / over-generated answer
- wrong evidence selected from correct page
- malformed yes/no output
- false negative due to metric limitation

#### Why this is worth adding

This will make it easier to explain why a prompt change improved EM/F1 even when retrieval stayed constant.

---

### Step J — Add report-ready prompt ablation outputs

Add one script that exports a small report table specifically for prompting decisions.

#### What to implement

Export a CSV or JSON table with columns like:

- prompt_variant
- evidence_order
- num_evidence
- EM
- F1
- recall_at_k
- avg_latency_ms
- format_error_rate

#### Why this is worth adding

This gives you an easy Q3/Q4 report story:

- structured evidence helped or not;
- more evidence helped or hurt;
- concise prompting reduced formatting failures.

---

## Exact plan edits to make

Below are concise edits you can paste into the existing implementation plan.

### Add under “Generation stack”

```markdown
- Add a strict short-answer prompt contract that requires output-only answers with no explanation and no newline characters.
- Format retrieved context as structured evidence blocks with title, URL, section heading, and chunk text.
- Add configurable evidence ordering before prompt construction (default: retrieval-score descending).
- Add configurable control over number of evidence blocks passed to the generator (default: 4).
- Add deterministic postprocessing and a shortest-supported-answer fallback when the model outputs verbose or malformed text.
- Avoid chain-of-thought or verbose demonstrations by default.
```

### Add under “Evaluation / ablations”

```markdown
- Run prompt ablations over:
  - strict short-answer prompt
  - strict short-answer + structured evidence
  - strict short-answer + shortest-span instruction
  - strict short-answer + one compact demonstration
- Run evidence-count ablations over top-2 / top-4 / top-6 evidence blocks.
- Run evidence-order ablations over score-desc / score-asc / page-order / random.
- Log EM, F1, retrieval recall@k, average latency, and output-format violation rate.
```

### Add under “Error analysis”

```markdown
- Add prompt-related categories:
  - verbose over-generation
  - malformed yes/no output
  - wrong span selected despite correct evidence
  - formatting failure affecting EM/F1
```

### Add under “Report-support artifacts”

```markdown
- Export a prompt-ablation summary table.
- Export a formatting-failure summary.
- Save 5-10 example questions showing when structured evidence or shorter prompts helped.
```

---

## Recommended default configuration

If you want one safe default to start with, use this:

- hybrid retrieval
- top 4 evidence blocks to generator
- evidence order = score descending
- structured evidence format = on
- demonstrations = off
- reasoning request = off
- strict concise-answer instruction = on
- deterministic answer postprocessing = on

This is the most assignment-aligned default because it keeps answers short, reduces format errors, stays lightweight, and gives you clean ablations later.

---

## Codex-ready task block

```text
Add prompt-engineering and evidence-formatting support to the repo.

Implement the following:
1. A strict short-answer prompt template that requires output-only answers, no explanation, and no newline characters.
2. Structured evidence formatting for each retrieved chunk with Title, URL, Section, and Text fields.
3. Evidence ordering strategies: score_desc, score_asc, page_order, random.
4. Configurable number of evidence blocks passed to the generator.
5. Deterministic answer postprocessing that strips to one line, removes prefixes like `Answer:`, collapses whitespace, and normalizes yes/no outputs.
6. A lightweight fallback that prefers the shortest directly supported answer when model output is verbose or malformed.
7. Prompt ablations over concise prompt variants and evidence count/order.
8. Logging for EM, F1, retrieval recall, latency, and output-format violations.
9. Error-analysis support for prompt/formatting failure categories.
10. Report-export support for a prompt-ablation summary table.

Default behavior should be:
- no chain-of-thought
- no verbose demonstrations
- structured evidence on
- evidence count = 4
- evidence order = score_desc
- strict concise-answer instructions on
```

---

## Final recommendation

Use the paper’s prompting ideas as **small, controlled, testable upgrades** rather than as a full prompting framework. For this assignment, the most valuable changes are:

1. concise direct answer instructions,
2. structured evidence formatting,
3. evidence-count ablations,
4. evidence-order ablations,
5. strict output sanitation.

Those are highly compatible with the assignment’s hidden evaluation format, latency constraints, and report requirements.
