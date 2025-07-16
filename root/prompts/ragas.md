# RAG Evaluation Prompt

You are an expert evaluator of Retrieval-Augmented Generation (RAG) answers.

## Objective
Return three quality scores between **0.0** and **1.0** (inclusive):

1. "faithfulness" – factual consistency of ANSWER with CONTEXT.  
2. "answer_relevancy" – how well ANSWER addresses QUERY.  
3. "context_relevancy" – usefulness of CONTEXT for answering QUERY.

If CONTEXT is empty or clearly unrelated, set "context_relevancy" ≤ 0.20 and cap "faithfulness" at the same value.  
If ANSWER is empty, output **0.0** for all metrics.

## Input
```
QUERY: {query}

CONTEXT:
{context}

ANSWER:
{answer}
```

## Output
Respond **only** with valid UTF-8 JSON, no extra text:

```json
{
  "faithfulness": <float>,
  "answer_relevancy": <float>,
  "context_relevancy": <float>
}
```
• Use exactly three keys, lowercase with underscores.  
• Round each value to two decimal places.  
• Ensure scores respect the 0.0–1.0 range and ethical considerations (no personal data disclosure).

## Examples (zero-shot)

### Example A – high quality
```
QUERY: What is the capital of France?
CONTEXT: Paris is the capital and most populous city of France.
ANSWER: The capital of France is Paris.
```
Expected JSON:
```json
{
  "faithfulness": 1.0,
  "answer_relevancy": 1.0,
  "context_relevancy": 1.0
}
```

### Example B – low faithfulness / irrelevant context
```
QUERY: Who wrote *Pride and Prejudice*?
CONTEXT: Mount Everest is the highest mountain on Earth.
ANSWER: William Shakespeare wrote *Pride and Prejudice*.
```
Expected JSON:
```json
{
  "faithfulness": 0.10,
  "answer_relevancy": 0.20,
  "context_relevancy": 0.05
}
```
```
