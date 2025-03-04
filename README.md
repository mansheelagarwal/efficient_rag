# [INSERT PROJECT TITLE HERE]

## 1. Generate Distractor Choices

To save time and storage when setting up the corpus, we generate distractor context for each query. The prompt to generate the context is as shown:

```
prompt = (
    f"Generate {k} completely unrelated, misleading long contexts for the following question: \"{original_query}\". "
    "Each snippet should be a plain-text, Wikipedia-like excerpt about a random, generic topic (e.g., history, science, art) that is entirely irrelevant to the question. "
    "Do not mention any keywords related to the original query (or similar terms). "
    "Each snippet should be at least 4 SENTENCES LONG and must not reference the question or its subject in any way. "
    "Do not include numbering, bullet points, extra characters, headers, or any extra labels. "
    "Do not include numbered lists or any type of ordered lists. "
)
```

We use the `"meta-llama/Llama-2-7b-chat-hf"` model provided in huggingface.
To create the distractor choices, run this:

```
cd xRAG/src/eval
python run_eval_multi_query.py \
    --data triviaqa_topk \
    --model_name_or_path Hannibal046/xrag-7b \
    --use_rag \
    --create_distractors \
    --save_dir xRAG/data/eval/triviaqa/retrieval/syn
```