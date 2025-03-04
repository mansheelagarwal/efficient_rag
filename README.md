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
    --save_dir ../../data/eval/triviaqa/retrieval/colbertv2
```

You can optionally use the --max_test_samples argument for debugging.

## 2. Generate Synthetic Queries and Ensemble Score

After generating your distractor choices, you can now generate `k` number of synthetic queries. Each synthetic query can now choose their own top 1 document. Afterwards, an ensemble scoring is done to select the overall top 1 document.

To do this run:

```
cd xRAG/src/eval
python run_eval_multi_query.py \
    --data triviaqa \
    --model_name_or_path Hannibal046/xrag-7b \
    --use_rag \
    --k_samples 5 \
    --retriever_name_or_path Salesforce/SFR-Embedding-Mistral \
    --ensemble_rerank \
    --save_dir ../../data/eval/triviaqa/retrieval/colbertv2
```

## 3. Evaluate Approach

```
cd xRAG/src/eval
python run_eval_multi_query.py \
    --data triviaqa \
    --model_name_or_path Hannibal046/xrag-7b \
    --use_rag \
    --k_samples 5 \
    --retriever_name_or_path Salesforce/SFR-Embedding-Mistral \
    --save_dir ../../data/eval/triviaqa/retrieval/colbertv2
```

If you defined `--max_test_samples` from step 1, please define it here as well.