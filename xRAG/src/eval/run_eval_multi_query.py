## built-in
import argparse,json,os
import time

## third party
from transformers import (
    MistralForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    MixtralForCausalLM,
)
from sklearn.metrics.pairwise import cosine_similarity
import torch
import datasets
from tqdm import tqdm
import numpy as np
import re

## own
from src.model import (
    XMistralForCausalLM,
    XMixtralForCausalLM,
    SFR,
)

from src.language_modeling.utils import (
    XRAG_TOKEN,
    get_retrieval_embeds,
)
from src.eval.utils import (
    stop_sequences_criteria,
    get_substring_match_score,
    eval_fact_checking,
    eval_truthfulqa,
    keyword_extraction_with_tfidf,
)
from src.utils import (
    get_jsonl,
)

def create_prompt_with_mistral_chat_format(messages,tokenizer,*args,**kwargs):
    # return tokenizer.apply_chat_template(messages,tokenize=False,add_special_tokens=False)
    formatted_text = ""
    for message in messages:
        if message['role'] == 'user':
            formatted_text += "[INST] " + message['content'] + " [/INST]"
        elif message['role'] == 'assistant':
            formatted_text += message['content'] + tokenizer.eos_token
        else:
            raise ValueError(
                "Mistral chat template only supports 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    # formatted_text += " The answer is:"
    return formatted_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval_prefix",
        default='colbertv2'
    )
    parser.add_argument(
        "--tf_idf_topk",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--base_model",
    )
    parser.add_argument(
        "--use_rag",
        action='store_true',
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=eval,
        default=True,
    )
    parser.add_argument(
        "--data",
    )
    parser.add_argument(
        "--model_name_or_path",
    )
    parser.add_argument(
        "--eval_metrics",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--retriever_name_or_path",
    )
    parser.add_argument(
        "--retrieval_topk",
        type=int,
        default=[1],
        nargs='+',
    )
    parser.add_argument(
        "--retrieval_embed_length",
        type=int,default=0,
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="for debug",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--chat_format",
        default='mistral',
    )
    
    # ** K SAMPLES FOR SYNTHETIC PROMPTS ** #
    parser.add_argument(
        "--k_samples",
        type=int,
        default=0,
        help="Number of synthetic queries to sample."
    )
    
    # ** STORE DISTRACTOR CONTEXTS ** #
    parser.add_argument(
        "--create_distractors",
        action="store_true",
        help="If set, generate distractor choices and save to a JSONL file; otherwise, load the JSONL file."
    )
    
    # ** STORE DISTRACTOR CONTEXTS ** #
    parser.add_argument(
        "--ensemble_rerank",
        action="store_true",
        help="If set, generate synthetic queries, ensemble rerank distractor choices, and save to triviaqa_syn_ensemble.jsonl"
    )
    
    args = parser.parse_args()

    ## post-process
    if args.data in ['nq_open','hotpotqa','triviaqa','webqa']:
        args.task_type = 'open_qa'
        args.eval_metrics = 'substring_match'
    elif args.data in ['truthfulqa']:
        args.task_type = 'open_qa'
        args.eval_metrics = 'truthfulqa_f1_rl'
    elif args.data in ['factkg']:
        args.task_type = 'fact_checking'
        args.eval_metrics = 'fact_checking_acc'
    
    args.retrieval_topk = [x-1 for x in args.retrieval_topk] ## rank starts from 1
    
    if args.chat_format is not None:
        args.chat_format = eval(f"create_prompt_with_{args.chat_format}_chat_format")    
    
    if args.retriever_name_or_path is not None:
        args.use_rag = True

    return args



QA_PROMPT = "Question: {question}?\n"
FECT_CHECKING_PROPMT = "Claim: {question}\n"
BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\n"

PROMPT_TEMPLATES = {
    "open_qa":QA_PROMPT,
    'fact_checking':FECT_CHECKING_PROPMT,
}

# ** GENERATE EMBEDDING ** #
def get_text_embedding(text, model, tokenizer, max_length=128):

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(model.device)
    with torch.no_grad():
        embeds = get_retrieval_embeds(
            model=model,
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        
    # convert it from BFloat16 to float32 for NumPy compatibility.
    emb = embeds[0].cpu().to(torch.float32).numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


# ** RE-RANK DISTRACTORS ** #
@torch.no_grad()
def ensemble_rerank_distractors(test_data, retriever, retriever_tokenizer, max_length=128):
    updated_test_data = []
    for sample in tqdm(test_data, desc="Ensemble re-ranking distractors"):

        synthetic_queries = sample.get("synthetic_queries", [])
        if sample["query"] not in synthetic_queries:
            synthetic_queries.append(sample["query"])
        
        distractors = sample.get("background", [])

        synthetic_embeddings = []
        for q in synthetic_queries:
            emb = get_text_embedding(q, retriever, retriever_tokenizer, max_length=max_length)
            synthetic_embeddings.append(emb)
        
        distractor_embeddings = []
        for d in distractors:
            emb = get_text_embedding(d, retriever, retriever_tokenizer, max_length=max_length)
            distractor_embeddings.append(emb)
        
        # ensemble voting: For each synthetic query, choose the distractor with the highest cosine similarity.
        votes = {}
        for syn_emb in synthetic_embeddings:
            sims = cosine_similarity([syn_emb], distractor_embeddings)[0]
            best_idx = sims.argmax()
            best_distractor = distractors[best_idx]
            votes[best_distractor] = votes.get(best_distractor, 0) + 1
        
        # select the distractor with the highest vote.
        final_distractor = max(votes.items(), key=lambda x: x[1])[0]
        sample["background"] = [final_distractor]
        updated_test_data.append(sample)
        
    return updated_test_data

# *** DISTRACTOR CONTEXT GENERATOR *** #
@torch.no_grad()
def generate_distractor_contexts(llm, tokenizer, original_query, k=5):

    prompt = (
        f"Generate {k} completely unrelated, misleading long contexts for the following question: \"{original_query}\". "
        "Each snippet should be at least 4 SENTENCES LONG, Wikipedia-like excerpt about a similar topic but is entirely irrelevant to the question. "
        "Do not mention any keywords related to the original query (or similar terms). "
        "Each snippet should be at least 4 SENTENCES LONG and must not reference the question or its subject in any way. "
        "Do not include numbering, bullet points, extra characters, headers, or any extra labels. "
        "Do not include numbered lists or any type of ordered lists. "
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(llm.device)
    output = llm.generate(**inputs, max_new_tokens=k*100, do_sample=True, temperature=0.9)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # split into lines and remove empty lines
    lines = [line.strip() for line in generated_text.split("\n") if line.strip()]
    
    # post processing
    cleaned_lines = []
    for line in lines:
        if line.startswith("Generate"):
            continue
        if line.startswith("Context"):
            continue
        if line.startswith("Snippet"):
            continue
        if line.startswith("Example"):
            continue
        if line.startswith("Question"):
            continue
        line = re.sub(r"^Generate\s*\d+[:\-]?\s*", "", line)
        line = re.sub(r"^Context\s*\d+[:\-]?\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned_lines.append(line)
    non_empty = [line for line in cleaned_lines if line]

    return non_empty[:k]

# *** GENERATE SYNTHETIC QUERIES *** #
@torch.no_grad()
def generate_synthetic_queries(llm, tokenizer, original_query, k=5):

    synthetic_queries = []
    prompt = (
        f"Generate {k} variations of this question while preserving the intent: \"{original_query}\". "
        "Each new question should be around the same length as the original question. "
        "Do not include numbering, bullet points, extra characters, headers, or any extra labels. "
        "Do not include numbered lists or any type of ordered lists. "
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(llm.device)
    output = llm.generate(**inputs, max_new_tokens=200) # NOTE: temperature can change intent
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    synthetic_queries = [q.strip() for q in generated_text.split("\n") if q.strip()]
    
    # post processing
    cleaned_lines = []
    for line in synthetic_queries:
        if line.startswith("Generate"):
            continue
        if line.startswith("Here"):
            continue
        if line.startswith("Snippet"):
            continue
        if line.startswith("Example"):
            continue
        if line.startswith("Question"):
            continue
        if line.startswith("Here"):
            continue
        line = re.sub(r"^Generate\s*\d+[:\-]?\s*", "", line)
        line = re.sub(r"^Context\s*\d+[:\-]?\s*", "", line)
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        cleaned_lines.append(line)
    non_empty = [line for line in cleaned_lines if line]

    return non_empty[:k]

def get_start_prompt(task_type,use_rag,sample=None):
    if task_type == 'open_qa':
        return {
            True: "Refer to the background document and answer the questions:",
            False:"Answer the questions:"
        }[use_rag]
    elif task_type == 'fact_checking':
        return {
            True: "Refer to the background document and verify the following claims with \"True\" or \"False\":",
            False:"Verify the following claims with \"True\" or \"False\":"
        }[use_rag]
        

@torch.no_grad()
def prepare_retrieval_embeds(backgrounds,retriever,tokenizer,batch_size = 16):
    backgrounds = [backgrounds[idx:idx+batch_size] for idx in range(0,len(backgrounds),batch_size)]
    device = retriever.device
    ret = []
    for background in tqdm(backgrounds):
        tokenized_retrieval_text = tokenizer(
            background, 
            max_length=180,
            padding=True, truncation=True, return_tensors="pt")
        
        ## return a torch tensor of shape [batch_size,d_model]
        embeds = get_retrieval_embeds(
            model = retriever,
            input_ids = tokenized_retrieval_text['input_ids'].to(device),
            attention_mask = tokenized_retrieval_text['attention_mask'].to(device),
        ).cpu()

        embeds = [embeds[idx] for idx in range(embeds.shape[0])]
        ret.extend(embeds)
    return ret

@torch.no_grad()
def llm_for_open_generation(
    llm,llm_tokenizer,
    prompts,
    retrieval_embeds,
    batch_size = 4,
    enable_progress_bar = True,
):
    generated_answers = []
    total_test_number = len(prompts)
    device = llm.device
    batched_prompts = [prompts[idx:idx+batch_size] for idx in range(0,len(prompts),batch_size)]
    if retrieval_embeds is not None:
        batched_retrieval_embeds = [retrieval_embeds[idx:idx+batch_size] for idx in range(0,len(retrieval_embeds),batch_size)]
        assert len(batched_prompts) == len(batched_retrieval_embeds)
    
    progress_bar = tqdm(range(total_test_number),ncols=60,disable= not enable_progress_bar)
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]
        tokenized_propmt = llm_tokenizer(prompt,padding='longest',return_tensors='pt')
        input_ids = tokenized_propmt.input_ids.to(device)
        attention_mask = tokenized_propmt.attention_mask.to(device)
        stopping_criteria = stop_sequences_criteria(llm_tokenizer, input_ids.shape[1], input_ids.shape[0])
        retrieval_kwargs = {}
        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]
            embeds = [x for y in embeds for x in y]
            embeds = torch.stack(embeds).to(device)
            retrieval_kwargs['retrieval_embeds'] = embeds
            stopping_criteria = stop_sequences_criteria(llm_tokenizer, 0, input_ids.shape[0])

        ## actual computation
        generated_output = llm.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            **retrieval_kwargs,
        )
        ## because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]
        results = tokenizer.batch_decode(generated_output[:,input_length:],skip_special_tokens=False)
        generated_answers.extend(results)
        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers

def format_one_example(
    sample,include_answer,use_rag,retrieval_embed_length,task_type,
):
    
    question   = sample['question']
    prompt_dict = dict(question=question)
    prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
    backgrounds = []

    if use_rag:
        backgrounds = sample['background'] ## a list
        background_prompts = ""
        
        for background in backgrounds:
            if retrieval_embed_length > 0:
                background_prompts += " ".join([XRAG_TOKEN]*retrieval_embed_length) + " "
            
            else:
                background_prompts += background + " "
        background_prompts = background_prompts.strip()
        prompt = BACKGROUND_PROMPT_TEMPLATE.format_map(dict(background=background_prompts)) + prompt


    return prompt,backgrounds

def get_n_shot_prompt(dev_data,n_shot,task_type,use_rag=False,retrieval_embed_length=0):
    assert n_shot >= 0,n_shot
    n_shot_prompt = []
    n_shot_background = []
    if dev_data is not None:
        n_shot_examples = dev_data[:n_shot]
        for example in n_shot_examples:
            prompt,background = format_one_example(example,include_answer=True,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            n_shot_prompt.append(prompt)
            n_shot_background.append(background)

    return n_shot_prompt,n_shot_background


def prepare_prompts(
    dev_data,test_data,task_type,tokenizer,
    n_shot = 0, use_rag = False,
    retrieval_embed_length=0,
    chat_format = None,
):
    splitter = "\n\n"
    prompts = []
    backgrounds = []
    original_n_shot = n_shot
    for idx,sample in enumerate(test_data):
        n_shot = original_n_shot
        while True:
            prompt_start  = get_start_prompt(task_type,use_rag=use_rag,sample=sample) 
            prompt_end,background    = format_one_example(
                sample,include_answer=False,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            if 'subject' not in sample.keys():
                n_shot_prompt,n_shot_background = get_n_shot_prompt(dev_data,n_shot=n_shot,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            else:
                ## select n-shot within the same subjects for MMLU
                dev_data_with_same_subjects = []
                for d in dev_data:
                    if d['subject'] == sample['subject']:
                        dev_data_with_same_subjects.append(d)
                assert len(dev_data_with_same_subjects)==5,sample['subject']
                n_shot_prompt,n_shot_background = get_n_shot_prompt(dev_data_with_same_subjects,n_shot=n_shot,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            
            if n_shot_prompt:  
                prompt = prompt_start + splitter + splitter.join(n_shot_prompt) + splitter + prompt_end  
            else: 
                prompt = prompt_start + splitter + prompt_end

            if chat_format is not None:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_format(messages, tokenizer) + " The answer is:"
                

            tokenized_prompt = tokenizer(prompt,truncation=False,add_special_tokens=False).input_ids

            if len(tokenized_prompt) > 2048 and n_shot >= 1:
                n_shot -= 1
            else:
                break
        
        prompts.append(prompt)
        backgrounds.append(background+n_shot_background)

    print("**"*20,"show one example","**"*20)
    print(prompts[0])
    print("**"*20,"show one example","**"*20)

    return prompts,backgrounds


def load_dataset(data,use_rag,args,use_distractors=False,k_samples=0,max_samples=None):
    
    dev_data = None
    test_data = None
    
    # *** USE TRIVIAQA CONTEXT *** #
    if data.lower() == "triviaqa_topk":
        
        # load triviaqa from hf
        dataset = datasets.load_dataset("mandarjoshi/trivia_qa", 'rc', split="validation")
        
        test_data = []
        print("Loading TriviaQA dataset...")
        for sample in tqdm(dataset):
            
            question = sample['question']
            
            answer = ""
            if "answer" in sample and isinstance(sample["answer"], dict):
                answer = sample["answer"].get("value", "")
                
            search_results = sample.get("search_results", {})
            search_contexts = search_results.get("search_context", [])
            
            if not isinstance(search_contexts, list):
                search_contexts = [search_contexts]
                
            background = [ctx for ctx in search_contexts if isinstance(ctx, str) and ctx.strip() != ""]
            
            test_data.append({
                "question": question,
                "answer": answer,
                "background": background
            })

    else:
        
        # fallback: load from JSONL file as before.
        if use_distractors:
            test_retrieval_path = f"../../data/eval/{data}/retrieval/colbertv2/triviaqa_syn.jsonl"
            
            if os.path.isfile(test_retrieval_path):
                test_data = get_jsonl(test_retrieval_path)

            if use_rag:
                test_retrieval = get_jsonl(test_retrieval_path)
                assert len(test_retrieval) == len(test_data), "Mismatch in retrieval and test data length"

                # Include all distractors as background context
                for idx in range(len(test_data)):
                    test_data[idx]['background'] = [entry['text'] for entry in test_retrieval[idx]['topk']]

                if args.tf_idf_topk > 0:
                    assert args.use_rag, "TF-IDF filtering requires RAG mode"
                    
                    # Extract documents for TF-IDF keyword extraction
                    documents = [" ".join(x['background']) for x in test_data]  
                    keywords = keyword_extraction_with_tfidf(documents, topk=args.tf_idf_topk)
                    
                    for idx in range(len(test_data)):
                        test_data[idx]['background'] = [keywords[idx]]

                if args.retriever_name_or_path is not None and args.retriever_name_or_path.lower() == "intfloat/e5-large-v2":
                    for idx in range(len(test_data)):
                        test_data[idx]['background'] = ["passage: " + text for text in test_data[idx]['background']]
        
        else:
            
            test_path = f"../../data/eval/{data}/test.jsonl"
            if k_samples > 0:
                test_retrieval_path = f"../../data/eval/{data}/retrieval/colbertv2/triviaqa_syn_ensemble_{k_samples}.jsonl"
            else:
                test_retrieval_path = f"../../data/eval/{data}/retrieval/colbertv2/test.jsonl"
            
            if os.path.isfile(test_path):
                test_data = get_jsonl(test_path)

            if use_rag:
                
                test_retrieval = get_jsonl(test_retrieval_path)
                
                if max_samples is not None:
                    test_data = test_data[:max_samples]
                    test_retrieval = test_retrieval[:max_samples]
                assert len(test_retrieval) == len(test_data)
                
                for idx in range(len(test_data)):
                    if k_samples > 0:
                        test_data[idx]['background'] = [test_retrieval[idx]['topk'] for rank in args.retrieval_topk]
                    else:
                        test_data[idx]['background'] = [test_retrieval[idx]['topk'][rank]['text'] for rank in args.retrieval_topk]

                if args.tf_idf_topk > 0:
                    assert args.use_rag
                    documents = [x['background'][0] for x in test_data]
                    keywords = keyword_extraction_with_tfidf(documents, topk=args.tf_idf_topk)
                    for idx in range(len(test_data)):
                        test_data[idx]['background'] = [keywords[idx]]

                if args.retriever_name_or_path is not None and args.retriever_name_or_path.lower() == "intfloat/e5-large-v2":
                    for idx in range(len(test_data)):
                        test_data[idx]['background'] = ["passage: " + x for x in test_data[idx]['background']]

    return dev_data, test_data

if __name__ == "__main__":

    args = parse_args()
    
    if args.create_distractors:
        
        ## prepare dataset
        dev_data, test_data = load_dataset(
            args.data,
            args.use_rag,
            args,
        )
        
        # *** ADD DISTRACTOR CHOICES *** #
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        temp_llm = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
        ).to(device)
        temp_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        temp_llm.eval()

        if args.max_test_samples is not None:
            test_data = test_data[:args.max_test_samples]
            
        ext = str(args.max_test_samples) if args.max_test_samples is not None else "all"
        output_file = os.path.join(args.save_dir, "triviaqa_syn.jsonl")
        with open(output_file, "w", encoding="utf-8") as f_out:
            for sample in tqdm(test_data, desc="Generating distractor JSONL"):
                query = sample["question"]
                orig_background = sample.get("background", [])
                distractor_snippets = generate_distractor_contexts(temp_llm, temp_tokenizer, query, k=5)
                
                topk_list = []
                for bg in orig_background:
                    topk_item = {
                        "pid": None,
                        "prob": None,
                        "rank": None,
                        "score": None,
                        "text": bg
                    }
                    topk_list.append(topk_item)
                for snippet in distractor_snippets:
                    topk_item = {
                        "pid": None,
                        "prob": None,
                        "rank": None,
                        "score": None,
                        "text": snippet
                    }
                    topk_list.append(topk_item)
                output_obj = {"query": query, "topk": topk_list}
                f_out.write(json.dumps(output_obj) + "\n")
        print(f"Distractor JSONL saved to {output_file}")
        exit(0)  # leave python script
    
    # ** LOAD MODEL FOR SYNTHETIC QUERY GENERATION ** #
    if args.ensemble_rerank:
        # load dataset with distractor choices
        dev_data, test_data = load_dataset(
            args.data,
            args.use_rag,
            args,
            use_distractors=True
        )
        if args.max_test_samples is not None:
            test_data = test_data[:args.max_test_samples]
        
        # load model to generate synthetic queries
        if args.k_samples > 0:
            syn_llm = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Llama-2-7b-chat-hf",
                torch_dtype=torch.float16,
                device_map='auto',
            )
            syn_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            syn_llm.eval()
            for sample in tqdm(test_data, desc="Generating synthetic queries"):
                original_query = sample["query"]
                synthetic_queries = generate_synthetic_queries(syn_llm, syn_tokenizer, original_query, k=args.k_samples)
                sample["synthetic_queries"] = synthetic_queries
            del syn_llm, syn_tokenizer
            torch.cuda.empty_cache()
        
        # load retriever for ensemble reranking
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        retrieval_embed_length = 0
        retriever, retriever_tokenizer = None, None
        if args.retriever_name_or_path is not None:
            if args.retriever_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
                retriever = SFR.from_pretrained(args.retriever_name_or_path, torch_dtype=torch.bfloat16)
                retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_name_or_path)
            retrieval_embed_length = retriever.get_embed_length()
            retriever_hidden_size = retriever.get_embed_dim()
            retriever.eval()
            retriever = retriever.to(device)
        
        # (This function uses the precomputed sample["synthetic_queries"] and the distractor choices in sample["background"])
        test_data = ensemble_rerank_distractors(test_data, retriever, retriever_tokenizer)
        
        # save the ensemble JSONL file
        ensemble_output_file = os.path.join(args.save_dir, f"triviaqa_syn_ensemble_{args.k_samples}.jsonl")
        with open(ensemble_output_file, "w", encoding="utf-8") as f_out:
            for sample in tqdm(test_data, desc="Saving ensemble JSONL"):
                output_obj = {
                    "query": sample["query"],
                    "synthetic_queries": sample.get("synthetic_queries", []),
                    "topk": sample["background"][0] if sample.get("background") else ""
                }
                f_out.write(json.dumps(output_obj) + "\n")
        print(f"Ensemble JSONL saved to {ensemble_output_file}")
        exit(0)
        
    ## load dataset
    dev_data, test_data = load_dataset(
        args.data,
        args.use_rag,
        args,
        k_samples=args.k_samples,
        max_samples=args.max_test_samples,
    )
    
    ## load retriever for ensemble reranking
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    retrieval_embed_length = 0
    retriever, retriever_tokenizer = None, None
    if args.retriever_name_or_path is not None:
        if args.retriever_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
            retriever = SFR.from_pretrained(args.retriever_name_or_path, torch_dtype=torch.bfloat16)
            retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_name_or_path)
        retrieval_embed_length = retriever.get_embed_length()
        retriever_hidden_size = retriever.get_embed_dim()
        retriever.eval()
        retriever = retriever.to(device)
    
    ## prepare prompts
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side = 'left',
        add_eos_token=False, ## import to include this!
        use_fast=False,
    )
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    prompts,backgrounds = prepare_prompts(
        dev_data = dev_data,
        test_data = test_data,
        task_type = args.task_type,
        tokenizer = tokenizer,
        n_shot = args.n_shot,
        use_rag = args.use_rag,
        retrieval_embed_length = retrieval_embed_length,
        chat_format = args.chat_format, 
    )

    retrieval_embeds = None
    if retriever is not None:
        # backgrounds List[List[String]]
        num_samples = len(backgrounds)
        original_orders = []
        for idx,background in enumerate(backgrounds):
            original_orders.extend(
                [idx] * len(background)
            )
        
        backgrounds = [x for y in backgrounds for x in y]
        print(f"Preparing document embedding with {args.retriever_name_or_path}...")
        _retrieval_embeds = prepare_retrieval_embeds(
            backgrounds,
            retriever,
            retriever_tokenizer,
        )

        retrieval_embeds = [[] for _ in range(num_samples)]
        assert len(_retrieval_embeds) == len(original_orders)
        for id,embeds in zip(original_orders,_retrieval_embeds):
            retrieval_embeds[id].append(embeds)

        retriever = retriever.to("cpu")


    avg_prompt_length = tokenizer(prompts,return_length=True).length
    avg_prompt_length = sum(avg_prompt_length)/len(avg_prompt_length)
    
    ## load llm
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    MODEL_CLASS = eval(config.architectures[0])
    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True,
        device_map='auto',
    )
    model.eval()
    
    if retriever is not None:
        assert XRAG_TOKEN in tokenizer.get_vocab() 
        model.set_xrag_token_id(tokenizer.convert_tokens_to_ids(XRAG_TOKEN))

    if args.task_type in ['open_qa','fact_checking']:
        generated_results = llm_for_open_generation(
            llm = model,
            llm_tokenizer = tokenizer,
            prompts = prompts,
            retrieval_embeds = retrieval_embeds,
            batch_size = args.eval_batch_size,
            enable_progress_bar= args.enable_progress_bar,
        )

    answers = [x['answer'] for x in test_data]
    if args.eval_metrics == 'substring_match':
        score,score_per_sample = get_substring_match_score(generated_results,answers)
    elif args.eval_metrics == 'fact_checking_acc':
        score,score_per_sample = eval_fact_checking(generated_results,answers)
    elif args.eval_metrics == 'truthfulqa_f1_rl':
        f1,rl,f1_scores,rl_scores = eval_truthfulqa(generated_results,answers)
        score = f"{f1}-{rl}"
        score_per_sample = [(f1_score,rl_score) for f1_score,rl_score in zip(f1_scores,rl_scores)]


    result_dict =   {
        "dataset":args.data,
        "batch_size":args.eval_batch_size,
        "include_retrieval":args.use_rag,
        "avg_prompt_length":avg_prompt_length,
        "model":args.model_name_or_path,
        f"{args.eval_metrics}":score,
    }

    if args.retriever_name_or_path is not None:
        result_dict['retriever'] = args.retriever_name_or_path
    print(json.dumps(result_dict,indent=4))
