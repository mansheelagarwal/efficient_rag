import pickle
from tqdm import tqdm
import os
import csv
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import BertTokenizer, AutoModel
import torch
from accelerate import PartialState
from tqdm import tqdm

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection_path", default="data/collection.tsv")
    parser.add_argument("--encoding_batch_size", type=int, default=1024)
    parser.add_argument("--max_doclen", type=int, default=180)
    parser.add_argument("--pretrained_model_path", default="colbert-ir/colbertv2.0")  # Default to Hugging Face model
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_embedding_num_per_shard", type=int, default=200_000)
    args = parser.parse_args()

    distributed_state = PartialState()
    device = distributed_state.device

    # Load Hugging Face ColBERT Model
    model = AutoModel.from_pretrained(args.pretrained_model_path)
    model.eval()
    model.to(device)

    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)

    # Function to extract document embeddings
    def get_doc_embedding(input_ids, attention_mask):
        """Extract document embeddings using ColBERT/BERT"""
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state  # Use hidden states as embeddings
        return embeddings

    collections = []
    print(f"Reading collection from {args.collection_path}")
    if "collection.tsv" in args.collection_path:
        pbar = tqdm(total=33200000, desc="Parsing collection...", unit=" lines", smoothing=0.01)
        with open(args.collection_path) as f:
            for i, line in enumerate(f):
                line_parts = line.strip().split("\t")
                pid, title, passage, *other = line_parts
                passage = passage.strip()
                assert len(passage) >= 1
                
                # Format: "Title | Passage"
                formatted_passage = f"{title} | {passage}"
                
                collections.append(passage)
                
                # Update progress bar every 10,000 lines (reduces overhead)
                if i % 10000 == 0:
                    pbar.update(10000)

    with distributed_state.split_between_processes(collections) as sharded_collections:
        
        sharded_collections = [sharded_collections[idx:idx+args.encoding_batch_size] for idx in range(0, len(sharded_collections), args.encoding_batch_size)]
        
        os.makedirs(args.output_dir, exist_ok=True)
        shard_id = 0
        doc_embeddings = []
        doc_embeddings_lengths = []
        
        print(f"Encoding {len(sharded_collections)} shards of collections")
        with tqdm(total=len(sharded_collections), desc="Encoding documents", unit=" batches") as encoding_pbar:
            for docs in sharded_collections:
                docs = ["[D] " + doc for doc in docs]
                model_input = tokenizer(docs, max_length=args.max_doclen, padding='max_length', return_tensors='pt', truncation=True).to(device)
                input_ids = model_input.input_ids
                attention_mask = model_input.attention_mask

                doc_embedding = get_doc_embedding(input_ids, attention_mask)  # Extract embeddings
                lengths = [doc.shape[0] for doc in doc_embedding]

                doc_embeddings.extend(doc_embedding)
                doc_embeddings_lengths.extend(lengths)
                encoding_pbar.update(1)  # Update encoding progress bar

                if len(doc_embeddings) >= args.max_embedding_num_per_shard:
                    print("Saving shard...")
                    # doc_embeddings = torch.cat(doc_embeddings, dim=0)
                    # torch.save(doc_embeddings, f'{args.output_dir}/collection_shard_{distributed_state.process_index}_{shard_id}.pt')
                    # pickle.dump(doc_embeddings_lengths, open(f"{args.output_dir}/length_shard_{distributed_state.process_index}_{shard_id}.pkl", 'wb'))
                    doc_embeddings = torch.cat(doc_embeddings, dim=0).cpu().to(torch.float16).numpy()  # Use float16 instead of bfloat16
                    np.savez_compressed(f"{args.output_dir}/collection_shard_{distributed_state.process_index}_{shard_id}.npz", doc_embeddings)
                    pickle.dump(doc_embeddings_lengths, open(f"{args.output_dir}/length_shard_{distributed_state.process_index}_{shard_id}.pkl", 'wb'))
                
                    # Reset for new shard
                    shard_id += 1
                    doc_embeddings = []
                    doc_embeddings_lengths = []

        # Save remaining embeddings
        if len(doc_embeddings) > 0:
            # doc_embeddings = torch.cat(doc_embeddings, dim=0)
            # torch.save(doc_embeddings, f'{args.output_dir}/collection_shard_{distributed_state.process_index}_{shard_id}.pt')
            # pickle.dump(doc_embeddings_lengths, open(f"{args.output_dir}/length_shard_{distributed_state.process_index}_{shard_id}.pkl", 'wb'))
            doc_embeddings = torch.cat(doc_embeddings, dim=0).cpu().to(torch.float16).numpy()  # Use float16
            np.savez_compressed(f"{args.output_dir}/collection_shard_{distributed_state.process_index}_{shard_id}.npz", doc_embeddings)
            pickle.dump(doc_embeddings_lengths, open(f"{args.output_dir}/length_shard_{distributed_state.process_index}_{shard_id}.pkl", 'wb'))
