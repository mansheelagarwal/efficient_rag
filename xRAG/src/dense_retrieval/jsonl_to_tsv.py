import json
from tqdm import tqdm

def jsonl_to_tsv(input_jsonl, output_tsv, include_infobox=False, total_estimate=33200000):
    """
    Converts a JSONL file to TSV format with an estimated progress bar.
    
    Args:
        input_jsonl (str): Path to input JSONL file.
        output_tsv (str): Path to output TSV file.
        include_infobox (bool): If True, append infobox content to text.
        total_estimate (int): Approximate number of lines for tqdm (default 37.5M).
    """
    with open(input_jsonl, "r", encoding="utf-8") as jsonl_file, open(output_tsv, "w", encoding="utf-8") as tsv_file:
        pbar = tqdm(total=total_estimate, desc="Converting JSONL to TSV", unit=" lines", smoothing=0.01)
        
        for i, line in enumerate(jsonl_file):
            data = json.loads(line)
            doc_id = data.get("id", "")
            title = data.get("title", "").replace("\t", " ")  # Replace tabs with spaces
            text = data.get("text", "").replace("\t", " ")  # Replace tabs with spaces

            if include_infobox and "infobox" in data:
                infobox = data["infobox"].replace("\t", " ")
                text += " " + infobox  # Append infobox content to text
            
            if text:  # Only write rows with valid text
                tsv_file.write(f"{doc_id}\t{title}\t{text}\n")

            # Update progress bar every 10,000 lines (reduces overhead)
            if i % 10000 == 0:
                pbar.update(10000)

        pbar.close()

if __name__ == "__main__":
    jsonl_to_tsv(
        "/group/jmearlesgrp/data/Atlas/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl",
        "/group/jmearlesgrp/data/Atlas/corpora/wiki/enwiki-dec2021/collection.tsv"
    )