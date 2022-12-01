from tokenizers import Tokenizer, Encoding
import glob
from collections import Counter
import os
import os.path as osp
from tqdm import tqdm
import json
import os



def tally_tokens(target_tab, tokenizer):
    with open(target_tab, "r") as fp:
        lines = fp.readlines()
        batch_ = tokenizer.encode_batch(lines)
        tokens = [token for enc in batch_ for token in enc.tokens]
    return tokens


tokenizer_path = "data/tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

table_glob = glob.glob("data/zenodo/txt_tables/*/*")

global_counter = Counter(dict(zip(tokenizer.get_vocab(), [0 for _ in range(tokenizer.get_vocab_size())])))

print("Starting tally operation.")

for idx, target_tab in tqdm(
        enumerate(table_glob), total = len(table_glob)
    ):
    try:
        global_counter.update(tally_tokens(target_tab, tokenizer))
    except KeyboardInterrupt:
        print("Interrupting...")
        break

json.dump(global_counter, open("global_counter.json", "w"))
