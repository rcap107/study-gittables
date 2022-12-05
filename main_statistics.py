"""This script combines the scripts that were used to prepare the statistics used to run observations on the repository.

""" 
import json
from tqdm import tqdm
from tokenizers import Tokenizer
import glob 


def tally_tokens(args):

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

