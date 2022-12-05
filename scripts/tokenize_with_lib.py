from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers import normalizers, pre_tokenizers, Tokenizer
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import glob


corpus_files = glob.glob("data/zenodo/txt_tables/*/*.txt")


normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
# Testing normalizer
normalizer.normalize_str("Héllò hôw are ü?")
pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Digits()])
# Testing pre-tokenizer
pre_tokenizer.pre_tokenize_str("Call 911!")
# Creating the tokenizer with the assigned normalizer and pre-tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer

trainer = BpeTrainer() # No special tokens for now

tokenizer.train(corpus_files, trainer)

tokenizer.save("data/tokenizer.json")