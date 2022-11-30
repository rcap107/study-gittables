# Tokenize the table content and check the distribution of tokens

Author: Riccardo Cappuzzo

In this notebook, I'll work on the content of the tables and use the Hugging Face
`tokenizers` library to create a tokenized vocabulary of the entire corpus. This
should help with understanding what kind of tokens appear most frequently, and 
give an idea of what we should be expecting from a dirty, mixed type corpus of 
tables sourced from the web. 


I'll start by importing some of the `normalizers` from the `tokenizer` library. 

More on `normalizers` in the [normalizers API page](https://huggingface.co/docs/tokenizers/v0.13.2/en/api/normalizers).
These normalizers rely on [unicode normalization](https://unicode.org/reports/tr15/).



```python
from sklearn.feature_extraction.text import CountVectorizer

from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers import normalizers, pre_tokenizers, Tokenizer
from tokenizers.pre_tokenizers import Whitespace, Digits
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# For reformatting tables
import pyarrow.parquet as pq
import pyarrow.csv as pv
from csv import QUOTE_NONE

# For path and file operations
import os.path as osp
import os
import glob
from random import sample
from joblib import Parallel, delayed

# For progress bar
from tqdm import tqdm

# For plotting and operations
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
```

## Converting csv files into text files

I'll start by "globbing" all the table paths. I'll have to first convert all the tables to 
text, then feed all the files to the tokenizer. After building the tokenizer, I'll go back and encode all tables again to 
find the most frequent tokens. 


```python
# Creating a new dir for the dumped data
dest_dir = "data/zenodo/txt_tables"
os.makedirs(dest_dir, exist_ok=True)
```


```python
# Creating a list with the paths for all tables in the collection
ls = glob.glob("data/zenodo/tables/*/*")
print(f"Found {len(ls)} files.")
```

For debugging, I'll select a small sample of tables first.


```python
ls_sample = sample(ls, k=100)
```

I'll use joblib to parallelize the next steps.


```python
def dump_table_to_csv(idx, table_path, dest_dir):
    try:
        tgt_dir, path = osp.split(osp.relpath(table_path, "data/zenodo/tables/"))
        path = osp.join(dest_dir, tgt_dir, osp.basename(path) + ".txt")
        print(path)
        if osp.exists(path):
            return (idx, 0)
        # print(path)
        # tab = pq.read_table(table_path)
        # tab.to_pandas().to_csv(path,  index=False, sep=" ", escapechar=" ", quoting=QUOTE_NONE)
        return (idx, 0)
    except Exception:
        return (idx, 1)

```

I don't really care about the result, since the conversion is the actual operation. Still, it's useful for tracking down those files that were not converted successfully for some 
reason. 


```python
r = Parallel(n_jobs=1, verbose=0)(
    delayed(dump_table_to_csv)(
        idx, table_path, dest_dir) for idx, table_path in 
    tqdm(enumerate(ls[:10]), position=0, leave=False, total=len(ls)))
```

                                               

    data/zenodo/txt_tables/abstraction_tables_licensed/Designite_Microsoft.Phone.Tools.Deploy.Patched_DesignSmells.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/Designite_VisualEffectPlayground_DesignSmells.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/Designite_Reign.Input.API_DesignSmells.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/08-09_18.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/13-14_43.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/Designite_Chinook_DesignSmells.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/03-04_81.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/Designite_squidtray_DesignSmells.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/Designite_ThreadedCoreData_DesignSmells.parquet.txt
    data/zenodo/txt_tables/abstraction_tables_licensed/Designite_Helpmebot.Tests_DesignSmells.parquet.txt


    


```python
print(f"There are {len(r)} results, and {len(os.listdir(dest_dir))} files in {dest_dir}.")
```

    There are 1018624 results, and 562190 files in data/zenodo/txt_tables.


It looks like about half of the tables have been overwritten because of repeated filenames. 

Whoops. 

Time to run this again, this time placing each file in its folder. 

To reduce the cost of each operation, all folders are created before running the code, and 
then populated while iterating. 


```python
for tab in glob.glob("data/zenodo/tables/*"):
    os.makedirs(osp.join(dest_dir, osp.basename(tab)), exist_ok=True)
```


```python
table_path = ls_sample[0]
tgt_dir, path = osp.split(osp.relpath(table_path, "data/zenodo/tables/"))
path = osp.join(dest_dir, tgt_dir, osp.basename(path) + ".txt")
print(path)
```

    data/zenodo/txt_tables/abstraction_tables_licensed/Designite_SqlLinq.UnitTests_DesignSmells.parquet.txt



```python
# Running the conversion task again (~45mins on drago3)
```

### Tokenizing with scikit-learn CountVectorizer


```python
pwd
```




    '/home/soda/rcappuzz/work/study-gittables/notebooks'




```python
corpus_files = glob.glob("../data/zenodo/txt_tables/*/*.txt")
```


```python
vectorizer = CountVectorizer(analyzer="word", ngram_range=(2,2))
X = vectorizer.fit_transform(corpus_files)
words = vectorizer.get_feature_names_out()
```

Here I am extracting the total number of occurrences of each token in each document in the corpus, then I am creating a 
new matrix with the token and its number of occurrences next to each other. At this point, I still hadn't realized I 
was making a mistake. Can you tell? 
```python
counts = X.sum(axis=0)
counts_array = np.squeeze(np.asarray(counts))

word_counts = np.concatenate([words, counts_array], axis=0).reshape(2, -1)
word_counts
```

    array([['00 00', '00 000', '00 00_1', ..., 'zyzyxia_1 parquet',
            'zzgq4 976', 'zzh8829__yolov3 tf2'],
           [8, 1, 1, ..., 1, 2, 5]], dtype=object)

Time to try and plot the values! 
```python
sns.histplot(data=counts_array,  log_scale=(True, True))
```

 
![png](tokenize-study_files/tokenize-study_28_1.png)

Well, this doesn't look right. In fact, the mistake I made was running the `CountVectorizer` on the *filenames*, rather 
than the content of the files themselves (I did not quite understand the `input` parameter before running `fit_transform`).
The result is a histogram of the most frequent tokens, which means that the three parts of the path `data/zenodo/txt_tables` 
constantly appear in each string, and thus dominate the tokens. 

At this point, I lost some hope this would be an easy task. 


### Tokenizing with the `tokenizers` library


```python
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
```


```python
cd /home/soda/rcappuzz/work/study-gittables
```

    /home/soda/rcappuzz/work/study-gittables


### Preparing a random table for tokenization


```python
tab = pq.read_table("data/zenodo/tables/allegro_con_spirito_tables_licensed/Aziende.parquet")
```

Saving the table to csv, removing all quotation and separators. `QUOTE_NONE` is 
used to remove all quote markers, `escapechar` is needed to avoid `csv` to throw
an error. By using `sep=" "` and `escapechar=" "`, there is no quoting and 
all fields are separated by `"  "` (two spaces).  


```python
tab.to_pandas().to_csv("tb.txt",  index=False, sep=" ", escapechar=" ", quoting=QUOTE_NONE)
```

### Tokenizing the content of the table


```python
trainer = BpeTrainer() # No special tokens for now
tokenizer.train(["tb.txt"], trainer)
# Testing the encoder on random letters
tokenizer.encode("It's Captain Jack Sparrow").tokens
```

    

    ['i', 't', 's', 'cap', 'ta', 'in', 'jack', 'spar', 'ro', 'w']

The rest of the notes will be in a different notebook.