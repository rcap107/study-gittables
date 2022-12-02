# Studying gittables

Repository: 

> GitTables 1M ([https://gittables.github.io](https://gittables.github.io/)) is a corpus of currently 1M relational tables extracted from CSV files in GitHub repositories, that are associated with a license that allows distribution. We aim to grow this to at least 10M tables.
> 
> 
> Each parquet file in this corpus represents a table with the original content (e.g. values and header) as extracted from the corresponding CSV file. Table columns are enriched with annotations corresponding to >2K semantic types from Schema.org and DBpedia (provided as metadata of the parquet file). These column annotations consist of, for example, semantic types, hierarchical relations to other types, and descriptions.
> 
> We believe GitTables can facilitate many use-cases, among which:
> 
> - Data integration, search and validation.
> - Data visualization and analysis recommendation.
> - Schema analysis and completion for e.g. database or knowledge base design.
> 
> If you have questions, the paper, documentation, and contact details are provided on the website: [https://gittables.github.io](https://gittables.github.io/). We recommend using Zenodo's API to easily download the full dataset (i.e. all zipped topic subsets). 

This is a repository that contains a number of (partially labeled) tables from a number of different domains. 

## Downloading the repository
The dataset was downloaded using https://github.com/rcap107/zenodo-download, while the actual study and preparation of 
the full dataset is done using a variety of scripts in https://github.com/rcap107/study-gittables. This readme is assuming
that all archives have already been downloaded in `data/zenodo/archives/`. 

## Preparing the tables for later use
The `main_preparation.py` script contains functions for extracting all tables in 
their respective folders, as well as converting them to a "pseudo-textual" form. 

This file is just a wrapper for the scripts `scripts/extracting_tables.py` and `scripts/convert_tables_to_text.py` 

The `extract` argument to `main_preparation.py` will extract all archives found in the root to the given folder. 

The `convert` argunebt to `main_preparation.py` will instead convert all the (already extracted) tables to a textual format.
More detail on this is reported in the section [Tokenizing the data](#tokenizing-the-data).


## Preliminary study
Each table is stored in a Parquet file which can be processed by different data processing libraries like Pandas, Spark, and Pyarrow. 

Each Parquet file consists of the table itself and its metadata, consisting of:
    
    URL to the original CSV file,
    License of the associated repository,
    Table ID,
    Table dimensions,
    Data types inferred with Pandas,
    Column annotations from different annotation methods and ontologies,
    Table topic annotation derived from column annotations.


We'll have to take the annotations into consideration when we try to use these tables. 

The metadata is stored in the parquet file and can be accessed as follows:
```py
table = pq.read_table(osp.join(tgt_folder, 'table-name.parquet'))
md = table.schema.metadata[b'gittables']
```
According to the original source, about ~70% of the tables have been annotated using types taken from [schema.org](www.schema.org). 

From my cursory search, tables are very tenuously related with each other, even when they're placed in the same "group" (denoted by the table). I need to study the annotations before I can better figure that part out. 

## Aggregate study
For the aggregate study, I have started by gathering some statistics on each table in the full dataset, grouping all statistics by folder. All info files are in `data/zenodo/info_tables`. 

```py
statistics = ['group_name', # The name of the parent folder
        'name', # The name of the scraped dataset 
        'tot_rows', # The number of rows in the dataset
        'tot_columns', # The number of columns in the dataset 
        'tot_values', # tot_rows*tot_columns
        'num_columns', # number of numerical columns (as inferred by pandas)
        'cat_columns', # number of categorical columns (as inferred by pandas)
        'uniq_cat_values', # number of unique categorical values 
        'avg_red_cat_values' # average redundancy of the unique cat values
]
```
The script `stat_tables.py` was used to gather the statistics. To improve performance, I used `joblib` to run the aggregation operations in parallel. 
All stat datasets are saved in different files (one per `tables/` folder). 

To combine the stat datasets in a single file, I just used the shell. 
```sh
tail -n +2 -q *.csv > info_compiled.csv
# tail -n +2 selects all the lines starting from the second (i.e. ignores the header)
# -q suppresses the file name
```


The actual study of the compiled information file is `notebooks/aggregate-study.ipynb`. 

## Studying the metadata
The study of the metadata is in `notebooks/metadata-study.ipynb`. Overall, the metadata contained by the parquet files is 
not particularly useful. The original url of the file is not reliable as a lot of the files are missing. In theory, the 
metadata should include the schema.org and dbpedia tags for the tables, however these tags often disagree, are not 
consistent or are just missing. 

Another issue that came up earlier, and that won't be solved by this step is how most tables lack an actual header, with
the first row being treated as header even though it normally is general data. This is particularly problematic because
the result is that the type tagging operation is executed on less-than-useful data, which explains the remarkably poor 
accuracy of the tags. 

The aggregate information is reported in the notebook. Given the less-than-optimistic outlook, I stopped working on that 
and moved on to the next steps. 

## Tokenizing the data
The objective of this step was tokenizing the content of the tables, then looking for the tokens that appear more 
frequently. The exploratory study is done in `tokenize-study.ipynb`, while the actual tokenization operations are done in 
`convert_tables_to_text.py`, `encode_with_lib.py` and `tokenize_with_lib.py`. 

For this, I first ran a pre-processing step (`convert_tables_to_text.py`) in which I simplified all the `csv` files in the repository by 
replacing all `,` with ` ` (commas with whitespaces), so that the field delimiters would not be considered as tokens by
the tokenizer. 
Something to note is that **many tables share the same name**, but are contained by different tables. For this reason, I 
could not "dump" everything in a single folder, because in that way about half of the tables are actually lost. 
Therefore, to avoid issues, the complete dir tree is used in all directories. 


While I have tried with the `CountVectorizer` function provided by `scikit-learn`, I was not able to obtain
good results. 

I then tried to use the Huggingface `tokenizers` library to implement a slightly customized BPE tokenizer. The creation 
and training of the tokenizer is done in `tokenize_with_lib.py`. The resulting tokenizer is saved in `data/tokenizer.json`. 

This is done in `encode_with_lib.py`, which is a script that reads once again all the tables in the repository, encodes 
them using the tokenizer saved in the previous step, then uses a `Counter` object to count the number of occurrences of
each token in all datasets. In the end, the counter is saved as json in `global_counter.json`. 

## Embeddings with fasttext? 
The potential final step is generating fasttext embeddings on the tables. How this will be done, remains to be seen.
However, the fact that the tables are now available in "textual form", and that it is also possible to encode them using 
a tokenizer opens some possibilities.

There also remains the issue of deciding the granularity of the embeddings: will it be at the level of a table value, 
table row, table column, table? This is something that remains to be seen. 