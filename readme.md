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
The dataset was downloaded using https://github.com/rcap107/zenodo-download, while the actual study and preparation of the full dataset is done using https://github.com/rcap107/study-gittables.

## Extracting the tables
All archives saved in `data/zenodo/archives` were extracted in `data/zenodo/tables` using `extracting_tables.py`. Each archive is extracted in its own subfolder. 

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
tail -n +1 -q *.csv > info_compiled.csv
```

The actual study of the compiled information file is `notebooks/aggregate-study.ipynb`. 

## Studying the metadata
The study of the metadata is in `notebooks/metadata-study.ipynb`. 


## Next steps

- Cleanup
    - Some cleanup is mandatory. There is just too much stuff to use.
    - Cleanup should probably be both at the table level (some tables are not useful), and at the column/value level (some columns are a mess to use, e.g. very long textual cells, lists of values)
    - Headers are not always present, table names are not representative of the content
    - Metadata is hard to unpack in its current format
- Encoding of the tables
    - Dones on value level