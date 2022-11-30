# For reformatting tables
import pyarrow.parquet as pq
import pyarrow.csv as pv
from csv import QUOTE_NONE
from joblib import Parallel, delayed
import glob

import os
import os.path as osp
from tqdm import tqdm

# Creating a new dir for the dumped data
print("Creating dirs")
dest_dir = "data/zenodo/txt_tables"
os.makedirs(dest_dir, exist_ok=True)
for tab in glob.glob("data/zenodo/tables/*"):
    os.makedirs(osp.join(dest_dir, osp.basename(tab)), exist_ok=True)

# Creating a list with the paths for all tables in the collection
print("Creating tables glob")
ls = glob.glob("data/zenodo/tables/*/*")
print(f"Found {len(ls)} files.")


def dump_table_to_csv(idx, table_path, dest_dir):
    try:
        tgt_dir, path = osp.split(osp.relpath(table_path, "data/zenodo/tables/"))
        path = osp.join(dest_dir, tgt_dir, osp.basename(path) + ".txt")
        if osp.exists(path):
            return (idx, 0)
        tab = pq.read_table(table_path)
        tab.to_pandas().to_csv(path,  index=False, sep=" ", escapechar=" ", quoting=QUOTE_NONE)
        return (idx, 0)
    except Exception:
        return (idx, 1)


print("Converting.")
r = Parallel(n_jobs=8, verbose=0)(
    delayed(dump_table_to_csv)(idx, table_path, dest_dir)
    for idx, table_path in tqdm(
        enumerate(ls), position=0, leave=False, total=len(ls)
    )
)
