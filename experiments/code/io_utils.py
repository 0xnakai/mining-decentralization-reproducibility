"""Atomic CSV output and file hashing shared by the simulation runner."""
from pathlib import Path
import gzip, hashlib, io, os
import pandas as pd

def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def csv(frame:pd.DataFrame,path:Path):
    """Commit complete files atomically, including the gzip trailer.

    A process-specific temporary path prevents concurrent or interrupted runs
    from truncating a delivered CSV. Empty gzip filenames and fixed mtime also
    make the compressed bytes independent of the output directory and clock.
    """
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        if path.name.endswith('.gz'):
            with temporary.open('wb') as raw:
                with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as compressed:
                    with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as stream:
                        frame.to_csv(stream,index=False)
        else:frame.to_csv(temporary,index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():temporary.unlink()
