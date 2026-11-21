from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'engine'))
# Optional dependency locations are supplied by the caller, never a home path.
for path in reversed(os.environ.get('STATESCOPE_DEPENDENCY_PATHS','').split(os.pathsep)):
    if path:sys.path.insert(1,path)
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'

