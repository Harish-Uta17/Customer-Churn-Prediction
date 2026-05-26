import sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

modules = [
    'src.data.loader',
    'src.data.cleaner',
    'src.preprocessing.preprocessor',
    'src.models.train',
    'src.models.evaluate'
]

for m in modules:
    try:
        __import__(m, fromlist=['*'])
        print(m, 'OK')
    except Exception:
        print('ERROR importing', m)
        traceback.print_exc()
