import sys
from claimtriageai.configs.paths import *

if len(sys.argv) != 2:
    print("Usage: python scripts/get_path.py <VARIABLE_NAME>")
    sys.exit(1)

var = sys.argv[1]
try:
    value = globals()[var]
    print(value)
except KeyError:
    print(f"Variable '{var}' not found in claimtriageai.configs.paths.")
    sys.exit(1)
