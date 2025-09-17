import sys

from claimtriageai.configs import paths

if len(sys.argv) != 2:
    print("Usage: python scripts/get_path.py <VARIABLE_NAME>")
    sys.exit(1)

var = sys.argv[1]
try:
    value = getattr(paths, var)
    print(value)
except AttributeError:
    print(f"Variable '{var}' not found in claimtriageai.configs.paths.")
    sys.exit(1)
