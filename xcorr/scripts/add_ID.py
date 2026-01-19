# add three blank columns and an ID column to the QM output (version difference?)
# for compatibility with new sort script
import sys

inp = sys.argv[1]

with open(inp) as f:
    lines = f.readlines()

with open(inp + "_wID", 'w') as f:
    for i, evt in enumerate(lines):
        new = evt.strip() + f",0,0,0,{i+1}\n"
        # evt += f",{i+1}\n"
        f.write(new)

