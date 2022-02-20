import sys
import pandas as pd
import os

dureldir = sys.argv[1]
sent_per_word = sys.argv[2]
resultdir = sys.argv[3]

durel_files = os.listdir(dureldir)

if not os.path.exists(resultdir):
    os.makedirs(resultdir)

os.chdir(resultdir)

for filename in durel_files:
    examples = pd.read_csv(os.path.join(os.pardir, dureldir, filename), sep='\t', header=0, compression='gzip', encoding='utf-8')
    print('In file', filename, len(examples), 'examples found')
    result = examples.sample(int(sent_per_word))

    result.to_csv(filename.replace('example', 'sample').split('.gz')[0], sep='\t', line_terminator='\n', index=False)
