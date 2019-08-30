# Usage: python3.5 src/cache/cache.py [gpu_id_range] [n_train] [n_dev] [n_test]

import os, sys
import progressbar
import time
from subprocess import Popen, PIPE

from utils.data import load_tokens

[gpu_min, gpu_max] = sys.argv[1].split('-')
[gpu_min, gpu_max] = [int(gpu_min), int(gpu_max) + 1]
n_gpu = gpu_max - gpu_min
n_train = int(sys.argv[2])
n_dev = int(sys.argv[3])
n_test = int(sys.argv[4])

print('Preparing %d train instances, %d dev instances, %d test instances ...' % (n_train, n_dev, n_test))
tokens_train = load_tokens('train', n_train)
tokens_dev = load_tokens('dev', n_dev)
tokens_test = load_tokens('test', n_test)
tokens = tokens_train + tokens_dev + tokens_test
tokens = [token for token in tokens if not os.path.isfile('cache/' + token)]
n_tokens = len(tokens)
n_cached = n_train + n_dev + n_test - n_tokens
print('Already cached %d instances. Preparing %d uncached instances ...' % (n_cached, n_tokens))

tokenss = [tokens[i:n_tokens:n_gpu] for i in range(n_gpu)]
subprocesses = []
for i in range(n_gpu):
    gpu_id = gpu_min + i
    if tokenss[i] == []:
        continue
    subprocess = Popen(['python', 'src/cache/worker.py', str(gpu_id)] + tokenss[i], stdout=PIPE, stderr=PIPE)
    subprocesses.append(subprocess)
pbar = progressbar.ProgressBar(widgets=[
	progressbar.Percentage(), ' ',
	progressbar.Bar(marker='>', fill='-'), ' ',
	progressbar.ETA(), ' ',
	], maxval=n_tokens).start()
while True:
    newly_cached = [token for token in tokens if os.path.isfile('cache/' + token)]
    pbar.update(len(newly_cached))
    if len(newly_cached) == n_tokens:
        pbar.finish()
        break
    time.sleep(10)
for subprocess in subprocesses:
    subprocess.wait()

