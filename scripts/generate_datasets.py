import os
import subprocess
import signal
import sys

DATA_DIR = os.environ.get('DATA_DIR', 'data')
DISP = os.environ.get('DISP', False)

print("Generating dataset... Folder:", DATA_DIR)

# Container to keep track of subprocesses
processes = []

#############################
# Language-Conditioned Tasks

train_n = 50
test_n = 25

LANG_TASKS = [
    'align-rope',
    'assembling-kits-seq-seen-colors',
    'assembling-kits-seq-unseen-colors',
    'packing-shapes',
    'packing-boxes-pairs-seen-colors',
    'packing-boxes-pairs-unseen-colors',
    'put-block-in-bowl-seen-colors',
    'put-block-in-bowl-unseen-colors',
    'stack-block-pyramid-seq-seen-colors',
    'stack-block-pyramid-seq-unseen-colors',
    'separating-piles-seen-colors',
    'separating-piles-unseen-colors',
    'towers-of-hanoi-seq-seen-colors',
    'towers-of-hanoi-seq-unseen-colors',
]

for task in LANG_TASKS:
    for mode in ['train', 'val', 'test']:
        n = 0
        if mode == 'train':
            n = train_n
        else:
            n = test_n
        cmd = f"python3 ./scripts/demos.py n={n} task={
            task} mode={mode} data_dir={DATA_DIR} disp={DISP}"
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

print("Finished Language Tasks.")

#########################
# Demo-Conditioned Tasks

DEMO_TASKS = [
    'align-box-corner',
    'assembling-kits',
    'block-insertion',
    'manipulating-rope',
    'packing-boxes',
    'palletizing-boxes',
    'place-red-in-green',
    'stack-block-pyramid',
    'sweeping-piles',
    'towers-of-hanoi',
]

for task in DEMO_TASKS:
    for mode in ['train', 'val', 'test']:
        n = 0
        if mode == 'train':
            n = train_n
        else:
            n = test_n
        cmd = f"python3 ./scripts/demos.py n={n} task={
            task} mode={mode} data_dir={DATA_DIR} disp={DISP}"
        process = subprocess.Popen(cmd, shell=True)
        processes.append(process)

print("Finished Demo Tasks.")


def signal_handler(sig, frame):
    print('Interrupt received, terminating subprocesses...')
    for proc in processes:
        proc.terminate()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Wait for all processes to complete
for proc in processes:
    proc.wait()
