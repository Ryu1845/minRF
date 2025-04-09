from random import sample
from argparse import ArgumentParser
from pathlib import Path

def create_dataset_file(set: str, samples: list):
    with Path(set+'.csv').open("w") as file:
        for sample in samples:
            file.write(f'{sample[0]},{sample[1]}\n')


def main():
    parser = ArgumentParser()
    parser.add_argument("root-dir")
    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    samples = []
    for dir in root_dir.iterdir():
        label = int(dir.name)
        for file in dir.iterdir():
            samples.append((str(file), label))
    counts = [int(len(samples)*0.8), int(len(samples)*0.1), int(len(samples)*0.1)]
    samples = random.shuffle(samples)
    set = [], [], []
    global_cnt = 0
    for count in counts:
        for i in range(count):
            set[i].append(samples[global_cnt])
            global_cnt+=1
    train, test, val = set
