"""This script will profile the fastbuild function."""
import cProfile
from pstats import Stats
from sidechainnet.examples import get_alphabet_protein
from tqdm import tqdm

def main():
    with_hydrogens()
    without_hydrogens()


def with_hydrogens():
    p = get_alphabet_protein()

    for i in tqdm(range(100)):
        p.fastbuild(add_hydrogens=True, inplace=True)


def without_hydrogens():
    p = get_alphabet_protein()

    for i in tqdm(range(100)):
        p.fastbuild(add_hydrogens=False, inplace=True)


if __name__ == "__main__":
    with cProfile.Profile() as pr:
        without_hydrogens()

    with open('profiling_stats_nohy.txt', 'w') as stream:
        stats = Stats(pr, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('time')
        stats.dump_stats('.prof_stats_no_hy')
        stats.print_stats()

    with cProfile.Profile() as pr2:
        with_hydrogens()

    with open('profiling_stats_why.txt', 'w') as stream:
        stats = Stats(pr2, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('time')
        stats.dump_stats('.prof_stats_yes_hy')
        stats.print_stats()
