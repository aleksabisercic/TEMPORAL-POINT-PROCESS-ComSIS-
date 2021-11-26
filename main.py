import glob
import os
# moduls for multiprocessing
import functools
import multiprocessing
from evaluation import get_simulation_times
import argparse

def get_models(folder: str = 'best_autoput_models', model_ext: str = '*.torch'):
    """Concat multiple .csv files into single df"""
    location_of_documents = os.path.join(folder, model_ext)  # get all docs from glue
    return list(glob.glob(location_of_documents))


def main(args):
    models = get_models()
    # get partial funcion of funcion get_simulation_times()
    # in order to work with multiprocessing
    processing_funcion = functools.partial(get_simulation_times,
                                           time_upper=args.time_upper,
                                           no_sim=args.no_sim)

    # Run multiprocessing
    processes = []
    # each subset of dfs is a unique process
    for index, model in enumerate(models):
        print(f'Process model: {index + 1}, out of {len(models)}')
        p = multiprocessing.Process(target=processing_funcion, args=(model,))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()

    # get_simulation_times(model_filepath = models[0], time_upper = 400, no_sim = 1)

    print('DONE')


def parse_args():
    # Local testing with CLI args
    parser = argparse.ArgumentParser(description='Process')

    parser.add_argument('--time-upper', type=int,
                        default=400,
                        help='this is the number of steps that we want our simulation to make'
                        )
    parser.add_argument('--no-sim', type=int,
                        default=1000,
                        help='number of simulation to generate point process')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print('Loaded arguments:')
    print(args)

    import sys
    sys.path.append('./')
    sys.path.append('./models')
    main(args)
