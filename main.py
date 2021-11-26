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
    
    dataset_type = args.dataset_type
    
    if dataset_type == 'ski':
        models = get_models(folder = 'best_ski_models')
    else:
        models = get_models()
    # get partial funcion of funcion get_simulation_times()
    # in order to work with multiprocessing
    processing_funcion = functools.partial(get_simulation_times,
                                           time_upper=args.time_upper,
                                           no_sim=args.no_sim,
                                           dataset_type = args.dataset_type)

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

    print('DONE')
    
    
def main2(args):
    """
    Optimized main()
    """
    dataset_type = args.dataset_type
    
    if dataset_type == 'ski':
        models = get_models(folder = 'best_ski_models')
    else:
        models = get_models()
    # get partial funcion of funcion get_simulation_times()
    # in order to work with multiprocessing
    processing_funcion = functools.partial(get_simulation_times,
                                           time_upper=args.time_upper,
                                           no_sim=args.no_sim,
                                           dataset_type = args.dataset_type) 
    
    num_cpus = multiprocessing.cpu_count()
    print('num_cpus {}'.format(num_cpus))

    p = multiprocessing.Pool(num_cpus)
    p.map(processing_funcion, models)
    
    print('DONE')

def test(dataset_type = 'ski'):
    
    if dataset_type == 'ski':
        models = get_models(folder = 'best_ski_models')
    else:
        models = get_models()
        
    for model in models:
        get_simulation_times(model_filepath = model,
                             time_upper = 300, 
                             no_sim = 1,
                             dataset_type = dataset_type)


def parse_args():
    # Local testing with CLI args
    parser = argparse.ArgumentParser(description='Process')

    parser.add_argument('--time-upper', type=int,
                        default=400,
                        help='this is the number of steps that we want our simulation to make'
                        )
    parser.add_argument('--no-sim', type=int,
                        default=1,
                        help='number of simulation to generate point process')

    parser.add_argument('--dataset-type', type=str,
                        default='autoput',
                        help='autoput or ski')
    return parser.parse_args()


if __name__ == "__main__":
    
     args = parse_args()
     print('Loaded arguments:')
     print(args)
    
     # import sys
    
     # sys.path.append('./')
     # sys.path.append('./models')
     main2(args)
     # print('BEGINGIN TEST')
     # test()