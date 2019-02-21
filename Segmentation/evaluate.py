from subprocess import check_output
from ast import literal_eval
from Segmentation.params import regions_file
from Segmentation.utilities import fix_json_fname
import numpy as np

def evaluation(fname_to_check, regions_fname=regions_file):
    """Evaluate results with neurofinder"""
    fname_to_check = fix_json_fname(fname_to_check)
    regions_fname = fix_json_fname(regions_fname)

    # run command and get output
    res = check_output(["neurofinder", "evaluate", regions_fname, fname_to_check], universal_newlines=True).rstrip()
    res_dict = literal_eval(res)

    return res_dict


def best_res(grid_search_results):
    """Finds the best result in the res list, produced by the grid search.
    Based on the evaluate function."""
    for k in grid_search_results[0]['evaluation'].keys():   # iterate over the each of the 5 keys
        print(k, np.argmax((x['evaluation'][k] for x in grid_search_results)))
