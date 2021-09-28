import itertools
def makeGrid(pars_dict):  # Make a grid suitable for Grid Search
    keys=pars_dict.keys()
    combinations=itertools.product(*pars_dict.values())
    return [dict(zip(keys,cc)) for cc in combinations]
    