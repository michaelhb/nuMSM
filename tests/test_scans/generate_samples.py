from nuMSM_solver.common import *
import random

def random(fixed_params, ranges, num_samples, seed):
    """
    fixed_params: ModelParams object; use value of "None" for varied params
    ranges: ModelParams object; use "None" for fixed params, 2-tuple for varied params
    num_samples: integer
    seed: integer

    Notes:
        - dM sampling will be 10^X where X is uniform within (min, max)
        - All other sampling will be uniform within (min, max)
    """
    random.seed(seed)
    samples = []

    for i in range(num_samples):
        sample = {}
        for par, val in fixed_params._asdict().items():

            if val == None:
                min, max = getattr(ranges, par)
                X = random.uniform(min, max)

                if par == "dM":
                    sample[par] = 10**X
                else:
                    sample[par] = X
            else:
                sample[par] = val

        samples.append(ModelParams(**sample))

    return samples