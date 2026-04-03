import logging
logger = logging.getLogger(__name__)

import rlhf_wrapper
from rlhf_wrapper import print_info
import torch

def first_segment(reason):
    logger.info(f"first segment preferred: {reason}")
    return torch.tensor([1.0])
def second_segment(reason):
    logger.info(f"second segment preferred: {reason}")
    return torch.tensor([0.0])

def compare_metrics(name, metric, info_a, info_b, threshold):
    metric_a = metric(info_a)
    metric_b = metric(info_b)
    # we prefer one metric if it is greater than threshold than the other
    if abs(metric_a - metric_b) > threshold:
        if metric_a > metric_b:
            return first_segment(name)
        else:
            return second_segment(name)
    else:
        return None

def get_human_preference(info_a, info_b):
    print_info(info_a, info_b)

    #preferred_segment = input("Which segment did you prefer (1 if you preferred the first segment, 0 for the second, and 0.5 for ties)?")
    
    # if one spends a lot more time attacking wood then it will be preferred
    fraction_of_time_attacking_wood = compare_metrics(
        name = "fraction of time attacking wood",
        metric = rlhf_wrapper.fraction_of_time_attacking_wood,
        info_a = info_a,
        info_b = info_b, 
        threshold = 0.0005
    )
    if fraction_of_time_attacking_wood is not None:
        return fraction_of_time_attacking_wood
    
    # if one spends more time moving towards wood, then we will prefer it
    fraction_of_time_moving_towards_wood = compare_metrics(
        name = "fraction of time moving towards wood",
        metric = rlhf_wrapper.fraction_of_time_moving_towards_wood,
        info_a = info_a,
        info_b = info_b, 
        threshold = 0.001
    )
    if fraction_of_time_moving_towards_wood is not None:
        return fraction_of_time_moving_towards_wood
    
    # if one spends more time looking at wood, then we will prefer it
    fraction_of_time_looking_at_wood = compare_metrics(
        name = "fraction of time looking at wood",
        metric = rlhf_wrapper.fraction_of_time_looking_at_wood,
        info_a = info_a,
        info_b = info_b, 
        threshold = 0.001
    )
    if fraction_of_time_looking_at_wood is not None:
        return fraction_of_time_looking_at_wood
    
    # we will prefer which ever segment moves farther
    horizontal_distance_traveled = compare_metrics(
        name = "horizontal distance traveled",
        metric = rlhf_wrapper.horizontal_distance_traveled,
        info_a = info_a,
        info_b = info_b, 
        threshold = 0.00
    )
    return horizontal_distance_traveled