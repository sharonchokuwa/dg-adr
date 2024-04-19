import numpy as np
import torch

domain_num_dict = {
    'messidor_2': 1744,
    'idrid': 516,
    'deepdr': 1600,
    'fgadr': 1842,
    'aptos': 3656,
    'rldr': 1593,
    'ddr': 12522,
    'eyepacs': 88702,
}

label_num_dict = {
    'messidor_2': [1016, 269, 347, 75, 35],
    'idrid': [175, 26, 163, 89, 60],
    'deepdr': [714, 186, 326, 282, 92],
    'fgadr': [100, 211, 595, 646, 286],
    'aptos': [1801, 369, 998, 193, 295],
    'rldr': [165, 336, 929, 98, 62],
    'ddr': [6266, 630, 4477, 236, 913],
    'eyepacs': [65343, 6205, 13153, 2087, 1914],
}

# this weight dict can be calculated from get_sample_weighting()
WEIGHT_DICT = {
    'messidor_2': [0.010192425774759806, 0.03849629958050544, 0.029842952700737647, 0.13807339449541284, 0.2958715596330275],
    'idrid': [0.14857142857142855, 1.0, 0.15950920245398773, 0.29213483146067415, 0.4333333333333333],
    'deepdr': [0.0415546218487395, 0.15951612903225806, 0.09101226993865032, 0.10521276595744682, 0.3225],
    'fgadr': [0.28013029315960913, 0.1327631721135588, 0.047080721539430104, 0.043363822470527724, 0.09794765495091227],
    'aptos': [0.015124711897231906, 0.07382007080464678, 0.027294194515946554, 0.14113785557986872, 0.09233764788784632],
    'rldr': [0.1217143183244878, 0.05977042417720384, 0.021617720692723885, 0.204927168607556, 0.3239171374764595],
    'ddr': [0.001552021076164807, 0.015436450894045526, 0.002172205508878419, 0.041207474844274075, 0.010651658338717066],
    'eyepacs': [0.00017039590488094467, 0.001794388334026683, 0.0008465125532301046, 0.005335016584875691, 0.005817230727604789]
}

def get_sample_weighting():
    # Calculate inverse class weights
    class_weights = {}
    for domain, class_occurrences in label_num_dict.items():
        total_samples_in_domain = domain_num_dict[domain]
        class_weights[domain] = [total_samples_in_domain / occurrence for occurrence in class_occurrences]

    # Calculate inverse domain weights
    total_samples = sum(domain_num_dict.values())
    domain_weights = {domain: total_samples / count for domain, count in domain_num_dict.items()}

    # Normalize class weights
    for domain, weights in class_weights.items():
        max_weight = max(weights)
        class_weights[domain] = [weight / max_weight for weight in weights]

    # Normalize domain weights
    max_domain_weight = max(domain_weights.values())
    domain_weights = {domain: weight / max_domain_weight for domain, weight in domain_weights.items()}

    # Calculate sample weights
    sample_weights = {}
    for domain, class_occurrences in label_num_dict.items():
        class_weight = class_weights[domain]
        domain_weight = domain_weights[domain]
        sample_weights[domain] = [class_weight[i] * domain_weight for i in range(len(class_weight))]
    
    return sample_weights


def get_sample_weights(batch_domains, batch_labels):
    batch_size = batch_labels.size()
    sample_weights = []

    for i in range(batch_size[0]):
        domain_name = batch_domains[i]
        label = batch_labels[i]

        # Retrieve the weight from the dictionary based on domain_name and label
        weight = WEIGHT_DICT[domain_name][label]
        sample_weights.append(weight)
    return torch.tensor(sample_weights).cuda()