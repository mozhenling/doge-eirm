# coding=utf-8
from algorithms.classes.ERM import ERM
from algorithms.classes.IRM import IRM
from algorithms.classes.EIRM import EIRM
from algorithms.classes.Mixup import Mixup
from algorithms.classes.ANDMask import ANDMask
from algorithms.classes.IB_IRM import IB_IRM
from algorithms.classes.RSC import RSC
from algorithms.classes.CaSN import CaSN
from algorithms.classes.RDM import RDM


ALGORITHMS = [
    'RDM',
    'CaSN',
    'ANDMask',
    'IB_IRM',
    'RSC',
    'ERM',
    'Mixup',
    'IRM',
    'EIRM',
]

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

