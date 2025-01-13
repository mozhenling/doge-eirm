
import copy
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset

def re_label(labels, tran_matrix=None):
    """
    relabeling based on the tran_matrix
    inputs: a list of labels, and the N-by-N transition matrix in form of a dictionary (N is the num. of classes)
    return: a list of labels
    """
    return [random.choices(population=list(tran_matrix[lab].keys()),
                           weights=list(tran_matrix[lab].values()),
                           k=1)[0] for lab in labels]



def re_label_show(labels, num_classes, tran_matrix=None, is_show=True):
    if tran_matrix is None:
        or_labels, re_labels =  labels, labels
        return or_labels, re_labels
    else:
        re_labels = []
        or_labels = copy.deepcopy(labels) # original labels
        flip_info = []
        for i, lab in enumerate(labels):
            re_lab = random.choices(population=list(tran_matrix[lab].keys()),
                                    weights=list(tran_matrix[lab].values()),
                                    k=1)[0]
            re_labels.append(re_lab)
            if re_lab !=lab:
                flip_info.append([lab, re_lab, i])
    if is_show:
        #-- actual tran_matrix
        print("Print noisy label generation statistics:")
        for c in range(num_classes):
            num_before = len([i for i in or_labels if i == c])
            print('\t', 'num. of ', str(c), ' before: ', num_before)

            num_after = len([i for i in re_labels if i == c])
            print('\t', 'num. of ', str(c), ' after: ', num_after)

            flip = [state[1] for state in flip_info if state[0]==c]

            for cc in range(num_classes):
                if cc==c:
                    continue
                num_flip = len([i for i in flip if i == cc])
                print('\t', 'num. of ', str(c), '->', str(cc), ': ', num_flip)
                print('\t', 'flip. prob. ', str(c), '->', str(cc), ': ', round(num_flip / num_before, 3))

            print('\n')
    return or_labels, re_labels

def dataset_transform(data, labels, input_shape, num_classes, device='cuda', tran_matrix=None, is_show=False):
    if is_show:
        _, re_labels = re_label_show(labels, num_classes, tran_matrix, is_show)
    else:
        re_labels = re_label(labels, tran_matrix) if tran_matrix is not None else labels

    data, re_labels = shuffle_dataAndlabel(data, re_labels)
    x = torch.tensor(np.array(data)).view(-1, *input_shape).float().to(device)
    y = torch.tensor(re_labels).view(-1).to(device).long()
    return TensorDataset(x, y)

def shuffle_dataAndlabel(data, labels):
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    return data, labels

def sig_segmentation(data, label, seg_len, start=0, stop=None):
    '''
    This function is mainly used to segment the raw 1-d signal into samples and labels
    using the sliding window to split the data
    '''
    data_seg = []
    lab_seg = []
    start_temp, stop_temp, stop = start, seg_len, stop if stop is not None else len(data)
    while stop_temp <= stop:
        sig = data[start_temp:stop_temp]
        sig = sig.reshape(-1, 1)
        data_seg.append( (sig - np.mean(sig)) / np.std(sig) ) # z-score normalization
        lab_seg.append(label)
        start_temp += seg_len
        stop_temp += seg_len
    return data_seg, lab_seg