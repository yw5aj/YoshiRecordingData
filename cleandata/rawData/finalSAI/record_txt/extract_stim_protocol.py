import numpy as np
import matplotlib.pyplot as plt
import os, re, pdb, pickle


def main():
    pass


def extract_disp_vel_acc(line):
    disp, vel, acc = [float(_) for _ in re.findall(r'\d+\.\d+', line)]
    disp = np.round(disp, decimals=2)
    disp_vel_acc = np.array([disp, vel, acc])
    return disp_vel_acc


def is_valid_protocol_line(line):
    return line.startswith('time:') and line.endswith('type: C')


def read_block(block):
    block_protocol = []
    for line in block.splitlines()[1:]:
        if is_valid_protocol_line(line):
            block_protocol.append(extract_disp_vel_acc(line).prod())
    return block_protocol


def get_protocol_no_list(protocol_prod_list):
    flattened_protocol_prod_list = np.array([item for sublist in protocol_prod_list for item in sublist])
    unique_inverse = np.unique(flattened_protocol_prod_list, return_inverse=True)[1]
    it_unique_inverse = unique_inverse.flat
    protocol_no_list = [[next(it_unique_inverse) for j in i] for i in protocol_prod_list]
    return protocol_no_list


def read_file(f):
    protocol_prod_list = []
    blocks = re.findall(r'\(\d+\)[^\(]+', f.read())
    for block in blocks:
        protocol_prod_list.append(read_block(block))
    protocol_no_list = get_protocol_no_list(protocol_prod_list)
    return protocol_no_list


if __name__ == '__main__':
    stim_order_dict = {}
    for filename in os.listdir('.'):
        if filename.endswith('.txt'):
            with open(filename, 'r') as f:
                protocol_no_list = read_file(f)
                stim_order_dict[filename[:-4]] = protocol_no_list
    with open('../stim_order.pkl', 'wb') as f:
        pickle.dump(stim_order_dict, f)
