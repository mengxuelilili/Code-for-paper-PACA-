# 清洗数据集了，保存5个seed的模型为了显著性分析
import torch
import torch.nn as nn
import numpy as np
import random
import argparse
import os
import time
import copy
import itertools
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from roformercnn import CombinedModel
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ======================================================
# 1. 随机种子设置函数 (将在 main 中根据参数调用)
# ======================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 保持确定性，但这可能会稍微降低速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 随机种子已设置为: {seed}")

# ======================================================
# CDR 区域定义（Chothia）
# ======================================================
def getCDRPos(_loop, cdr_scheme='chothia'):
    # ASSUMES SEQUENCES ARE NUMBERED IN CHOTHIA SCHEME
    if cdr_scheme == 'chothia':
        CDRS = {'L1F': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                        '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D', '30E', '30F',
                       '30G', '30H', '30I', '31', '32', '33', '34'],
                'L2F': ['35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
                        '48', '49'],
                'L2': ['50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
                'L3F': ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
                        '73', '74', '75',
                        '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'],
                'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F',
                       '95G', '95H', '95I', '95J', '96', '97'],
                'L4F': ['98', '99', '100', '101', '102', '103', '104', '105', '106', '106A', '107'],

                'H1F': ['0', '1', '2', '3', '4', '5', '6', '6A', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                        '16', '17', '18', '19',
                        '20', '21', '22', '23', '24', '25'],
                'H1': ['26', '27', '28', '29', '30', '31', '31A', '31B', '31C', '31D', '31E', '31F', '31G', '31H',
                       '31I', '31J', '32'],
                'H2F': ['33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51'],
                'H2': ['52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K', '52L', '52M',
                       '52N', '52O', '53', '54', '55', '56'],
                'H3F': ['57', '58', '59', '59A', '60', '60A', '61', '62', '63', '64', '64A', '65', '66', '67', '68',
                        '69', '70', '71', '72', '73', '74', '75', '76', '76A', '76B', '76C', '76D', '76E', '76F', '76G',
                        '76H', '76I', '77', '78', '79', '80', '81', '82', '82A', '82B', '82C', '83',
                        '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94'],
                'H3': ['95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D',
                       '100E', '100F', '100G', '100H', '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P',
                       '100Q', '100R', '100S', '100T', '100U', '100V', '100W', '100X', '100Y', '100Z', '101', '102'],
                'H4F': ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']}
    elif cdr_scheme == 'kabat':
        CDRS = {'L1F': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                        '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D',
                       '30E', '30F', '30G', '30H', '30I', '31', '32', '33', '34'],
                'L2F': ['35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
                        '48', '49'],
                'L2': ['50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
                'L3F': ['57', '58', '59', '60', '61',
                        '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
                        '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'],
                'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F', '96', '97'],
                'L4F': ['98', '99', '100', '101', '102', '103', '104', '105', '106', '106A', '107'],

                'H1F': ['0', '1', '2', '3', '4', '5', '6', '6A', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                        '16', '17', '18', '19',
                        '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'],
                'H1': ['31', '31A', '31B', '31C', '31D', '31E', '31F', '31G', '31H', '31I', '31J', '32', '33', '34',
                       '35'],
                'H2F': ['36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'],
                'H2': ['50', '51', '52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K',
                       '52L',
                       '52M', '52N', '52O', '53', '54', '55', '56', '57', '58', '59', '60', '60A', '61', '62', '63',
                       '64', '64A', '65'],
                'H3F': ['66', '67', '68',
                        '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '82A',
                        '82B', '82C', '83',
                        '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94'],
                'H3': ['95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D', '100E', '100F', '100G',
                       '100H',
                       '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P', '100Q', '100R', '100S', '100T',
                       '100U',
                       '100V', '100W', '100X', '100Y', '100Z', '101', '102'],
                'H4F': ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']}
    elif cdr_scheme == 'abm':
        CDRS = {'L1F': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                        '15', '16', '17', '18', '19', '20', '21', '22', '23'],
                'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D',
                       '30E', '30F', '30G', '30H', '30I', '31', '32', '33', '34'],
                'L2F': ['35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
                        '48', '49'],
                'L2': ['50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
                'L3F': ['57', '58', '59', '60', '61',
                        '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
                        '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'],
                'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F', '96', '97'],
                'L4F': ['98', '99', '100', '101', '102', '103', '104', '105', '106', '106A', '107'],

                'H1F': ['0', '1', '2', '3', '4', '5', '6', '6A', '7', '8', '9', '10', '11', '12', '13', '14', '15',
                        '16', '17', '18', '19',
                        '20', '21', '22', '23', '24', '25'],
                'H1': ['26', '27', '28', '29', '30', '31', '31A', '31B', '31C', '31D', '31E', '31F', '31G', '31H',
                       '31I', '31J',
                       '32', '33', '34', '35'],
                'H2F': ['36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'],
                'H2': ['50', '51', '52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K',
                       '52L',
                       '52M', '52N', '52O', '53', '54', '55', '56', '57', '58'],
                'H3F': ['59', '60', '60A', '61', '62', '63', '64', '64A', '65', '66', '67', '68',
                        '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '82A',
                        '82B', '82C', '83',
                        '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94'],
                'H3': ['95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D',
                       '100E', '100F', '100G', '100H', '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P',
                       '100Q', '100R', '100S', '100T', '100U', '100V', '100W', '100X', '100Y', '100Z', '101', '102'],
                'H4F': ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']}
    elif cdr_scheme == 'imgt':
        ###THESE ARE IMGT CDRS IN IMGT NUMBERING
        CDRS = {'L1F': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                        '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'],
                'L1': ['27', '28', '29', '30', '31', '32', '32A', '32B', '32C', '32D', '32E', '32F', '32G',
                       '32H', '32I', '32J', '32K', '32L', '32M', '32N', '32O', '32P', '32Q', '32R', '32S',
                       '32T', '32U', '32V', '32W', '32X', '32Y', '32Z', '33Z', '33Y', '33X', '33W', '33V',
                       '33U', '33T', '33S', '33R', '33Q', '33P', '33O', '33N', '33M', '33L', '33K', '33J',
                       '33I', '33H', '33G', '33F', '33E', '33D', '33C', '33B', '33A', '33', '34', '35', '36',
                       '37', '38'],
                'L2F': ['39', '40', '41', '42', '43', '44', '45', '46', '47',
                        '48', '49', '50', '51', '52', '53', '54', '55'],
                'L2': ['56', '57', '58', '59', '60', '60A', '60B', '60C', '60D', '60E', '60F', '60G', '60H',
                       '60I', '60J', '60K', '60L', '60M', '60N', '60O', '60P', '60Q', '60R', '60S', '60T', '60U',
                       '60V', '60W', '60X', '60Y', '60Z', '61Z', '61Y', '61X', '61W', '61V', '61U', '61T', '61S',
                       '61R', '61Q', '61P', '61O', '61N', '61M', '61L', '61K', '61J', '61I', '61H', '61G', '61F',
                       '61E', '61D', '61C', '61B', '61A', '61', '62', '63', '64', '65'],
                'L3F': ['66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
                        '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88',
                        '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103',
                        '104'],
                'L3': ['105', '106', '107', '108', '109', '110', '111', '111A', '111B', '111C', '111D', '111E',
                       '111F', '111G', '111H', '111I', '111J', '111K', '111L', '111M', '111N', '111O', '111P', '111Q',
                       '111R', '111S', '111T', '111U', '111V', '111W', '111X', '111Y', '111Z',
                       '112Z', '112Y', '112X', '112W', '112V',
                       '112U', '112T', '112S', '112R', '112Q', '112P', '112O', '112N',
                       '112M', '112L', '112K', '112J', '112I', '112H', '112G', '112F', '112E', '112D', '112C', '112B',
                       '112A', '112',
                       '113', '114', '115', '116', '117'],
                'L4F': ['118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129'],

                'H1F': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                        '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'],
                'H1': ['27', '28', '29', '30', '31', '32', '32A', '32B', '32C', '32D', '32E', '32F', '32G',
                       '32H', '32I', '32J', '32K', '32L', '32M', '32N', '32O', '32P', '32Q', '32R', '32S',
                       '32T', '32U', '32V', '32W', '32X', '32Y', '32Z', '33Z', '33Y', '33X', '33W', '33V',
                       '33U', '33T', '33S', '33R', '33Q', '33P', '33O', '33N', '33M', '33L', '33K', '33J',
                       '33I', '33H', '33G', '33F', '33E', '33D', '33C', '33B', '33A', '33', '34', '35', '36',
                       '37', '38'],
                'H2F': ['39', '40', '41', '42', '43', '44', '45', '46', '47',
                        '48', '49', '50', '51', '52', '53', '54', '55'],
                'H2': ['56', '57', '58', '59', '60', '60A', '60B', '60C', '60D', '60E', '60F', '60G', '60H',
                       '60I', '60J', '60K', '60L', '60M', '60N', '60O', '60P', '60Q', '60R', '60S', '60T',
                       '60U', '60V', '60W', '60X', '60Y', '60Z', '61Z', '61Y', '61X', '61W', '61V', '61U',
                       '61T', '61S', '61R', '61Q', '61P', '61O', '61N', '61M', '61L', '61K', '61J', '61I',
                       '61H', '61G', '61F', '61E', '61D', '61C', '61B', '61A', '61', '62', '63', '64', '65'],
                'H3F': ['66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
                        '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88',
                        '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103',
                        '104'],
                'H3': ['105', '106', '107', '108', '109', '110', '111', '111A', '111B', '111C', '111D', '111E',
                       '111F', '111G', '111H', '111I', '111J', '111K', '111L', '111M', '111N', '111O', '111P', '111Q',
                       '111R', '111S', '111T', '111U', '111V', '111W', '111X', '111Y', '111Z',
                       '111AA', '111BB', '111CC', '111DD', '111EE', '111FF', '112FF', '112EE', '112DD', '112CC',
                       '112BB', '112AA',
                       '112Z', '112Y', '112X', '112W', '112V',
                       '112U', '112T', '112S', '112R', '112Q', '112P', '112O', '112N',
                       '112M', '112L', '112K', '112J', '112I', '112H', '112G', '112F', '112E', '112D', '112C', '112B',
                       '112A', '112',
                       '113', '114', '115', '116', '117'],
                'H4F': ['118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129']}

    elif cdr_scheme == 'north':
        CDRS = {
            'L1F': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
                    '18', '19', '20', '21', '22', '23'],
            'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D', '30E', '30F', '30G', '30H',
                   '30I', '31', '32', '33', '34'],
            'L2F': ['35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48'],
            'L2': ['49', '50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
            'L3F': ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
                    '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'],
            'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F', '95G', '95H',
                   '95I', '95J', '96', '97'],
            'L4F': ['98', '99', '100', '101', '102', '103', '104', '105', '106', '106A', '107'],
            'H1F': ['0', '1', '2', '3', '4', '5', '6', '6A', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
                    '17', '18', '19', '20', '21', '22'],
            'H1': ['23', '24', '25', '26', '27', '28', '29', '30', '31', '31A', '31B', '31C', '31D', '31E', '31F',
                   '31G', '31H', '31I', '31J', '32', '33', '34', '35'],
            'H2F': ['36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'],
            'H2': ['50', '51', '52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K', '52L',
                   '52M', '52N', '52O', '53', '54', '55', '56', '57', '58'],
            'H3F': ['59', '59A', '60', '60A', '61', '62', '63', '64', '64A', '65', '66', '67', '68', '69', '70', '71',
                    '72', '73', '74', '75', '76', '76A', '76B', '76C', '76D', '76E', '76F', '76G', '76H', '76I', '77',
                    '78', '79', '80', '81', '82', '82A', '82B', '82C', '83', '84', '85', '86', '87', '88', '89', '90',
                    '91', '92'],
            'H3': ['93', '94', '95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D', '100E', '100F',
                   '100G', '100H', '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P', '100Q', '100R',
                   '100S', '100T', '100U', '100V', '100W', '100X', '100Y', '100Z', '101', '102'],
            'H4F': ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']}

    return CDRS[_loop]

# ======================================================
# Dataset & Collate (保持不变)
# ======================================================
class ListDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    X_a_list = [torch.tensor(item[0], dtype=torch.float32) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch]
    X_b_list = [torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch]
    ag_list = [torch.tensor(item[2], dtype=torch.float32) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch]
    y_list = [torch.tensor(item[3], dtype=torch.float32) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch]

    if not X_a_list:
        return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

    max_len = max(
        max(x.shape[0] for x in X_a_list),
        max(x.shape[0] for x in X_b_list),
        max(x.shape[0] for x in ag_list)
    )

    def pad_to_len(x, L):
        if x.shape[0] < L:
            pad = torch.zeros(L - x.shape[0], x.shape[1], dtype=x.dtype)
            return torch.cat([x, pad], dim=0)
        else:
            return x[:L]

    X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
    X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
    ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])
    y_tensor = torch.stack(y_list)
    return X_a_padded, X_b_padded, ag_padded, y_tensor

# ======================================================
# Evaluation (保持不变)
# ======================================================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_a, X_b, ag, y in loader:
            if X_a.shape[0] == 0: continue
            X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
            pred = model(X_b, X_a, ag).view(-1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())

    if len(y_true) == 0:
        return {"MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "PCC": 0.0}

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pcc = pearsonr(y_true, y_pred)[0] if len(set(y_true)) > 1 else 0.0
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc}

# ======================================================
# Data Loading & Splitting
# ======================================================
def load_dataset(path):
    d = torch.load(path, map_location="cpu")
    return {
        "X_a": d["X_a"].cpu().numpy(),
        "X_b": d["X_b"].cpu().numpy(),
        "antigen": d["antigen"].cpu().numpy(),
        "y": d["y"].cpu().numpy()
    }

def split_dataset(data, test_size=0.2, val_size=0.2, seed=42):
    X_a, X_b, ag, y = data["X_a"], data["X_b"], data["antigen"], data["y"]
    
    # 【关键修改】：如果想让数据划分也随种子变化，将 random_state=42 改为 random_state=seed
    # 这里为了保守起见，暂时保持固定划分，仅改变模型初始化。如需完全独立实验，请取消注释下一行的 seed 变量
    # current_split_seed = seed 
    current_split_seed = 42 

    X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
        X_a, X_b, ag, y, test_size=test_size, random_state=current_split_seed
    )
    
    val_ratio = val_size / (1 - test_size)
    X_a_tr, X_a_val, X_b_tr, X_b_val, ag_tr, ag_val, y_tr, y_val = train_test_split(
        X_a_tv, X_b_tv, ag_tv, y_tv, test_size=val_ratio, random_state=current_split_seed
    )
    
    return {
        "train": (X_a_tr, X_b_tr, ag_tr, y_tr),
        "val": (X_a_val, X_b_val, ag_val, y_val),
        "test": (X_a_test, X_b_test, ag_test, y_test)
    }

# ======================================================
# Trainer (保持不变)
# ======================================================
class TrainerWithScheduler:
    def __init__(self, model, train_loader, val_loader, params, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.opt = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=params.get("lr_factor", 0.5),
                                           patience=params.get("scheduler_patience", 3),
                                           min_lr=params.get("min_lr", 1e-6), verbose=False)
        self.criterion = nn.MSELoss()
        self.epochs = params["epochs"]
        self.patience = params["patience"]

    def train(self):
        best_mse = np.inf
        best_state = None
        wait = 0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            num_batches = 0
            for X_a, X_b, ag, y in self.train_loader:
                if X_a.shape[0] == 0: continue
                X_a, X_b, ag, y = X_a.to(self.device), X_b.to(self.device), ag.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                pred = self.model(X_b, X_a, ag).view(-1)
                loss = self.criterion(pred, y)
                loss.backward()
                self.opt.step()
                total_loss += loss.item()
                num_batches += 1
            
            if num_batches == 0: continue
            avg_train_loss = total_loss / num_batches
            val_metrics = evaluate(self.model, self.val_loader, self.device)
            val_mse = val_metrics["MSE"]
            self.scheduler.step(val_mse)
            
            # 减少打印频率，避免日志过多
            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep {epoch:02d} | Loss: {avg_train_loss:.4f} | Val MSE: {val_mse:.4f}")

            if val_mse < best_mse:
                best_mse = val_mse
                best_state = copy.deepcopy(self.model.state_dict())
                wait = 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.model

# ======================================================
# MAIN (核心修改部分)
# ======================================================
def main():
    # 1. 解析命令行参数
    parser = argparse.ArgumentParser(description="Train PACA with specific seed")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for this run")
    args = parser.parse_args()
    
    # 2. 设置种子
    set_seed(args.seed)
    current_seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"💻 Using device: {device}")

    # 数据路径配置
    base_dir = "/tmp/AbAgCDR/data"
    paths = {
        "train": os.path.join(base_dir, "train_data.pt"),
        "abbind": os.path.join(base_dir, "abbind_data.pt"),
        "sabdab": os.path.join(base_dir, "sabdab_data.pt"),
        "skempi": os.path.join(base_dir, "skempi_data.pt")
    }

    # 输出目录配置 (根据 seed 动态创建或区分)
    model_save_dir = "/tmp/AbAgCDR/model"
    result_save_dir = "/tmp/AbAgCDR/resultsxin"
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(result_save_dir, exist_ok=True)

    # Load and split
    all_splits = {}
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"⚠️ Warning: File not found: {path}")
            continue
        data = load_dataset(path)
        # 传入 seed 以支持动态划分（如果 split_dataset 内部使用了该参数）
        all_splits[name] = split_dataset(data, seed=current_seed)

    if not all_splits:
        print("❌ Error: No data loaded.")
        return

    # Prepare Samples
    all_train_samples = []
    sample_weights = []
    dataset_weights = {'train': 4.0, 'abbind': 1.0, 'sabdab': 1.5, 'skempi': 1.5}

    for name in paths.keys():
        if name not in all_splits: continue
        tr = all_splits[name]["train"]
        w = dataset_weights.get(name, 0.1)
        for i in range(len(tr[3])):
            all_train_samples.append((tr[0][i], tr[1][i], tr[2][i], tr[3][i]))
            sample_weights.append(w)

    val_samples = []
    for name, split in all_splits.items():
        va = split["val"]
        for i in range(len(va[3])):
            val_samples.append((va[0][i], va[1][i], va[2][i], va[3][i]))

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Grid Search Params (简化版以节省时间，可根据需要恢复完整版)
    param_grid = {
        'lr': [1e-4, 5e-4],
        'batch_size': [16, 32],
        'epochs': [60],
        'patience': [15],
        'weight_decay': [1e-5, 1e-4],
        'scheduler_patience': [3],
        'lr_factor': [0.5],
        'min_lr': [1e-6]
    }

    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"🚀 Starting Grid Search for Seed {current_seed} ({len(param_combinations)} trials)...")

    best_score = -np.inf
    best_params = None
    best_model_state = None
    
    # Load scaler if exists (assuming same for all seeds)
    label_scaler = None
    first_path = list(paths.values())[0]
    if os.path.exists(first_path):
        temp_d = torch.load(first_path, map_location="cpu")
        label_scaler = temp_d.get("label_scaler", None)

    for trial_idx, params in enumerate(param_combinations):
        if trial_idx % 5 == 0:
            print(f"  ... Trial {trial_idx+1}/{len(param_combinations)}")
        
        current_bs = params['batch_size']
        train_loader = DataLoader(ListDataset(all_train_samples), batch_size=current_bs, sampler=sampler, collate_fn=collate_fn, shuffle=False)
        val_loader = DataLoader(ListDataset(val_samples), batch_size=current_bs, shuffle=False, collate_fn=collate_fn)

        model = CombinedModel(
            [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
            [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
            num_heads=2, embed_dim=532, antigen_embed_dim=500
        )

        trainer = TrainerWithScheduler(model, train_loader, val_loader, params, device)
        trained_model = trainer.train()

        val_metrics = evaluate(trained_model, val_loader, device)
        score = val_metrics['PCC']

        if score > best_score:
            best_score = score
            best_params = copy.deepcopy(params)
            best_model_state = copy.deepcopy(trained_model.state_dict())

    # Save Best Model for THIS Seed
    model_filename = f"PWAARPEbest_model_seed_{current_seed}.pth"
    model_path = os.path.join(model_save_dir, model_filename)
    
    if best_model_state is not None:
        torch.save({
            'model_state_dict': best_model_state,
            'params': best_params,
            'label_scaler': label_scaler,
            'seed': current_seed
        }, model_path)
        print(f"💾 Best model for Seed {current_seed} saved to: {model_path}")
    else:
        print("❌ No model trained successfully for this seed.")
        return

    # Final Evaluation & CSV Generation
    final_model = CombinedModel(
        [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
        [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
        num_heads=2, embed_dim=532, antigen_embed_dim=500
    )
    final_model.load_state_dict(best_model_state)
    final_model.to(device)
    final_model.eval()

    print(f"\n🧪 Generating Predictions for Seed {current_seed}...")

    # Helper to save predictions
    def save_predictions(name, test_data_tuple, filename_suffix):
        samples = []
        for i in range(len(test_data_tuple[3])):
            samples.append((test_data_tuple[0][i], test_data_tuple[1][i], test_data_tuple[2][i], test_data_tuple[3][i]))
        
        loader = DataLoader(ListDataset(samples), batch_size=32, shuffle=False, collate_fn=collate_fn)
        
        y_true_list = []
        y_pred_list = []
        indices = [] # 如果需要索引列

        with torch.no_grad():
            for idx, (X_a, X_b, ag, y) in enumerate(loader):
                if X_a.shape[0] == 0: continue
                X_a, X_b, ag = X_a.to(device), X_b.to(device), ag.to(device)
                pred = final_model(X_b, X_a, ag).view(-1)
                
                y_true_list.extend(y.cpu().numpy())
                y_pred_list.extend(pred.cpu().numpy())
                # 生成简单的索引 (全局索引可能需要更复杂的逻辑，这里用相对索引)
                start_idx = idx * loader.batch_size
                indices.extend(range(start_idx, start_idx + len(y)))

        import pandas as pd
        df = pd.DataFrame({
            'Index': indices,
            'true_ddg': y_true_list,
            'pred_ddg': y_pred_list
        })
        
        out_path = os.path.join(result_save_dir, filename_suffix)
        df.to_csv(out_path, index=False)
        print(f"✅ Saved predictions for {name}: {out_path}")
        
        # Print metrics
        mae = mean_absolute_error(y_true_list, y_pred_list)
        pcc = pearsonr(y_true_list, y_pred_list)[0] if len(set(y_true_list))>1 else 0
        print(f"   -> MAE: {mae:.4f}, PCC: {pcc:.4f}")
        return mae, pcc

    results_summary = {}

    # Generate for each dataset
    for name, split in all_splits.items():
        te = split["test"]
        file_name = f"PWAARPE{name}_predictions_seed_{current_seed}.csv"
        mae, pcc = save_predictions(name, te, file_name)
        results_summary[name] = {'MAE': mae, 'PCC': pcc}

    # Summary
    print("\n" + "="*60)
    print(f"🏆 SEED {current_seed} COMPLETE")
    print("="*60)
    for name, metrics in results_summary.items():
        print(f"{name.upper():10} | MAE: {metrics['MAE']:.4f} | PCC: {metrics['PCC']:.4f}")
    print("="*60)
    print(f"💡 下一步：请对 seeds 0, 1, 2, 3, 4 重复此过程，然后运行统计脚本。")

if __name__ == "__main__":
    main()

# # 这个是最新去掉独立测试集benchmark数据后的测试集，清洗数据集了,不保存最佳模型
# import torch
# import torch.nn as nn
# import numpy as np
# import random
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# # 请确保 roformercnn.py 在当前目录或 PYTHONPATH 中
# from roformercnn import CombinedModel
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import itertools
# import copy
# import os
# import time

# # ======================================================
# # 固定随机种子
# # ======================================================
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

# # ======================================================
# # CDR 区域定义（Chothia）
# # ======================================================
# def getCDRPos(_loop, cdr_scheme='chothia'):
#     # ASSUMES SEQUENCES ARE NUMBERED IN CHOTHIA SCHEME
#     if cdr_scheme == 'chothia':
#         CDRS = {'L1F': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
#                         '15', '16', '17', '18', '19', '20', '21', '22', '23'],
#                 'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D', '30E', '30F',
#                        '30G', '30H', '30I', '31', '32', '33', '34'],
#                 'L2F': ['35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
#                         '48', '49'],
#                 'L2': ['50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
#                 'L3F': ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
#                         '73', '74', '75',
#                         '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'],
#                 'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F',
#                        '95G', '95H', '95I', '95J', '96', '97'],
#                 'L4F': ['98', '99', '100', '101', '102', '103', '104', '105', '106', '106A', '107'],

#                 'H1F': ['0', '1', '2', '3', '4', '5', '6', '6A', '7', '8', '9', '10', '11', '12', '13', '14', '15',
#                         '16', '17', '18', '19',
#                         '20', '21', '22', '23', '24', '25'],
#                 'H1': ['26', '27', '28', '29', '30', '31', '31A', '31B', '31C', '31D', '31E', '31F', '31G', '31H',
#                        '31I', '31J', '32'],
#                 'H2F': ['33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51'],
#                 'H2': ['52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K', '52L', '52M',
#                        '52N', '52O', '53', '54', '55', '56'],
#                 'H3F': ['57', '58', '59', '59A', '60', '60A', '61', '62', '63', '64', '64A', '65', '66', '67', '68',
#                         '69', '70', '71', '72', '73', '74', '75', '76', '76A', '76B', '76C', '76D', '76E', '76F', '76G',
#                         '76H', '76I', '77', '78', '79', '80', '81', '82', '82A', '82B', '82C', '83',
#                         '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94'],
#                 'H3': ['95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D',
#                        '100E', '100F', '100G', '100H', '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P',
#                        '100Q', '100R', '100S', '100T', '100U', '100V', '100W', '100X', '100Y', '100Z', '101', '102'],
#                 'H4F': ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']}
#     elif cdr_scheme == 'kabat':
#         CDRS = {'L1F': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
#                         '15', '16', '17', '18', '19', '20', '21', '22', '23'],
#                 'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D',
#                        '30E', '30F', '30G', '30H', '30I', '31', '32', '33', '34'],
#                 'L2F': ['35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
#                         '48', '49'],
#                 'L2': ['50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
#                 'L3F': ['57', '58', '59', '60', '61',
#                         '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
#                         '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'],
#                 'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F', '96', '97'],
#                 'L4F': ['98', '99', '100', '101', '102', '103', '104', '105', '106', '106A', '107'],

#                 'H1F': ['0', '1', '2', '3', '4', '5', '6', '6A', '7', '8', '9', '10', '11', '12', '13', '14', '15',
#                         '16', '17', '18', '19',
#                         '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30'],
#                 'H1': ['31', '31A', '31B', '31C', '31D', '31E', '31F', '31G', '31H', '31I', '31J', '32', '33', '34',
#                        '35'],
#                 'H2F': ['36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'],
#                 'H2': ['50', '51', '52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K',
#                        '52L',
#                        '52M', '52N', '52O', '53', '54', '55', '56', '57', '58', '59', '60', '60A', '61', '62', '63',
#                        '64', '64A', '65'],
#                 'H3F': ['66', '67', '68',
#                         '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '82A',
#                         '82B', '82C', '83',
#                         '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94'],
#                 'H3': ['95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D', '100E', '100F', '100G',
#                        '100H',
#                        '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P', '100Q', '100R', '100S', '100T',
#                        '100U',
#                        '100V', '100W', '100X', '100Y', '100Z', '101', '102'],
#                 'H4F': ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']}
#     elif cdr_scheme == 'abm':
#         CDRS = {'L1F': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
#                         '15', '16', '17', '18', '19', '20', '21', '22', '23'],
#                 'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D',
#                        '30E', '30F', '30G', '30H', '30I', '31', '32', '33', '34'],
#                 'L2F': ['35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47',
#                         '48', '49'],
#                 'L2': ['50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
#                 'L3F': ['57', '58', '59', '60', '61',
#                         '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
#                         '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'],
#                 'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F', '96', '97'],
#                 'L4F': ['98', '99', '100', '101', '102', '103', '104', '105', '106', '106A', '107'],

#                 'H1F': ['0', '1', '2', '3', '4', '5', '6', '6A', '7', '8', '9', '10', '11', '12', '13', '14', '15',
#                         '16', '17', '18', '19',
#                         '20', '21', '22', '23', '24', '25'],
#                 'H1': ['26', '27', '28', '29', '30', '31', '31A', '31B', '31C', '31D', '31E', '31F', '31G', '31H',
#                        '31I', '31J',
#                        '32', '33', '34', '35'],
#                 'H2F': ['36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'],
#                 'H2': ['50', '51', '52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K',
#                        '52L',
#                        '52M', '52N', '52O', '53', '54', '55', '56', '57', '58'],
#                 'H3F': ['59', '60', '60A', '61', '62', '63', '64', '64A', '65', '66', '67', '68',
#                         '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '82A',
#                         '82B', '82C', '83',
#                         '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94'],
#                 'H3': ['95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D',
#                        '100E', '100F', '100G', '100H', '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P',
#                        '100Q', '100R', '100S', '100T', '100U', '100V', '100W', '100X', '100Y', '100Z', '101', '102'],
#                 'H4F': ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']}
#     elif cdr_scheme == 'imgt':
#         ###THESE ARE IMGT CDRS IN IMGT NUMBERING
#         CDRS = {'L1F': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
#                         '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'],
#                 'L1': ['27', '28', '29', '30', '31', '32', '32A', '32B', '32C', '32D', '32E', '32F', '32G',
#                        '32H', '32I', '32J', '32K', '32L', '32M', '32N', '32O', '32P', '32Q', '32R', '32S',
#                        '32T', '32U', '32V', '32W', '32X', '32Y', '32Z', '33Z', '33Y', '33X', '33W', '33V',
#                        '33U', '33T', '33S', '33R', '33Q', '33P', '33O', '33N', '33M', '33L', '33K', '33J',
#                        '33I', '33H', '33G', '33F', '33E', '33D', '33C', '33B', '33A', '33', '34', '35', '36',
#                        '37', '38'],
#                 'L2F': ['39', '40', '41', '42', '43', '44', '45', '46', '47',
#                         '48', '49', '50', '51', '52', '53', '54', '55'],
#                 'L2': ['56', '57', '58', '59', '60', '60A', '60B', '60C', '60D', '60E', '60F', '60G', '60H',
#                        '60I', '60J', '60K', '60L', '60M', '60N', '60O', '60P', '60Q', '60R', '60S', '60T', '60U',
#                        '60V', '60W', '60X', '60Y', '60Z', '61Z', '61Y', '61X', '61W', '61V', '61U', '61T', '61S',
#                        '61R', '61Q', '61P', '61O', '61N', '61M', '61L', '61K', '61J', '61I', '61H', '61G', '61F',
#                        '61E', '61D', '61C', '61B', '61A', '61', '62', '63', '64', '65'],
#                 'L3F': ['66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
#                         '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88',
#                         '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103',
#                         '104'],
#                 'L3': ['105', '106', '107', '108', '109', '110', '111', '111A', '111B', '111C', '111D', '111E',
#                        '111F', '111G', '111H', '111I', '111J', '111K', '111L', '111M', '111N', '111O', '111P', '111Q',
#                        '111R', '111S', '111T', '111U', '111V', '111W', '111X', '111Y', '111Z',
#                        '112Z', '112Y', '112X', '112W', '112V',
#                        '112U', '112T', '112S', '112R', '112Q', '112P', '112O', '112N',
#                        '112M', '112L', '112K', '112J', '112I', '112H', '112G', '112F', '112E', '112D', '112C', '112B',
#                        '112A', '112',
#                        '113', '114', '115', '116', '117'],
#                 'L4F': ['118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129'],

#                 'H1F': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
#                         '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26'],
#                 'H1': ['27', '28', '29', '30', '31', '32', '32A', '32B', '32C', '32D', '32E', '32F', '32G',
#                        '32H', '32I', '32J', '32K', '32L', '32M', '32N', '32O', '32P', '32Q', '32R', '32S',
#                        '32T', '32U', '32V', '32W', '32X', '32Y', '32Z', '33Z', '33Y', '33X', '33W', '33V',
#                        '33U', '33T', '33S', '33R', '33Q', '33P', '33O', '33N', '33M', '33L', '33K', '33J',
#                        '33I', '33H', '33G', '33F', '33E', '33D', '33C', '33B', '33A', '33', '34', '35', '36',
#                        '37', '38'],
#                 'H2F': ['39', '40', '41', '42', '43', '44', '45', '46', '47',
#                         '48', '49', '50', '51', '52', '53', '54', '55'],
#                 'H2': ['56', '57', '58', '59', '60', '60A', '60B', '60C', '60D', '60E', '60F', '60G', '60H',
#                        '60I', '60J', '60K', '60L', '60M', '60N', '60O', '60P', '60Q', '60R', '60S', '60T',
#                        '60U', '60V', '60W', '60X', '60Y', '60Z', '61Z', '61Y', '61X', '61W', '61V', '61U',
#                        '61T', '61S', '61R', '61Q', '61P', '61O', '61N', '61M', '61L', '61K', '61J', '61I',
#                        '61H', '61G', '61F', '61E', '61D', '61C', '61B', '61A', '61', '62', '63', '64', '65'],
#                 'H3F': ['66', '67', '68', '69', '70', '71', '72', '73', '74', '75',
#                         '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88',
#                         '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103',
#                         '104'],
#                 'H3': ['105', '106', '107', '108', '109', '110', '111', '111A', '111B', '111C', '111D', '111E',
#                        '111F', '111G', '111H', '111I', '111J', '111K', '111L', '111M', '111N', '111O', '111P', '111Q',
#                        '111R', '111S', '111T', '111U', '111V', '111W', '111X', '111Y', '111Z',
#                        '111AA', '111BB', '111CC', '111DD', '111EE', '111FF', '112FF', '112EE', '112DD', '112CC',
#                        '112BB', '112AA',
#                        '112Z', '112Y', '112X', '112W', '112V',
#                        '112U', '112T', '112S', '112R', '112Q', '112P', '112O', '112N',
#                        '112M', '112L', '112K', '112J', '112I', '112H', '112G', '112F', '112E', '112D', '112C', '112B',
#                        '112A', '112',
#                        '113', '114', '115', '116', '117'],
#                 'H4F': ['118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129']}

#     elif cdr_scheme == 'north':
#         CDRS = {
#             'L1F': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17',
#                     '18', '19', '20', '21', '22', '23'],
#             'L1': ['24', '25', '26', '27', '28', '29', '30', '30A', '30B', '30C', '30D', '30E', '30F', '30G', '30H',
#                    '30I', '31', '32', '33', '34'],
#             'L2F': ['35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48'],
#             'L2': ['49', '50', '51', '51A', '52', '52A', '52B', '52C', '52D', '53', '54', '55', '56'],
#             'L3F': ['57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72',
#                     '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88'],
#             'L3': ['89', '90', '91', '92', '93', '94', '95', '95A', '95B', '95C', '95D', '95E', '95F', '95G', '95H',
#                    '95I', '95J', '96', '97'],
#             'L4F': ['98', '99', '100', '101', '102', '103', '104', '105', '106', '106A', '107'],
#             'H1F': ['0', '1', '2', '3', '4', '5', '6', '6A', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16',
#                     '17', '18', '19', '20', '21', '22'],
#             'H1': ['23', '24', '25', '26', '27', '28', '29', '30', '31', '31A', '31B', '31C', '31D', '31E', '31F',
#                    '31G', '31H', '31I', '31J', '32', '33', '34', '35'],
#             'H2F': ['36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49'],
#             'H2': ['50', '51', '52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K', '52L',
#                    '52M', '52N', '52O', '53', '54', '55', '56', '57', '58'],
#             'H3F': ['59', '59A', '60', '60A', '61', '62', '63', '64', '64A', '65', '66', '67', '68', '69', '70', '71',
#                     '72', '73', '74', '75', '76', '76A', '76B', '76C', '76D', '76E', '76F', '76G', '76H', '76I', '77',
#                     '78', '79', '80', '81', '82', '82A', '82B', '82C', '83', '84', '85', '86', '87', '88', '89', '90',
#                     '91', '92'],
#             'H3': ['93', '94', '95', '96', '97', '98', '99', '100', '100A', '100B', '100C', '100D', '100E', '100F',
#                    '100G', '100H', '100I', '100J', '100K', '100L', '100M', '100N', '100O', '100P', '100Q', '100R',
#                    '100S', '100T', '100U', '100V', '100W', '100X', '100Y', '100Z', '101', '102'],
#             'H4F': ['103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113']}

#     return CDRS[_loop]


# # ======================================================
# # Simple list-based dataset
# # ======================================================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]

# # ======================================================
# # Collate function with dynamic padding per batch
# # ======================================================
# def collate_fn(batch):
#     X_a_list = [torch.tensor(item[0], dtype=torch.float32) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch]
#     X_b_list = [torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch]
#     ag_list = [torch.tensor(item[2], dtype=torch.float32) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch]
#     y_list = [torch.tensor(item[3], dtype=torch.float32) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch]

#     if not X_a_list:
#         return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)

#     max_len = max(
#         max(x.shape[0] for x in X_a_list),
#         max(x.shape[0] for x in X_b_list),
#         max(x.shape[0] for x in ag_list)
#     )

#     def pad_to_len(x, L):
#         if x.shape[0] < L:
#             pad = torch.zeros(L - x.shape[0], x.shape[1], dtype=x.dtype)
#             return torch.cat([x, pad], dim=0)
#         else:
#             return x[:L]

#     X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
#     X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
#     ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])
#     y_tensor = torch.stack(y_list)

#     return X_a_padded, X_b_padded, ag_padded, y_tensor

# # ======================================================
# # Evaluation
# # ======================================================
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []

#     with torch.no_grad():
#         for X_a, X_b, ag, y in loader:
#             if X_a.shape[0] == 0: continue
#             X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
#             pred = model(X_b, X_a, ag).view(-1)
#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())

#     if len(y_true) == 0:
#         return {"MSE": 0.0, "RMSE": 0.0, "MAE": 0.0, "R2": 0.0, "PCC": 0.0}

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
    
#     if len(set(y_true)) > 1:
#         pcc = pearsonr(y_true, y_pred)[0]
#     else:
#         pcc = 0.0

#     return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc}

# # ======================================================
# # Load dataset
# # ======================================================
# def load_dataset(path):
#     d = torch.load(path, map_location="cpu")
#     return {
#         "X_a": d["X_a"].cpu().numpy(),
#         "X_b": d["X_b"].cpu().numpy(),
#         "antigen": d["antigen"].cpu().numpy(),
#         "y": d["y"].cpu().numpy()
#     }

# # ======================================================
# # Split dataset (Strict separation)
# # ======================================================
# def split_dataset(data, test_size=0.2, val_size=0.2):
#     X_a, X_b, ag, y = data["X_a"], data["X_b"], data["antigen"], data["y"]
    
#     X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
#         X_a, X_b, ag, y, test_size=test_size, random_state=42
#     )
    
#     val_ratio = val_size / (1 - test_size)
#     X_a_tr, X_a_val, X_b_tr, X_b_val, ag_tr, ag_val, y_tr, y_val = train_test_split(
#         X_a_tv, X_b_tv, ag_tv, y_tv, test_size=val_ratio, random_state=42
#     )
    
#     return {
#         "train": (X_a_tr, X_b_tr, ag_tr, y_tr),
#         "val": (X_a_val, X_b_val, ag_val, y_val),
#         "test": (X_a_test, X_b_test, ag_test, y_test)
#     }

# # ======================================================
# # Enhanced Trainer with LR Scheduler
# # ======================================================
# class TrainerWithScheduler:
#     def __init__(self, model, train_loader, val_loader, params, device):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = device
#         self.opt = optim.Adam(
#             model.parameters(),
#             lr=params["lr"],
#             weight_decay=params["weight_decay"]
#         )
#         self.scheduler = ReduceLROnPlateau(
#             self.opt,
#             mode='min',
#             factor=params.get("lr_factor", 0.5),
#             patience=params.get("scheduler_patience", 3),
#             min_lr=params.get("min_lr", 1e-6),
#             verbose=False
#         )
#         self.criterion = nn.MSELoss()
#         self.epochs = params["epochs"]
#         self.patience = params["patience"]

#     def train(self):
#         best_mse = np.inf
#         best_state = None
#         wait = 0

#         for epoch in range(1, self.epochs + 1):
#             self.model.train()
#             total_loss = 0.0
#             num_batches = 0

#             for X_a, X_b, ag, y in self.train_loader:
#                 if X_a.shape[0] == 0: continue
#                 X_a, X_b, ag, y = X_a.to(self.device), X_b.to(self.device), ag.to(self.device), y.to(self.device)
#                 self.opt.zero_grad()
#                 pred = self.model(X_b, X_a, ag).view(-1)
#                 loss = self.criterion(pred, y)
#                 loss.backward()
#                 self.opt.step()

#                 total_loss += loss.item()
#                 num_batches += 1

#             if num_batches == 0:
#                 print(f"Epoch {epoch}: No batches found.")
#                 continue

#             avg_train_loss = total_loss / num_batches
#             val_metrics = evaluate(self.model, self.val_loader, self.device)
#             val_mse = val_metrics["MSE"]

#             self.scheduler.step(val_mse)

#             print(f"Epoch {epoch:02d}/{self.epochs} | "
#                   f"Train Loss: {avg_train_loss:.6f} | "
#                   f"Val MSE: {val_mse:.4f} | "
#                   f"Val R²: {val_metrics['R2']:.4f} | "
#                   f"Val PCC: {val_metrics['PCC']:.4f}")

#             if val_mse < best_mse:
#                 best_mse = val_mse
#                 best_state = copy.deepcopy(self.model.state_dict())
#                 wait = 0
#             else:
#                 wait += 1
#                 if wait >= self.patience:
#                     print("🛑 Early stopping triggered.")
#                     break

#         if best_state is not None:
#             self.model.load_state_dict(best_state)
#         return self.model

# # ======================================================
# # MAIN
# # ======================================================
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # 数据路径配置
#     paths = {
#         "train": "/tmp/AbAgCDR/data/train_data.pt",
#         "abbind": "/tmp/AbAgCDR/data/abbind_data.pt",
#         "sabdab": "/tmp/AbAgCDR/data/sabdab_data.pt",
#         "skempi": "/tmp/AbAgCDR/data/skempi_data.pt"
#     }

#     # Load and split training datasets
#     all_splits = {}
#     for name, path in paths.items():
#         if not os.path.exists(path):
#             print(f"⚠️ Warning: File not found: {path}")
#             continue
#         data = load_dataset(path)
#         all_splits[name] = split_dataset(data)

#     if not all_splits:
#         print("❌ Error: No data loaded. Exiting.")
#         return

#     # ========================
#     # STEP 1: Collect samples WITH dataset label (Weighted Sampling)
#     # ========================
#     all_train_samples = []
#     sample_weights = []
    
#     dataset_weights = {
#         'train': 4.0,
#         'abbind': 1.0,
#         'sabdab': 1.5,
#         'skempi': 1.5
#     }


#     for name in paths.keys():
#         if name not in all_splits: continue
#         tr = all_splits[name]["train"]
#         w = dataset_weights.get(name, 0.1)
#         for i in range(len(tr[3])):
#             all_train_samples.append((tr[0][i], tr[1][i], tr[2][i], tr[3][i]))
#             sample_weights.append(w)

#     val_samples = []
#     for name, split in all_splits.items():
#         va = split["val"]
#         for i in range(len(va[3])):
#             val_samples.append((va[0][i], va[1][i], va[2][i], va[3][i]))

#     if len(all_train_samples) == 0:
#         print("❌ Error: No training samples found.")
#         return

#     sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(sample_weights),
#         replacement=True
#     )

#     # ========================
#     # STEP 2: Hyperparameter Grid Search
#     # ========================
#     param_grid = {
#         'lr': [1e-4, 5e-4],       # 宽范围
#         'batch_size': [16, 32],
#         'epochs': [60],
#         'patience': [15],
#         'weight_decay': [1e-5, 1e-4],    
#         'scheduler_patience': [3],
#         'lr_factor': [0.5],
#         'min_lr': [1e-6]
#     }


#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     print(f"🚀 Starting grid search over {len(param_combinations)} combinations...")

#     best_score = -np.inf
#     best_params = None
#     best_model_state = None  # 仅在内存中存储最佳权重
    
#     # ❌ 删除了 label_scaler 的加载逻辑 (如果不需要反归一化输出，可以不要)
#     # 如果需要反归一化，请确保你的数据加载时已经处理好了，或者从第一个文件读取 scaler 但不保存模型
#     label_scaler = None 
#     if paths:
#         first_path = list(paths.values())[0]
#         if os.path.exists(first_path):
#             temp_d = torch.load(first_path, map_location="cpu")
#             label_scaler = temp_d.get("label_scaler", None)

#     original_train_samples = all_train_samples
#     original_val_samples = val_samples

#     # ❌ 删除了 save_path 定义和目录创建
#     # save_path = "/tmp/AbAgCDR/model/best_model3.pth" 
#     # os.makedirs(os.path.dirname(save_path), exist_ok=True)

#     for trial_idx, params in enumerate(param_combinations):
#         print(f"\n{'='*50}")
#         print(f"Trial {trial_idx+1}/{len(param_combinations)} | Params: {params}")
#         print('='*50)

#         current_bs = params['batch_size']
        
#         train_loader = DataLoader(
#             ListDataset(original_train_samples),
#             batch_size=current_bs,
#             sampler=sampler,
#             collate_fn=collate_fn,
#             shuffle=False
#         )
#         val_loader = DataLoader(
#             ListDataset(original_val_samples),
#             batch_size=current_bs,
#             shuffle=False,
#             collate_fn=collate_fn
#         )

#         model = CombinedModel(
#             [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#             [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#             num_heads=2,
#             embed_dim=532,
#             antigen_embed_dim=500
#         )

#         trainer = TrainerWithScheduler(model, train_loader, val_loader, params, device)
#         trained_model = trainer.train()

#         val_metrics = evaluate(trained_model, val_loader, device)
#         score = val_metrics['PCC']

#         print(f"✅ Val PCC: {score:.4f}")

#         if score > best_score:
#             best_score = score
#             best_params = copy.deepcopy(params)
#             # ✅ 仅在内存中复制最佳状态字典，不写入磁盘
#             best_model_state = copy.deepcopy(trained_model.state_dict())
            
#             # ❌ 删除了 torch.save (...)
#             print(f"🎉 New best model found in memory (No file saved).")

#     # ========================
#     # STEP 3: Final evaluation with best model (From Memory)
#     # ========================
#     if best_model_state is None:
#         print("❌ No model was trained successfully.")
#         return

#     print(f"\n🏆 Best hyperparameters: {best_params} | Best Val PCC: {best_score:.4f}")

#     # ✅ 直接从内存状态重建模型
#     final_model = CombinedModel(
#         [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#         [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#         num_heads=2,
#         embed_dim=532,
#         antigen_embed_dim=500
#     )
#     final_model.load_state_dict(best_model_state)
#     final_model.to(device)
#     final_model.eval()

#     print("\n" + "="*60)
#     print("🧪 FINAL TEST RESULTS (Independent Test Sets)")
#     print("="*60)

#     # 收集所有测试集样本用于后续的推理速度测试
#     all_test_samples = []
    
#     for name, split in all_splits.items():
#         test_samples = []
#         te = split["test"]
#         for i in range(len(te[3])):
#             test_samples.append((te[0][i], te[1][i], te[2][i], te[3][i]))
        
#         test_loader = DataLoader(
#             ListDataset(test_samples), 
#             batch_size=32, 
#             shuffle=False, 
#             collate_fn=collate_fn
#         )
        
#         metrics = evaluate(final_model, test_loader, device)
#         print(f"\n{name.upper()} TEST → R²: {metrics['R2']:.4f}, MSE: {metrics['MSE']:.4f}, RMSE: {metrics['RMSE']:.4f}, MAE: {metrics['MAE']:.4f}, PCC: {metrics['PCC']:.4f}")
        
#         all_test_samples.extend(test_samples)

#     # ======================================================
#     # STEP 4: Model Efficiency Analysis (For Reviewer)
#     # ======================================================
#     print("\n" + "="*60)
#     print("📊 MODEL EFFICIENCY ANALYSIS (For Reviewer)")
#     print("="*60)

#     # 1. Parameter Counts
#     total_params = sum(p.numel() for p in final_model.parameters())
#     trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
#     param_mb = (total_params * 4) / (1024 ** 2)  # Assuming float32 (4 bytes)
    
#     print(f"Total Parameters: {total_params:,} ({param_mb:.2f} MB)")
#     print(f"Trainable Parameters: {trainable_params:,}")

#     # 2. GPU Memory Consumption
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.reset_peak_memory_stats(device)
        
#         # Move model to device again to ensure it's counted
#         final_model.to(device)
        
#         # Baseline memory (Model weights only)
#         mem_baseline = torch.cuda.memory_allocated(device) / (1024 ** 2)
        
#         # Run a few forward passes to capture activation memory peak
#         # Use a subset of test data for measurement to avoid OOM on huge datasets
#         measure_loader = DataLoader(
#             ListDataset(all_test_samples[:100]), # Measure on first 100 samples
#             batch_size=32,
#             shuffle=False,
#             collate_fn=collate_fn
#         )
        
#         with torch.no_grad():
#             for X_a, X_b, ag, y in measure_loader:
#                 X_a, X_b, ag = X_a.to(device), X_b.to(device), ag.to(device)
#                 _ = final_model(X_b, X_a, ag)
        
#         mem_peak = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
#         mem_diff = mem_peak - mem_baseline
        
#         print(f"GPU Memory (Model Weights Only): {mem_baseline:.2f} MB")
#         print(f"GPU Memory (Peak with Activations): {mem_peak:.2f} MB")
#         print(f"GPU Memory (Overhead for Batch Inference): ~{mem_diff:.2f} MB")
#     else:
#         print("GPU Memory analysis skipped (CUDA not available).")

#     # 3. Inference Time
#     if len(all_test_samples) > 0:
#         # Warmup
#         dummy_sample = all_test_samples[0]
#         dummy_loader = DataLoader(ListDataset([dummy_sample]), batch_size=1, collate_fn=collate_fn)
#         with torch.no_grad():
#             X_a, X_b, ag, y = next(iter(dummy_loader))
#             X_a, X_b, ag = X_a.to(device), X_b.to(device), ag.to(device)
#             _ = final_model(X_b, X_a, ag)
        
#         # Timing
#         full_test_loader = DataLoader(
#             ListDataset(all_test_samples),
#             batch_size=1, # Measure per-sample time, or use batch_size=32 for throughput
#             shuffle=False,
#             collate_fn=collate_fn
#         )
        
#         start_time = time.time()
#         sample_count = 0
#         with torch.no_grad():
#             for X_a, X_b, ag, y in full_test_loader:
#                 X_a, X_b, ag = X_a.to(device), X_b.to(device), ag.to(device)
#                 _ = final_model(X_b, X_a, ag)
#                 sample_count += X_a.shape[0]
        
#         end_time = time.time()
#         total_time = end_time - start_time
#         avg_time_per_sample_ms = (total_time / sample_count) * 1000
#         samples_per_sec = sample_count / total_time
        
#         print(f"Inference Time (Total {sample_count} samples): {total_time:.4f} s")
#         print(f"Average Inference Time per Sample: {avg_time_per_sample_ms:.4f} ms")
#         print(f"Throughput: {samples_per_sec:.2f} samples/sec")
#     else:
#         print("Inference time analysis skipped (No test samples).")

#     print("\n✅ Training, Evaluation, and Efficiency Analysis Complete.")

# if __name__ == "__main__":
#     main()

# # ======================================================
# # 🔥 多 Seed + 固定划分 版本
# # ======================================================

# import torch
# import torch.nn as nn
# import numpy as np
# import random
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# from roformercnn import CombinedModel
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import itertools
# import copy
# import os

# # ======================================================
# # 固定数据划分 Seed（不随模型变化）
# # ======================================================
# SPLIT_SEED = 2024

# # 模型随机种子（多次独立训练）
# MODEL_SEEDS = [42, 2023, 3407, 0, 123]

# # ======================================================
# # 固定随机种子
# # ======================================================
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def getCDRPos(_loop, cdr_scheme='chothia'):
#     CDRS = {
#         'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
#         'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
#         'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
#         'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
#         'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
#         'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H',
#                '100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T',
#                '100U','100V','100W','100X','100Y','100Z','101','102']
#     }
#     return CDRS[_loop]

# # ======================================================
# # Dataset
# # ======================================================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         return self.samples[idx]

# # ======================================================
# # 修复后的 Collate
# # ======================================================
# def collate_fn(batch):
#     X_a = [torch.as_tensor(x[0], dtype=torch.float32) for x in batch]
#     X_b = [torch.as_tensor(x[1], dtype=torch.float32) for x in batch]
#     ag = [torch.as_tensor(x[2], dtype=torch.float32) for x in batch]
#     y = torch.as_tensor([x[3] for x in batch], dtype=torch.float32)

#     max_len = max(
#         max(x.shape[0] for x in X_a),
#         max(x.shape[0] for x in X_b),
#         max(x.shape[0] for x in ag)
#     )

#     def pad(x):
#         if x.shape[0] < max_len:
#             pad_tensor = torch.zeros(
#                 (max_len - x.shape[0], x.shape[1]),
#                 dtype=x.dtype,
#                 device=x.device  # 🔥 关键修复
#             )
#             return torch.cat([x, pad_tensor], dim=0)
#         return x[:max_len]

#     X_a = torch.stack([pad(x) for x in X_a])
#     X_b = torch.stack([pad(x) for x in X_b])
#     ag = torch.stack([pad(x) for x in ag])

#     return X_a, X_b, ag, y

# # ======================================================
# # Evaluate
# # ======================================================
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []

#     with torch.no_grad():
#         for X_a, X_b, ag, y in loader:
#             X_a, X_b, ag = X_a.to(device), X_b.to(device), ag.to(device)
#             pred = model(X_a, X_b, ag).view(-1)

#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())

#     mse = mean_squared_error(y_true, y_pred)
#     return {
#         "MSE": mse,
#         "RMSE": np.sqrt(mse),
#         "MAE": mean_absolute_error(y_true, y_pred),
#         "R2": r2_score(y_true, y_pred),
#         "PCC": pearsonr(y_true, y_pred)[0]
#     }

# # ======================================================
# # 固定划分
# # ======================================================
# def split_dataset(data):
#     X_a, X_b, ag, y = data
#     X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
#         X_a, X_b, ag, y, test_size=0.2, random_state=SPLIT_SEED
#     )
#     X_a_tr, X_a_val, X_b_tr, X_b_val, ag_tr, ag_val, y_tr, y_val = train_test_split(
#         X_a_tv, X_b_tv, ag_tv, y_tv,
#         test_size=0.25,
#         random_state=SPLIT_SEED
#     )
#     return (X_a_tr, X_b_tr, ag_tr, y_tr), \
#            (X_a_val, X_b_val, ag_val, y_val), \
#            (X_a_test, X_b_test, ag_test, y_test)

# # ======================================================
# # 单 seed 训练
# # ======================================================
# def train_one_seed(seed, train_loader, val_loader, device):

#     print("\n==============================")
#     print(f"🔥 Training with MODEL_SEED = {seed}")
#     print("==============================")

#     set_seed(seed)

#     model = CombinedModel(
#         cdr_boundaries_light=[getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#         cdr_boundaries_heavy=[getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#         num_heads=2,
#         embed_dim=532,
#         antigen_embed_dim=500
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
#     criterion = nn.MSELoss()

#     best_state = None
#     best_mse = np.inf
#     patience = 15
#     wait = 0

#     for epoch in range(50):
#         model.train()

#         for X_a, X_b, ag, y in train_loader:
#             X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)

#             optimizer.zero_grad()
#             pred = model(X_a, X_b, ag).view(-1)
#             loss = criterion(pred, y)
#             loss.backward()
#             optimizer.step()

#         val_metrics = evaluate(model, val_loader, device)
#         scheduler.step(val_metrics["MSE"])

#         print(f"Epoch {epoch+1:02d} | Val PCC: {val_metrics['PCC']:.4f}")

#         if val_metrics["MSE"] < best_mse:
#             best_mse = val_metrics["MSE"]
#             best_state = copy.deepcopy(model.state_dict())
#             wait = 0
#         else:
#             wait += 1
#             if wait >= patience:
#                 print("Early stopping")
#                 break

#     model.load_state_dict(best_state)
#     return model

# # ======================================================
# # MAIN
# # ======================================================
# def main():

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print("📌 SPLIT_SEED =", SPLIT_SEED)
#     print("📌 MODEL_SEEDS =", MODEL_SEEDS)

#     d = torch.load("/tmp/AbAgCDR/data/skempi_data.pt", map_location="cpu")
#     data = (d["X_a"], d["X_b"], d["antigen"], d["y"])

#     train_data, val_data, test_data = split_dataset(data)

#     train_samples = list(zip(*train_data))
#     val_samples = list(zip(*val_data))
#     test_samples = list(zip(*test_data))

#     train_loader = DataLoader(ListDataset(train_samples), batch_size=32, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(ListDataset(val_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(ListDataset(test_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)

#     all_results = []

#     for seed in MODEL_SEEDS:
#         model = train_one_seed(seed, train_loader, val_loader, device)
#         test_metrics = evaluate(model, test_loader, device)

#         print(f"\n✅ Seed {seed} Test PCC: {test_metrics['PCC']:.4f}")
#         all_results.append(test_metrics["MAE"])

#     print("\n==============================")
#     print("🎯 Multi-seed Results (MAE)")
#     print("==============================")
#     print(all_results)
#     print("Mean:", np.mean(all_results))
#     print("Std :", np.std(all_results))

# if __name__ == "__main__":
#     main()

# # ======================================================
# # Dataset 类
# # ======================================================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         return self.samples[idx]


# # ======================================================
# # Collate
# # ======================================================
# def collate_fn(batch):
#     X_a_list = [torch.tensor(item[0], dtype=torch.float32) for item in batch]
#     X_b_list = [torch.tensor(item[1], dtype=torch.float32) for item in batch]
#     ag_list = [torch.tensor(item[2], dtype=torch.float32) for item in batch]
#     y_list = [torch.tensor(item[3], dtype=torch.float32) for item in batch]

#     max_len = max(
#         max(x.shape[0] for x in X_a_list),
#         max(x.shape[0] for x in X_b_list),
#         max(x.shape[0] for x in ag_list)
#     )

#     def pad(x):
#         if x.shape[0] < max_len:
#             pad = torch.zeros(max_len - x.shape[0], x.shape[1])
#             return torch.cat([x, pad], dim=0)
#         return x[:max_len]

#     return (
#         torch.stack([pad(x) for x in X_a_list]),
#         torch.stack([pad(x) for x in X_b_list]),
#         torch.stack([pad(x) for x in ag_list]),
#         torch.stack(y_list)
#     )


# # ======================================================
# # Evaluate
# # ======================================================
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for X_a, X_b, ag, y in loader:
#             X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
#             pred = model(X_b, X_a, ag).view(-1)
#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     pcc = pearsonr(y_true, y_pred)[0] if len(set(y_true)) > 1 else 0.0

#     return {"RMSE": rmse, "MSE": mse, "MAE": mae, "R2": r2, "PCC": pcc}


# # ======================================================
# # 数据加载
# # ======================================================
# def load_dataset(path):
#     d = torch.load(path)
#     return {
#         "X_a": d["X_a"].cpu().numpy(),
#         "X_b": d["X_b"].cpu().numpy(),
#         "antigen": d["antigen"].cpu().numpy(),
#         "y": d["y"].cpu().numpy()
#     }


# # ======================================================
# # 固定划分（只执行一次）
# # ======================================================
# def split_dataset(data):
#     X_a, X_b, ag, y = data.values()
#     X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
#         X_a, X_b, ag, y, test_size=0.2, random_state=SPLIT_SEED
#     )
#     X_a_tr, X_a_val, X_b_tr, X_b_val, ag_tr, ag_val, y_tr, y_val = train_test_split(
#         X_a_tv, X_b_tv, ag_tv, y_tv,
#         test_size=0.25,  # 0.25 * 0.8 = 0.2
#         random_state=SPLIT_SEED
#     )
#     return {
#         "train": (X_a_tr, X_b_tr, ag_tr, y_tr),
#         "val": (X_a_val, X_b_val, ag_val, y_val),
#         "test": (X_a_test, X_b_test, ag_test, y_test)
#     }


# # ======================================================
# # 主程序
# # ======================================================
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     paths = {
#         "train": "/tmp/AbAgCDR/data/train_data.pt",
#         "abbind": "/tmp/AbAgCDR/data/abbind_data.pt",
#         "sabdab": "/tmp/AbAgCDR/data/sabdab_data.pt",
#         "skempi": "/tmp/AbAgCDR/data/skempi_data.pt"
#     }
#     benchmark_path = "/tmp/AbAgCDR/data/benchmark_data.pt"

#     # ================================
#     # 1️⃣ 固定划分（只做一次）
#     # ================================
#     all_splits = {}
#     for name, path in paths.items():
#         data = load_dataset(path)
#         all_splits[name] = split_dataset(data)

#     benchmark_data = load_dataset(benchmark_path)

#     # 记录结果
#     all_results = {name: [] for name in list(paths.keys()) + ["benchmark"]}

#     # ================================
#     # 2️⃣ 多 Seed 训练
#     # ================================
#     for seed in MODEL_SEEDS:
#         print(f"\n{'='*70}")
#         print(f"🔥 Training with MODEL SEED = {seed}")
#         print('='*70)

#         set_seed(seed)

#         # 构建 train/val loader
#         train_samples, val_samples = [], []

#         for split in all_splits.values():
#             tr = split["train"]
#             va = split["val"]

#             for i in range(len(tr[3])):
#                 train_samples.append((tr[0][i], tr[1][i], tr[2][i], tr[3][i]))

#             for i in range(len(va[3])):
#                 val_samples.append((va[0][i], va[1][i], va[2][i], va[3][i]))

#         train_loader = DataLoader(ListDataset(train_samples),
#                                   batch_size=32, shuffle=True,
#                                   collate_fn=collate_fn)

#         val_loader = DataLoader(ListDataset(val_samples),
#                                 batch_size=32, shuffle=False,
#                                 collate_fn=collate_fn)

#         model = CombinedModel(
#         [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#         [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#         num_heads=2,
#         embed_dim=532,
#         antigen_embed_dim=500
#     )
#         model.to(device)

#         optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#         criterion = nn.MSELoss()

#         best_val = np.inf
#         best_state = None

#         for epoch in range(50):
#             model.train()
#             for X_a, X_b, ag, y in train_loader:
#                 X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
#                 optimizer.zero_grad()
#                 pred = model(X_b, X_a, ag).view(-1)
#                 loss = criterion(pred, y)
#                 loss.backward()
#                 optimizer.step()

#             val_metrics = evaluate(model, val_loader, device)
#             if val_metrics["RMSE"] < best_val:
#                 best_val = val_metrics["RMSE"]
#                 best_state = copy.deepcopy(model.state_dict())

#         model.load_state_dict(best_state)

#         # ================================
#         # 3️⃣ 各 test 集评估
#         # ================================
#         for name, split in all_splits.items():
#             te = split["test"]
#             test_samples = [(te[0][i], te[1][i], te[2][i], te[3][i])
#                             for i in range(len(te[3]))]
#             test_loader = DataLoader(ListDataset(test_samples),
#                                      batch_size=32, shuffle=False,
#                                      collate_fn=collate_fn)
#             metrics = evaluate(model, test_loader, device)
#             all_results[name].append(metrics)

#         # benchmark
#         bench_samples = [(benchmark_data["X_a"][i],
#                           benchmark_data["X_b"][i],
#                           benchmark_data["antigen"][i],
#                           benchmark_data["y"][i])
#                          for i in range(len(benchmark_data["y"]))]

#         bench_loader = DataLoader(ListDataset(bench_samples),
#                                   batch_size=32, shuffle=False,
#                                   collate_fn=collate_fn)

#         bench_metrics = evaluate(model, bench_loader, device)
#         all_results["benchmark"].append(bench_metrics)

#     # ================================
#     # 4️⃣ 输出 mean ± std
#     # ================================
#     print("\n\n" + "="*70)
#     print("📊 FINAL MULTI-SEED RESULTS (mean ± std)")
#     print("="*70)

#     for name, results in all_results.items():
#         rmse = [r["RMSE"] for r in results]
#         mae = [r["MAE"] for r in results]
#         pcc = [r["PCC"] for r in results]

#         print(f"\n{name.upper()}")
#         print(f"RMSE: {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")
#         print(f"MAE : {np.mean(mae):.4f} ± {np.std(mae):.4f}")
#         print(f"PCC : {np.mean(pcc):.4f} ± {np.std(pcc):.4f}")


# if __name__ == "__main__":
#     main()

# # PWAA+RPE模型训练
# # -*- coding: utf-8 -*-
# """
# 训练脚本 - 适配 CombinedModel 架构
# 模型输入顺序：(antibody_light, antibody_heavy, antigen)
# """

# import torch
# import torch.nn as nn
# import numpy as np
# import random
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# from roformercnn import CombinedModel
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import itertools
# import copy
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # ======================================================
# # 固定随机种子
# # ======================================================
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

# # ======================================================
# # CDR 区域定义（Chothia）
# # ======================================================
# def getCDRPos(_loop, cdr_scheme='chothia'):
#     CDRS = {
#         'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
#         'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
#         'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
#         'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
#         'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
#         'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H',
#                '100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T',
#                '100U','100V','100W','100X','100Y','100Z','101','102']
#     }
#     return CDRS[_loop]

# # ======================================================
# # Dataset
# # ======================================================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]

# # ======================================================
# # Collate function - ✅ 关键修复：确保维度正确
# # ======================================================
# def collate_fn(batch):
#     """
#     batch: list of (light, heavy, antigen, label)
#     返回：(light, heavy, antigen, label) 均为 [batch, seq_len, dim]
#     """
#     # 提取并转换
#     X_a_list = [torch.tensor(item[0], dtype=torch.float32) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch]
#     X_b_list = [torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch]
#     ag_list = [torch.tensor(item[2], dtype=torch.float32) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch]
#     y_list = [torch.tensor(item[3], dtype=torch.float32) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch]

#     # 获取最大序列长度
#     max_len = max(
#         max(x.shape[0] for x in X_a_list),
#         max(x.shape[0] for x in X_b_list),
#         max(x.shape[0] for x in ag_list)
#     )

#     def pad_to_len(x, L):
#         """填充或截断到指定长度"""
#         if x.shape[0] < L:
#             pad = torch.zeros(L - x.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device)
#             return torch.cat([x, pad], dim=0)
#         else:
#             return x[:L]

#     # 填充
#     X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
#     X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
#     ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])
#     y_tensor = torch.stack(y_list)

#     # ✅ 确保是 3D tensor: [batch, seq_len, dim]
#     if X_a_padded.dim() == 2:
#         X_a_padded = X_a_padded.unsqueeze(0)
#     if X_b_padded.dim() == 2:
#         X_b_padded = X_b_padded.unsqueeze(0)
#     if ag_padded.dim() == 2:
#         ag_padded = ag_padded.unsqueeze(0)

#     return X_a_padded, X_b_padded, ag_padded, y_tensor

# # ======================================================
# # Evaluation - ✅ 关键修复：输入顺序 (light, heavy, antigen)
# # ======================================================
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []

#     with torch.no_grad():
#         for X_a, X_b, ag, y in loader:
#             X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
#             # ✅ 修复：模型输入顺序是 (antibody_light, antibody_heavy, antigen)
#             pred = model(X_a, X_b, ag).view(-1)
#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     pcc = pearsonr(y_true, y_pred)[0] if len(set(y_true)) > 1 else 0.0

#     return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc}

# # ======================================================
# # Load dataset
# # ======================================================
# def load_dataset(path):
#     d = torch.load(path, map_location='cpu')
#     return {
#         "X_a": d["X_a"].cpu().numpy(),
#         "X_b": d["X_b"].cpu().numpy(),
#         "antigen": d["antigen"].cpu().numpy(),
#         "y": d["y"].cpu().numpy()
#     }

# # ======================================================
# # Split dataset
# # ======================================================
# def split_dataset(data, test_size=0.2, val_size=0.2):
#     X_a, X_b, ag, y = data.values()
#     X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
#         X_a, X_b, ag, y, test_size=test_size, random_state=42
#     )
#     val_ratio = val_size / (1 - test_size)
#     X_a_tr, X_a_val, X_b_tr, X_b_val, ag_tr, ag_val, y_tr, y_val = train_test_split(
#         X_a_tv, X_b_tv, ag_tv, y_tv, test_size=val_ratio, random_state=42
#     )
#     return {
#         "train": (X_a_tr, X_b_tr, ag_tr, y_tr),
#         "val": (X_a_val, X_b_val, ag_val, y_val),
#         "test": (X_a_test, X_b_test, ag_test, y_test)
#     }

# # ======================================================
# # Trainer - ✅ 关键修复：输入顺序 (light, heavy, antigen)
# # ======================================================
# class TrainerWithScheduler:
#     def __init__(self, model, train_loader, val_loader, params, device):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = device
#         self.opt = optim.Adam(
#             model.parameters(),
#             lr=params["lr"],
#             weight_decay=params["weight_decay"]
#         )
#         self.scheduler = ReduceLROnPlateau(
#             self.opt,
#             mode='min',
#             factor=params.get("lr_factor", 0.5),
#             patience=params.get("scheduler_patience", 3),
#             min_lr=params.get("min_lr", 1e-6),
#             verbose=False
#         )
#         self.criterion = nn.MSELoss()
#         self.epochs = params["epochs"]
#         self.patience = params["patience"]

#     def train(self):
#         best_mse = np.inf
#         best_state = None
#         wait = 0

#         for epoch in range(1, self.epochs + 1):
#             self.model.train()
#             total_loss = 0.0
#             num_batches = 0

#             for X_a, X_b, ag, y in self.train_loader:
#                 X_a, X_b, ag, y = X_a.to(self.device), X_b.to(self.device), ag.to(self.device), y.to(self.device)
#                 self.opt.zero_grad()
                
#                 # ✅ 修复：模型输入顺序是 (antibody_light, antibody_heavy, antigen)
#                 pred = self.model(X_a, X_b, ag).view(-1)
                
#                 loss = self.criterion(pred, y)
#                 loss.backward()
#                 self.opt.step()

#                 total_loss += loss.item()
#                 num_batches += 1

#             avg_train_loss = total_loss / num_batches
#             val_metrics = evaluate(self.model, self.val_loader, self.device)
#             val_mse = val_metrics["MSE"]

#             self.scheduler.step(val_mse)

#             print(f"Epoch {epoch:02d}/{self.epochs} | "
#                   f"Train Loss: {avg_train_loss:.6f} | "
#                   f"Val MSE: {val_mse:.4f} | "
#                   f"Val R²: {val_metrics['R2']:.4f} | "
#                   f"Val PCC: {val_metrics['PCC']:.4f}")

#             if val_mse < best_mse:
#                 best_mse = val_mse
#                 best_state = copy.deepcopy(self.model.state_dict())
#                 wait = 0
#             else:
#                 wait += 1
#                 if wait >= self.patience:
#                     print("🛑 Early stopping triggered.")
#                     break

#         self.model.load_state_dict(best_state)
#         return self.model

# # ======================================================
# # MAIN
# # ======================================================
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🖥️  Using device: {device}")

#     paths = {
#         "train": "/tmp/AbAgCDR/data/train_data.pt",
#         "abbind": "/tmp/AbAgCDR/data/abbind_data.pt",
#         "sabdab": "/tmp/AbAgCDR/data/sabdab_data.pt",
#         "skempi": "/tmp/AbAgCDR/data/skempi_data.pt"
#     }
#     benchmark_path = "/tmp/AbAgCDR/data/benchmark_data.pt"

#     # ======================================================
#     # Load and split training datasets
#     # ======================================================
#     print("\n" + "="*70)
#     print("📊 加载并划分数据集")
#     print("="*70)
    
#     all_splits = {}
#     for name, path in paths.items():
#         if not os.path.exists(path):
#             print(f"⚠️  跳过 {name}: 文件不存在")
#             continue
#         print(f"加载 {name}...")
#         data = load_dataset(path)
#         all_splits[name] = split_dataset(data)
#         print(f"   train={len(data['y'])}, val={len(all_splits[name]['val'][3])}, test={len(all_splits[name]['test'][3])}")

#     # ======================================================
#     # Load benchmark
#     # ======================================================
#     print("\n" + "="*70)
#     print("📊 加载 Benchmark")
#     print("="*70)
    
#     if os.path.exists(benchmark_path):
#         benchmark_data = load_dataset(benchmark_path)
#         benchmark_samples = []
#         for i in range(len(benchmark_data["y"])):
#             benchmark_samples.append((
#                 benchmark_data["X_a"][i],
#                 benchmark_data["X_b"][i],
#                 benchmark_data["antigen"][i],
#                 benchmark_data["y"][i]
#             ))
#         benchmark_loader = DataLoader(
#             ListDataset(benchmark_samples),
#             batch_size=32,
#             shuffle=False,
#             collate_fn=collate_fn
#         )
#         print(f"✅ Benchmark 样本数：{len(benchmark_samples)}")
#     else:
#         print(f"⚠️  Benchmark 文件不存在：{benchmark_path}")
#         benchmark_loader = None

#     # ======================================================
#     # Collect samples with dataset weights
#     # ======================================================
#     print("\n" + "="*70)
#     print("📦 准备训练数据")
#     print("="*70)
    
#     all_train_samples = []
#     sample_weights = []
#     dataset_weights = {
#         'train': 0.7,
#         'abbind': 0.1,
#         'sabdab': 0.1,
#         'skempi': 0.1
#     }

#     for name in paths.keys():
#         if name not in all_splits:
#             continue
#         tr = all_splits[name]["train"]
#         w = dataset_weights.get(name, 0.1)
#         for i in range(len(tr[3])):
#             # ✅ 样本格式：(light, heavy, antigen, label)
#             all_train_samples.append((tr[0][i], tr[1][i], tr[2][i], tr[3][i]))
#             sample_weights.append(w)

#     # Validation samples
#     val_samples = []
#     for split in all_splits.values():
#         va = split["val"]
#         for i in range(len(va[3])):
#             val_samples.append((va[0][i], va[1][i], va[2][i], va[3][i]))

#     print(f"✅ 总训练样本：{len(all_train_samples)}")
#     print(f"✅ 总验证样本：{len(val_samples)}")

#     # Create weighted sampler
#     sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(sample_weights),
#         replacement=True
#     )

#     # ======================================================
#     # Hyperparameter Grid Search
#     # ======================================================
#     print("\n" + "="*70)
#     print("🔍 超参数网格搜索")
#     print("="*70)
    
#     param_grid = {
#         'lr': [1e-4, 5e-4],
#         'batch_size': [16, 32],
#         'epochs': [50],
#         'patience': [15],
#         'weight_decay': [1e-5],
#         'scheduler_patience': [3],
#         'lr_factor': [0.5],
#         'min_lr': [1e-6]
#     }

#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     print(f"🚀 共 {len(param_combinations)} 组超参数组合")

#     best_score = -np.inf
#     best_params = None
#     best_model_state = None
    
#     # Load label scaler if exists
#     try:
#         train_data_for_scaler = torch.load(paths["train"], map_location="cpu")
#         label_scaler = train_data_for_scaler.get("label_scaler", None)
#     except:
#         label_scaler = None

#     original_train_samples = all_train_samples
#     original_val_samples = val_samples

#     save_dir = "/tmp/AbAgCDR/model"
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "PWAARPEbest_model.pth")

#     for trial_idx, params in enumerate(param_combinations):
#         print(f"\n{'='*60}")
#         print(f"Trial {trial_idx+1}/{len(param_combinations)}")
#         print(f"{'='*60}")
#         print(f"Params: {params}")

#         # Rebuild loaders with current batch_size
#         current_bs = params['batch_size']
#         train_loader = DataLoader(
#             ListDataset(original_train_samples),
#             batch_size=current_bs,
#             sampler=sampler,
#             collate_fn=collate_fn,
#             shuffle=False,
#             num_workers=0
#         )
#         val_loader = DataLoader(
#             ListDataset(original_val_samples),
#             batch_size=current_bs,
#             shuffle=False,
#             collate_fn=collate_fn,
#             num_workers=0
#         )

#         # ✅ 修复：CDR 边界正确传递给模型
#         # 轻链 CDR: L1, L2, L3
#         # 重链 CDR: H1, H2, H3
#         model = CombinedModel(
#             cdr_boundaries_light=[getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#             cdr_boundaries_heavy=[getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#             num_heads=2,
#             embed_dim=532,
#             antigen_embed_dim=500,
#             hidden_dim=256
#         )

#         # Count parameters
#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"   总参数量：{total_params:,}")
#         print(f"   可训练参数：{trainable_params:,}")

#         # Train
#         trainer = TrainerWithScheduler(model, train_loader, val_loader, params, device)
#         trained_model = trainer.train()

#         # Evaluate on validation set
#         val_metrics = evaluate(trained_model, val_loader, device)
#         score = val_metrics['PCC']

#         print(f"\n✅ Val PCC: {score:.4f}")

#         if score > best_score:
#             best_score = score
#             best_params = copy.deepcopy(params)
#             best_model_state = copy.deepcopy(trained_model.state_dict())
            
#             # Save the best model
#             torch.save({
#                 'model_state_dict': best_model_state,
#                 'params': best_params,
#                 'label_scaler': label_scaler,
#                 'config': {
#                     'embed_dim': 532,
#                     'antigen_embed_dim': 500,
#                     'hidden_dim': 256,
#                     'num_heads': 2
#                 }
#             }, save_path)
#             print(f"🎉 新最佳模型已保存：{save_path}")

#     # ======================================================
#     # Final evaluation with best model
#     # ======================================================
#     print("\n" + "="*70)
#     print("🏆 最佳模型测试")
#     print("="*70)
#     print(f"最佳超参数：{best_params}")
#     print(f"最佳 Val PCC: {best_score:.4f}")

#     # Reinitialize final model
#     final_model = CombinedModel(
#         cdr_boundaries_light=[getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#         cdr_boundaries_heavy=[getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#         num_heads=2,
#         embed_dim=532,
#         antigen_embed_dim=500,
#         hidden_dim=256
#     )
#     final_model.load_state_dict(best_model_state)
#     final_model.to(device)
#     final_model.eval()

#     print("\n" + "="*70)
#     print("🧪 各数据集测试结果")
#     print("="*70)

#     results = {}
#     for name, split in all_splits.items():
#         test_samples = []
#         te = split["test"]
#         for i in range(len(te[3])):
#             test_samples.append((te[0][i], te[1][i], te[2][i], te[3][i]))
#         test_loader = DataLoader(
#             ListDataset(test_samples), 
#             batch_size=32, 
#             shuffle=False, 
#             collate_fn=collate_fn,
#             num_workers=0
#         )
#         metrics = evaluate(final_model, test_loader, device)
#         results[name] = metrics
#         print(f"\n{name.upper()} TEST → "
#               f"R²: {metrics['R2']:.4f}, "
#               f"MSE: {metrics['MSE']:.4f}, "
#               f"RMSE: {metrics['RMSE']:.4f}, "
#               f"MAE: {metrics['MAE']:.4f}, "
#               f"PCC: {metrics['PCC']:.4f}")

#     if benchmark_loader is not None:
#         print("\n" + "="*70)
#         print("🎯 Benchmark 测试结果")
#         print("="*70)
#         bench_metrics = evaluate(final_model, benchmark_loader, device)
#         results['benchmark'] = bench_metrics
#         print(f"\n🎯 BENCHMARK TEST → "
#               f"R²: {bench_metrics['R2']:.4f}, "
#               f"MSE: {bench_metrics['MSE']:.4f}, "
#               f"RMSE: {bench_metrics['RMSE']:.4f}, "
#               f"MAE: {bench_metrics['MAE']:.4f}, "
#               f"PCC: {bench_metrics['PCC']:.4f}")

#     # ======================================================
#     # Save results
#     # ======================================================
#     print("\n" + "="*70)
#     print("💾 保存结果")
#     print("="*70)
    
#     results_path = os.path.join(save_dir, "results.txt")
#     with open(results_path, 'w', encoding='utf-8') as f:
#         f.write("="*70 + "\n")
#         f.write("CombinedModel 训练结果\n")
#         f.write("="*70 + "\n\n")
#         f.write(f"最佳超参数：{best_params}\n")
#         f.write(f"最佳 Val PCC: {best_score:.4f}\n\n")
#         f.write("-"*70 + "\n")
#         f.write("测试结果\n")
#         f.write("-"*70 + "\n")
#         for name, metrics in results.items():
#             f.write(f"\n{name.upper()}:\n")
#             f.write(f"  R²:   {metrics['R2']:.4f}\n")
#             f.write(f"  MSE:  {metrics['MSE']:.4f}\n")
#             f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
#             f.write(f"  MAE:  {metrics['MAE']:.4f}\n")
#             f.write(f"  PCC:  {metrics['PCC']:.4f}\n")
    
#     print(f"✅ 结果已保存：{results_path}")
#     print(f"✅ 模型已保存：{save_path}")
    
#     print("\n" + "="*70)
#     print("✅ 训练完成！")
#     print("="*70)

# if __name__ == "__main__":
#     main()

# import torch
# import torch.nn as nn
# import numpy as np
# import random
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# from roformercnn import CombinedModel
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import copy
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # ======================================================
# # ✅ 固定数据划分种子
# # ======================================================
# SPLIT_SEED = 2024

# # ======================================================
# # ✅ 模型随机种子（多次独立训练）
# # ======================================================
# MODEL_SEEDS = [42, 2023, 3407, 0, 123]

# # ======================================================
# # 设置随机种子
# # ======================================================
# def set_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # ======================================================
# # CDR 区域定义（Chothia）
# # ======================================================
# def getCDRPos(_loop, cdr_scheme='chothia'):
#     CDRS = {
#         'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
#         'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
#         'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
#         'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
#         'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
#         'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H',
#                '100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T',
#                '100U','100V','100W','100X','100Y','100Z','101','102']
#     }
#     return CDRS[_loop]
# # ======================================================
# # Dataset
# # ======================================================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         return self.samples[idx]

# # ======================================================
# # 修复后的 Collate
# # ======================================================
# def collate_fn(batch):
#     X_a = [torch.as_tensor(x[0], dtype=torch.float32) for x in batch]
#     X_b = [torch.as_tensor(x[1], dtype=torch.float32) for x in batch]
#     ag = [torch.as_tensor(x[2], dtype=torch.float32) for x in batch]
#     y = torch.as_tensor([x[3] for x in batch], dtype=torch.float32)

#     max_len = max(
#         max(x.shape[0] for x in X_a),
#         max(x.shape[0] for x in X_b),
#         max(x.shape[0] for x in ag)
#     )

#     def pad(x):
#         if x.shape[0] < max_len:
#             pad_tensor = torch.zeros(
#                 (max_len - x.shape[0], x.shape[1]),
#                 dtype=x.dtype,
#                 device=x.device  # 🔥 关键修复
#             )
#             return torch.cat([x, pad_tensor], dim=0)
#         return x[:max_len]

#     X_a = torch.stack([pad(x) for x in X_a])
#     X_b = torch.stack([pad(x) for x in X_b])
#     ag = torch.stack([pad(x) for x in ag])

#     return X_a, X_b, ag, y

# # ======================================================
# # Evaluate
# # ======================================================
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []

#     with torch.no_grad():
#         for X_a, X_b, ag, y in loader:
#             X_a, X_b, ag = X_a.to(device), X_b.to(device), ag.to(device)
#             pred = model(X_a, X_b, ag).view(-1)

#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())

#     mse = mean_squared_error(y_true, y_pred)
#     return {
#         "MSE": mse,
#         "RMSE": np.sqrt(mse),
#         "MAE": mean_absolute_error(y_true, y_pred),
#         "R2": r2_score(y_true, y_pred),
#         "PCC": pearsonr(y_true, y_pred)[0]
#     }

# # ======================================================
# # 固定划分
# # ======================================================
# def split_dataset(data):
#     X_a, X_b, ag, y = data
#     X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
#         X_a, X_b, ag, y, test_size=0.2, random_state=SPLIT_SEED
#     )
#     X_a_tr, X_a_val, X_b_tr, X_b_val, ag_tr, ag_val, y_tr, y_val = train_test_split(
#         X_a_tv, X_b_tv, ag_tv, y_tv,
#         test_size=0.25,
#         random_state=SPLIT_SEED
#     )
#     return (X_a_tr, X_b_tr, ag_tr, y_tr), \
#            (X_a_val, X_b_val, ag_val, y_val), \
#            (X_a_test, X_b_test, ag_test, y_test)

# # ======================================================
# # 单 seed 训练
# # ======================================================
# def train_one_seed(seed, train_loader, val_loader, device):

#     print("\n==============================")
#     print(f"🔥 Training with MODEL_SEED = {seed}")
#     print("==============================")

#     set_seed(seed)

#     model = CombinedModel(
#         cdr_boundaries_light=[getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#         cdr_boundaries_heavy=[getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#         num_heads=2,
#         embed_dim=532,
#         antigen_embed_dim=500,
#         hidden_dim=256
#     ).to(device)

#     optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#     scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3)
#     criterion = nn.MSELoss()

#     best_state = None
#     best_mse = np.inf
#     patience = 15
#     wait = 0

#     for epoch in range(50):
#         model.train()

#         for X_a, X_b, ag, y in train_loader:
#             X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)

#             optimizer.zero_grad()
#             pred = model(X_a, X_b, ag).view(-1)
#             loss = criterion(pred, y)
#             loss.backward()
#             optimizer.step()

#         val_metrics = evaluate(model, val_loader, device)
#         scheduler.step(val_metrics["MSE"])

#         print(f"Epoch {epoch+1:02d} | Val PCC: {val_metrics['PCC']:.4f}")

#         if val_metrics["MSE"] < best_mse:
#             best_mse = val_metrics["MSE"]
#             best_state = copy.deepcopy(model.state_dict())
#             wait = 0
#         else:
#             wait += 1
#             if wait >= patience:
#                 print("Early stopping")
#                 break

#     model.load_state_dict(best_state)
#     return model

# # ======================================================
# # MAIN
# # ======================================================
# def main():

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     print("📌 SPLIT_SEED =", SPLIT_SEED)
#     print("📌 MODEL_SEEDS =", MODEL_SEEDS)

#     d = torch.load("/tmp/AbAgCDR/data/train_data.pt", map_location="cpu")
#     data = (d["X_a"], d["X_b"], d["antigen"], d["y"])

#     train_data, val_data, test_data = split_dataset(data)

#     train_samples = list(zip(*train_data))
#     val_samples = list(zip(*val_data))
#     test_samples = list(zip(*test_data))

#     train_loader = DataLoader(ListDataset(train_samples), batch_size=32, shuffle=True, collate_fn=collate_fn)
#     val_loader = DataLoader(ListDataset(val_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)
#     test_loader = DataLoader(ListDataset(test_samples), batch_size=32, shuffle=False, collate_fn=collate_fn)

#     all_results = []

#     for seed in MODEL_SEEDS:
#         model = train_one_seed(seed, train_loader, val_loader, device)
#         test_metrics = evaluate(model, test_loader, device)

#         print(f"\n✅ Seed {seed} Test PCC: {test_metrics['PCC']:.4f}")
#         all_results.append(test_metrics["MAE"])

#     print("\n==============================")
#     print("🎯 Multi-seed Results (MAE)")
#     print("==============================")
#     print(all_results)
#     print("Mean:", np.mean(all_results))
#     print("Std :", np.std(all_results))

# if __name__ == "__main__":
#     main()

# # ======================================================
# # Dataset
# # ======================================================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples
#     def __len__(self):
#         return len(self.samples)
#     def __getitem__(self, idx):
#         return self.samples[idx]

# # ======================================================
# # Collate
# # ======================================================
# def collate_fn(batch):
#     X_a = [torch.as_tensor(x[0], dtype=torch.float32) for x in batch]
#     X_b = [torch.as_tensor(x[1], dtype=torch.float32) for x in batch]
#     ag  = [torch.as_tensor(x[2], dtype=torch.float32) for x in batch]
#     y   = torch.as_tensor([x[3] for x in batch], dtype=torch.float32)

#     max_len = max(
#         max(x.shape[0] for x in X_a),
#         max(x.shape[0] for x in X_b),
#         max(x.shape[0] for x in ag)
#     )

#     def pad(x):
#         if x.shape[0] < max_len:
#             pad_tensor = torch.zeros(
#                 (max_len - x.shape[0], x.shape[1]),
#                 dtype=x.dtype,
#                 device=x.device
#             )
#             return torch.cat([x, pad_tensor], dim=0)
#         return x[:max_len]

#     return (
#         torch.stack([pad(x) for x in X_a]),
#         torch.stack([pad(x) for x in X_b]),
#         torch.stack([pad(x) for x in ag]),
#         y
#     )

# # ======================================================
# # Evaluate
# # ======================================================
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for X_a, X_b, ag, y in loader:
#             X_a, X_b, ag = X_a.to(device), X_b.to(device), ag.to(device)
#             pred = model(X_b, X_a, ag).view(-1)
#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())

#     mse = mean_squared_error(y_true, y_pred)
#     return {
#         "RMSE": np.sqrt(mse),
#         "MSE": mse,
#         "MAE": mean_absolute_error(y_true, y_pred),
#         "R2": r2_score(y_true, y_pred),
#         "PCC": pearsonr(y_true, y_pred)[0] if len(set(y_true)) > 1 else 0.0
#     }

# # ======================================================
# # Load dataset
# # ======================================================
# def load_dataset(path):
#     d = torch.load(path, map_location="cpu")
#     return {
#         "X_a": d["X_a"].cpu().numpy(),
#         "X_b": d["X_b"].cpu().numpy(),
#         "antigen": d["antigen"].cpu().numpy(),
#         "y": d["y"].cpu().numpy()
#     }

# # ======================================================
# # 固定划分 6:2:2
# # ======================================================
# def split_dataset(data):
#     X_a, X_b, ag, y = data.values()

#     X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
#         X_a, X_b, ag, y,
#         test_size=0.2,
#         random_state=SPLIT_SEED
#     )

#     X_a_tr, X_a_val, X_b_tr, X_b_val, ag_tr, ag_val, y_tr, y_val = train_test_split(
#         X_a_tv, X_b_tv, ag_tv, y_tv,
#         test_size=0.25,   # 0.25 * 0.8 = 0.2
#         random_state=SPLIT_SEED
#     )

#     return {
#         "train": (X_a_tr, X_b_tr, ag_tr, y_tr),
#         "val":   (X_a_val, X_b_val, ag_val, y_val),
#         "test":  (X_a_test, X_b_test, ag_test, y_test)
#     }

# # ======================================================
# # 主程序
# # ======================================================
# def main():

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     paths = {
#         "train":  "/tmp/AbAgCDR/data/train_data.pt",
#         "abbind": "/tmp/AbAgCDR/data/abbind_data.pt",
#         "sabdab": "/tmp/AbAgCDR/data/sabdab_data.pt",
#         "skempi": "/tmp/AbAgCDR/data/skempi_data.pt"
#     }

#     benchmark_path = "/tmp/AbAgCDR/data/benchmark_data.pt"

#     # ================================
#     # 1️⃣ 固定划分
#     # ================================
#     all_splits = {}
#     for name, path in paths.items():
#         all_splits[name] = split_dataset(load_dataset(path))

#     benchmark_data = load_dataset(benchmark_path)

#     # 采样权重
#     dataset_weights = {
#         "train": 0.7,
#         "abbind": 0.1,
#         "sabdab": 0.1,
#         "skempi": 0.1
#     }

#     all_results = {name: [] for name in list(paths.keys()) + ["benchmark"]}

#     # ================================
#     # 2️⃣ 多 Seed 训练
#     # ================================
#     for seed in MODEL_SEEDS:

#         print("\n" + "="*70)
#         print(f"🔥 Training with MODEL SEED = {seed}")
#         print("="*70)

#         set_seed(seed)

#         train_samples = []
#         val_samples   = []
#         sample_weights = []

#         for name, split in all_splits.items():

#             tr = split["train"]
#             va = split["val"]

#             weight = dataset_weights[name]

#             for i in range(len(tr[3])):
#                 train_samples.append((tr[0][i], tr[1][i], tr[2][i], tr[3][i]))
#                 sample_weights.append(weight)

#             for i in range(len(va[3])):
#                 val_samples.append((va[0][i], va[1][i], va[2][i], va[3][i]))

#         sampler = WeightedRandomSampler(
#             sample_weights,
#             num_samples=len(sample_weights),
#             replacement=True
#         )

#         train_loader = DataLoader(
#             ListDataset(train_samples),
#             batch_size=32,
#             sampler=sampler,
#             collate_fn=collate_fn
#         )

#         val_loader = DataLoader(
#             ListDataset(val_samples),
#             batch_size=32,
#             shuffle=False,
#             collate_fn=collate_fn
#         )

#         model = CombinedModel(
#             [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#             [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#             num_heads=2,
#             embed_dim=532,
#             antigen_embed_dim=500
#         ).to(device)

#         optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#         criterion = nn.MSELoss()

#         best_val = np.inf
#         best_state = None

#         for epoch in range(50):
#             model.train()

#             for X_a, X_b, ag, y in train_loader:
#                 X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
#                 optimizer.zero_grad()
#                 pred = model(X_b, X_a, ag).view(-1)
#                 loss = criterion(pred, y)
#                 loss.backward()
#                 optimizer.step()

#             val_metrics = evaluate(model, val_loader, device)

#             if val_metrics["RMSE"] < best_val:
#                 best_val = val_metrics["RMSE"]
#                 best_state = copy.deepcopy(model.state_dict())

#         model.load_state_dict(best_state)

#         # ================================
#         # 3️⃣ Test + Benchmark
#         # ================================
#         for name, split in all_splits.items():
#             te = split["test"]
#             test_samples = [(te[0][i], te[1][i], te[2][i], te[3][i])
#                             for i in range(len(te[3]))]

#             test_loader = DataLoader(
#                 ListDataset(test_samples),
#                 batch_size=32,
#                 shuffle=False,
#                 collate_fn=collate_fn
#             )

#             metrics = evaluate(model, test_loader, device)
#             all_results[name].append(metrics)

#         # benchmark（完全独立）
#         bench_samples = [(benchmark_data["X_a"][i],
#                           benchmark_data["X_b"][i],
#                           benchmark_data["antigen"][i],
#                           benchmark_data["y"][i])
#                          for i in range(len(benchmark_data["y"]))]

#         bench_loader = DataLoader(
#             ListDataset(bench_samples),
#             batch_size=32,
#             shuffle=False,
#             collate_fn=collate_fn
#         )

#         bench_metrics = evaluate(model, bench_loader, device)
#         all_results["benchmark"].append(bench_metrics)

#     # ================================
#     # 4️⃣ 输出结果
#     # ================================
#     print("\n" + "="*70)
#     print("📊 FINAL MULTI-SEED RESULTS (mean ± std)")
#     print("="*70)

#     for name, results in all_results.items():
#         rmse = [r["RMSE"] for r in results]
#         mae  = [r["MAE"] for r in results]
#         pcc  = [r["PCC"] for r in results]

#         print(f"\n{name.upper()}")
#         print(f"RMSE: {np.mean(rmse):.4f} ± {np.std(rmse):.4f}")
#         print(f"MAE : {np.mean(mae):.4f} ± {np.std(mae):.4f}")
#         print(f"PCC : {np.mean(pcc):.4f} ± {np.std(pcc):.4f}")

# if __name__ == "__main__":
#     main()

# # -*- coding: utf-8 -*-
# """
# 训练脚本 - 适配 PWAA+LPE 模型架构 (LearnedPositionalAttention)
# 模型输入顺序：(antibody_light, antibody_heavy, antigen)
# """

# import torch
# import torch.nn as nn
# import numpy as np
# import random
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from scipy.stats import pearsonr
# from roformercnn import CombinedModel  # ✅ 修改：导入 PWAA+LPE 模型
# import torch.optim as optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import itertools
# import copy
# import os
# import warnings
# warnings.filterwarnings('ignore')

# # ======================================================
# # 固定随机种子
# # ======================================================
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# set_seed(42)

# # ======================================================
# # CDR 区域定义（Chothia）
# # ======================================================
# def getCDRPos(_loop, cdr_scheme='chothia'):
#     CDRS = {
#         'L1': ['24','25','26','27','28','29','30','30A','30B','30C','30D','30E','30F','30G','30H','30I','31','32','33','34'],
#         'L2': ['50','51','51A','52','52A','52B','52C','52D','53','54','55','56'],
#         'L3': ['89','90','91','92','93','94','95','95A','95B','95C','95D','95E','95F','95G','95H','95I','95J','96','97'],
#         'H1': ['26','27','28','29','30','31','31A','31B','31C','31D','31E','31F','31G','31H','31I','31J','32'],
#         'H2': ['52','52A','52B','52C','52D','52E','52F','52G','52H','52I','52J','52K','52L','52M','52N','52O','53','54','55','56'],
#         'H3': ['95','96','97','98','99','100','100A','100B','100C','100D','100E','100F','100G','100H',
#                '100I','100J','100K','100L','100M','100N','100O','100P','100Q','100R','100S','100T',
#                '100U','100V','100W','100X','100Y','100Z','101','102']
#     }
#     return CDRS[_loop]

# # ======================================================
# # Dataset
# # ======================================================
# class ListDataset(Dataset):
#     def __init__(self, samples):
#         self.samples = samples

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         return self.samples[idx]

# # ======================================================
# # Collate function
# # ======================================================
# def collate_fn(batch):
#     """
#     batch: list of (light, heavy, antigen, label)
#     返回：(light, heavy, antigen, label) 均为 [batch, seq_len, dim]
#     """
#     X_a_list = [torch.tensor(item[0], dtype=torch.float32) if not isinstance(item[0], torch.Tensor) else item[0] for item in batch]
#     X_b_list = [torch.tensor(item[1], dtype=torch.float32) if not isinstance(item[1], torch.Tensor) else item[1] for item in batch]
#     ag_list = [torch.tensor(item[2], dtype=torch.float32) if not isinstance(item[2], torch.Tensor) else item[2] for item in batch]
#     y_list = [torch.tensor(item[3], dtype=torch.float32) if not isinstance(item[3], torch.Tensor) else item[3] for item in batch]

#     max_len = max(
#         max(x.shape[0] for x in X_a_list),
#         max(x.shape[0] for x in X_b_list),
#         max(x.shape[0] for x in ag_list)
#     )

#     def pad_to_len(x, L):
#         if x.shape[0] < L:
#             pad = torch.zeros(L - x.shape[0], *x.shape[1:], dtype=x.dtype, device=x.device)
#             return torch.cat([x, pad], dim=0)
#         else:
#             return x[:L]

#     X_a_padded = torch.stack([pad_to_len(x, max_len) for x in X_a_list])
#     X_b_padded = torch.stack([pad_to_len(x, max_len) for x in X_b_list])
#     ag_padded = torch.stack([pad_to_len(x, max_len) for x in ag_list])
#     y_tensor = torch.stack(y_list)

#     if X_a_padded.dim() == 2:
#         X_a_padded = X_a_padded.unsqueeze(0)
#     if X_b_padded.dim() == 2:
#         X_b_padded = X_b_padded.unsqueeze(0)
#     if ag_padded.dim() == 2:
#         ag_padded = ag_padded.unsqueeze(0)

#     return X_a_padded, X_b_padded, ag_padded, y_tensor

# # ======================================================
# # Evaluation
# # ======================================================
# def evaluate(model, loader, device):
#     model.eval()
#     y_true, y_pred = [], []

#     with torch.no_grad():
#         for X_a, X_b, ag, y in loader:
#             X_a, X_b, ag, y = X_a.to(device), X_b.to(device), ag.to(device), y.to(device)
#             pred = model(X_a, X_b, ag).view(-1)
#             y_true.extend(y.cpu().numpy())
#             y_pred.extend(pred.cpu().numpy())

#     mse = mean_squared_error(y_true, y_pred)
#     rmse = np.sqrt(mse)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     pcc = pearsonr(y_true, y_pred)[0] if len(set(y_true)) > 1 else 0.0

#     return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "PCC": pcc}

# # ======================================================
# # Load dataset
# # ======================================================
# def load_dataset(path):
#     d = torch.load(path, map_location='cpu')
#     return {
#         "X_a": d["X_a"].cpu().numpy(),
#         "X_b": d["X_b"].cpu().numpy(),
#         "antigen": d["antigen"].cpu().numpy(),
#         "y": d["y"].cpu().numpy()
#     }

# # ======================================================
# # Split dataset
# # ======================================================
# def split_dataset(data, test_size=0.2, val_size=0.2):
#     X_a, X_b, ag, y = data.values()
#     X_a_tv, X_a_test, X_b_tv, X_b_test, ag_tv, ag_test, y_tv, y_test = train_test_split(
#         X_a, X_b, ag, y, test_size=test_size, random_state=42
#     )
#     val_ratio = val_size / (1 - test_size)
#     X_a_tr, X_a_val, X_b_tr, X_b_val, ag_tr, ag_val, y_tr, y_val = train_test_split(
#         X_a_tv, X_b_tv, ag_tv, y_tv, test_size=val_ratio, random_state=42
#     )
#     return {
#         "train": (X_a_tr, X_b_tr, ag_tr, y_tr),
#         "val": (X_a_val, X_b_val, ag_val, y_val),
#         "test": (X_a_test, X_b_test, ag_test, y_test)
#     }

# # ======================================================
# # Trainer
# # ======================================================
# class TrainerWithScheduler:
#     def __init__(self, model, train_loader, val_loader, params, device):
#         self.model = model.to(device)
#         self.train_loader = train_loader
#         self.val_loader = val_loader
#         self.device = device
#         self.opt = optim.Adam(
#             model.parameters(),
#             lr=params["lr"],
#             weight_decay=params["weight_decay"]
#         )
#         self.scheduler = ReduceLROnPlateau(
#             self.opt,
#             mode='min',
#             factor=params.get("lr_factor", 0.5),
#             patience=params.get("scheduler_patience", 3),
#             min_lr=params.get("min_lr", 1e-6),
#             verbose=False
#         )
#         self.criterion = nn.MSELoss()
#         self.epochs = params["epochs"]
#         self.patience = params["patience"]

#     def train(self):
#         best_mse = np.inf
#         best_state = None
#         wait = 0

#         for epoch in range(1, self.epochs + 1):
#             self.model.train()
#             total_loss = 0.0
#             num_batches = 0

#             for X_a, X_b, ag, y in self.train_loader:
#                 X_a, X_b, ag, y = X_a.to(self.device), X_b.to(self.device), ag.to(self.device), y.to(self.device)
#                 self.opt.zero_grad()
                
#                 pred = self.model(X_a, X_b, ag).view(-1)
                
#                 loss = self.criterion(pred, y)
#                 loss.backward()
#                 self.opt.step()

#                 total_loss += loss.item()
#                 num_batches += 1

#             avg_train_loss = total_loss / num_batches
#             val_metrics = evaluate(self.model, self.val_loader, self.device)
#             val_mse = val_metrics["MSE"]

#             self.scheduler.step(val_mse)

#             print(f"Epoch {epoch:02d}/{self.epochs} | "
#                   f"Train Loss: {avg_train_loss:.6f} | "
#                   f"Val MSE: {val_mse:.4f} | "
#                   f"Val R²: {val_metrics['R2']:.4f} | "
#                   f"Val PCC: {val_metrics['PCC']:.4f}")

#             if val_mse < best_mse:
#                 best_mse = val_mse
#                 best_state = copy.deepcopy(self.model.state_dict())
#                 wait = 0
#             else:
#                 wait += 1
#                 if wait >= self.patience:
#                     print("🛑 Early stopping triggered.")
#                     break

#         self.model.load_state_dict(best_state)
#         return self.model

# # ======================================================
# # MAIN
# # ======================================================
# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"🖥️  Using device: {device}")

#     paths = {
#         "train": "/tmp/AbAgCDR/data/train_data.pt",
#         "abbind": "/tmp/AbAgCDR/data/abbind_data.pt",
#         "sabdab": "/tmp/AbAgCDR/data/sabdab_data.pt",
#         "skempi": "/tmp/AbAgCDR/data/skempi_data.pt"
#     }
#     benchmark_path = "/tmp/AbAgCDR/data/benchmark_data.pt"

#     # ======================================================
#     # Load and split training datasets
#     # ======================================================
#     print("\n" + "="*70)
#     print("📊 加载并划分数据集")
#     print("="*70)
    
#     all_splits = {}
#     for name, path in paths.items():
#         if not os.path.exists(path):
#             print(f"⚠️  跳过 {name}: 文件不存在")
#             continue
#         print(f"加载 {name}...")
#         data = load_dataset(path)
#         all_splits[name] = split_dataset(data)
#         print(f"   train={len(data['y'])}, val={len(all_splits[name]['val'][3])}, test={len(all_splits[name]['test'][3])}")

#     # ======================================================
#     # Load benchmark
#     # ======================================================
#     print("\n" + "="*70)
#     print("📊 加载 Benchmark")
#     print("="*70)
    
#     if os.path.exists(benchmark_path):
#         benchmark_data = load_dataset(benchmark_path)
#         benchmark_samples = []
#         for i in range(len(benchmark_data["y"])):
#             benchmark_samples.append((
#                 benchmark_data["X_a"][i],
#                 benchmark_data["X_b"][i],
#                 benchmark_data["antigen"][i],
#                 benchmark_data["y"][i]
#             ))
#         benchmark_loader = DataLoader(
#             ListDataset(benchmark_samples),
#             batch_size=32,
#             shuffle=False,
#             collate_fn=collate_fn
#         )
#         print(f"✅ Benchmark 样本数：{len(benchmark_samples)}")
#     else:
#         print(f"⚠️  Benchmark 文件不存在：{benchmark_path}")
#         benchmark_loader = None

#     # ======================================================
#     # Collect samples with dataset weights
#     # ======================================================
#     print("\n" + "="*70)
#     print("📦 准备训练数据")
#     print("="*70)
    
#     all_train_samples = []
#     sample_weights = []
#     dataset_weights = {
#         'train': 4.0,
#         'abbind': 1.0,
#         'sabdab': 1.5,
#         'skempi': 1.5
#     }

#     for name in paths.keys():
#         if name not in all_splits:
#             continue
#         tr = all_splits[name]["train"]
#         w = dataset_weights.get(name, 0.1)
#         for i in range(len(tr[3])):
#             all_train_samples.append((tr[0][i], tr[1][i], tr[2][i], tr[3][i]))
#             sample_weights.append(w)

#     val_samples = []
#     for split in all_splits.values():
#         va = split["val"]
#         for i in range(len(va[3])):
#             val_samples.append((va[0][i], va[1][i], va[2][i], va[3][i]))

#     print(f"✅ 总训练样本：{len(all_train_samples)}")
#     print(f"✅ 总验证样本：{len(val_samples)}")

#     sampler = WeightedRandomSampler(
#         weights=sample_weights,
#         num_samples=len(sample_weights),
#         replacement=True
#     )

#     # ======================================================
#     # Hyperparameter Grid Search
#     # ======================================================
#     print("\n" + "="*70)
#     print("🔍 超参数网格搜索")
#     print("="*70)
    
#     param_grid = {
#         'lr': [1e-4, 5e-4, 1e-3],
#         'batch_size': [16, 32],
#         'epochs': [50],
#         'patience': [15],
#         'weight_decay': [1e-5, 1e-4],
#         'scheduler_patience': [3],
#         'lr_factor': [0.5],
#         'min_lr': [1e-6]
#     }

#     keys, values = zip(*param_grid.items())
#     param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

#     print(f"🚀 共 {len(param_combinations)} 组超参数组合")

#     best_score = -np.inf
#     best_params = None
#     best_model_state = None
    
#     try:
#         train_data_for_scaler = torch.load(paths["train"], map_location="cpu")
#         label_scaler = train_data_for_scaler.get("label_scaler", None)
#     except:
#         label_scaler = None

#     original_train_samples = all_train_samples
#     original_val_samples = val_samples

#     save_dir = "/tmp/AbAgCDR/model"
#     os.makedirs(save_dir, exist_ok=True)
#     # ✅ 修改：保存路径改为 PWAA+LPE
#     save_path = os.path.join(save_dir, "PWAA_LPE_best_model.pth")

#     for trial_idx, params in enumerate(param_combinations):
#         print(f"\n{'='*60}")
#         print(f"Trial {trial_idx+1}/{len(param_combinations)}")
#         print(f"{'='*60}")
#         print(f"Params: {params}")

#         current_bs = params['batch_size']
#         train_loader = DataLoader(
#             ListDataset(original_train_samples),
#             batch_size=current_bs,
#             sampler=sampler,
#             collate_fn=collate_fn,
#             shuffle=False,
#             num_workers=0
#         )
#         val_loader = DataLoader(
#             ListDataset(original_val_samples),
#             batch_size=current_bs,
#             shuffle=False,
#             collate_fn=collate_fn,
#             num_workers=0
#         )

#         # ✅ 修改：PWAA+LPE 模型参数 (embed_dim=512, antigen_embed_dim=480)
#         model = CombinedModel(
#             cdr_boundaries_light=[getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#             cdr_boundaries_heavy=[getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#             num_heads=2,
#             embed_dim=532,          # ✅ 修改：512 (不是 532)
#             antigen_embed_dim=500,  # ✅ 修改：480 (不是 500)
#             hidden_dim=256
#         )

#         total_params = sum(p.numel() for p in model.parameters())
#         trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#         print(f"   总参数量：{total_params:,}")
#         print(f"   可训练参数：{trainable_params:,}")

#         trainer = TrainerWithScheduler(model, train_loader, val_loader, params, device)
#         trained_model = trainer.train()

#         val_metrics = evaluate(trained_model, val_loader, device)
#         score = val_metrics['PCC']

#         print(f"\n✅ Val PCC: {score:.4f}")

#         if score > best_score:
#             best_score = score
#             best_params = copy.deepcopy(params)
#             best_model_state = copy.deepcopy(trained_model.state_dict())
            
#             torch.save({
#                 'model_state_dict': best_model_state,
#                 'params': best_params,
#                 'label_scaler': label_scaler,
#                 'config': {
#                     'model_type': 'PWAA+LPE',
#                     'embed_dim': 532,
#                     'antigen_embed_dim': 500,
#                     'hidden_dim': 256,
#                     'num_heads': 2
#                 }
#             }, save_path)
#             print(f"🎉 新最佳模型已保存：{save_path}")

#     # ======================================================
#     # Final evaluation with best model
#     # ======================================================
#     print("\n" + "="*70)
#     print("🏆 最佳模型测试")
#     print("="*70)
#     print(f"最佳超参数：{best_params}")
#     print(f"最佳 Val PCC: {best_score:.4f}")

#     final_model = CombinedModel(
#         cdr_boundaries_light=[getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#         cdr_boundaries_heavy=[getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#         num_heads=2,
#         embed_dim=532,
#         antigen_embed_dim=500,
#         hidden_dim=256
#     )
#     final_model.load_state_dict(best_model_state)
#     final_model.to(device)
#     final_model.eval()

#     print("\n" + "="*70)
#     print("🧪 各数据集测试结果")
#     print("="*70)

#     results = {}
#     for name, split in all_splits.items():
#         test_samples = []
#         te = split["test"]
#         for i in range(len(te[3])):
#             test_samples.append((te[0][i], te[1][i], te[2][i], te[3][i]))
#         test_loader = DataLoader(
#             ListDataset(test_samples), 
#             batch_size=32, 
#             shuffle=False, 
#             collate_fn=collate_fn,
#             num_workers=0
#         )
#         metrics = evaluate(final_model, test_loader, device)
#         results[name] = metrics
#         print(f"\n{name.upper()} TEST → "
#               f"R²: {metrics['R2']:.4f}, "
#               f"MSE: {metrics['MSE']:.4f}, "
#               f"RMSE: {metrics['RMSE']:.4f}, "
#               f"MAE: {metrics['MAE']:.4f}, "
#               f"PCC: {metrics['PCC']:.4f}")

#     if benchmark_loader is not None:
#         print("\n" + "="*70)
#         print("🎯 Benchmark 测试结果")
#         print("="*70)
#         bench_metrics = evaluate(final_model, benchmark_loader, device)
#         results['benchmark'] = bench_metrics
#         print(f"\n🎯 BENCHMARK TEST → "
#               f"R²: {bench_metrics['R2']:.4f}, "
#               f"MSE: {bench_metrics['MSE']:.4f}, "
#               f"RMSE: {bench_metrics['RMSE']:.4f}, "
#               f"MAE: {bench_metrics['MAE']:.4f}, "
#               f"PCC: {bench_metrics['PCC']:.4f}")

#     # ======================================================
#     # Save results
#     # ======================================================
#     print("\n" + "="*70)
#     print("💾 保存结果")
#     print("="*70)
    
#     results_path = os.path.join(save_dir, "PWAA_LPE_results.txt")
#     with open(results_path, 'w', encoding='utf-8') as f:
#         f.write("="*70 + "\n")
#         f.write("PWAA+LPE 模型训练结果\n")
#         f.write("="*70 + "\n\n")
#         f.write(f"最佳超参数：{best_params}\n")
#         f.write(f"最佳 Val PCC: {best_score:.4f}\n\n")
#         f.write("-"*70 + "\n")
#         f.write("测试结果\n")
#         f.write("-"*70 + "\n")
#         for name, metrics in results.items():
#             f.write(f"\n{name.upper()}:\n")
#             f.write(f"  R²:   {metrics['R2']:.4f}\n")
#             f.write(f"  MSE:  {metrics['MSE']:.4f}\n")
#             f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
#             f.write(f"  MAE:  {metrics['MAE']:.4f}\n")
#             f.write(f"  PCC:  {metrics['PCC']:.4f}\n")
    
#     print(f"✅ 结果已保存：{results_path}")
#     print(f"✅ 模型已保存：{save_path}")
    
#     print("\n" + "="*70)
#     print("✅ 训练完成！")
#     print("="*70)

# if __name__ == "__main__":
#     main()
