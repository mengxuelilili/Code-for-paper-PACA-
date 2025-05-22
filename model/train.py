

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from models.roformercnn import CombinedModel  # 确保这个路径是正确的
import torch.optim as optim
from scipy.stats import pearsonr
import joblib


def dynamic_pad_collate(batch, max_lengths):
    """根据训练集最大长度进行动态填充"""
    ab_light_batch, ab_heavy_batch, antigen_batch, delta_g_batch = [], [], [], []

    for item in batch:
        ab_light, ab_heavy, antigen, delta_g = item

        
        ab_light_padded = torch.zeros((max_lengths['ab_light'], ab_light.shape[-1]),
                                      device=ab_light.device)
        ab_light_padded[:min(ab_light.shape[0], max_lengths['ab_light'])] = \
            ab_light[:max_lengths['ab_light']]

       
        ab_heavy_padded = torch.zeros((max_lengths['ab_heavy'], ab_heavy.shape[-1]),
                                      device=ab_heavy.device)
        ab_heavy_padded[:min(ab_heavy.shape[0], max_lengths['ab_heavy'])] = \
            ab_heavy[:max_lengths['ab_heavy']]

        antigen_padded = torch.zeros((max_lengths['antigen'], antigen.shape[-1]),
                                     device=antigen.device)
        antigen_padded[:min(antigen.shape[0], max_lengths['antigen'])] = \
            antigen[:max_lengths['antigen']]

        ab_light_batch.append(ab_light_padded)
        ab_heavy_batch.append(ab_heavy_padded)
        antigen_batch.append(antigen_padded)
        delta_g_batch.append(delta_g)

    return (
        torch.stack(ab_light_batch),
        torch.stack(ab_heavy_batch),
        torch.stack(antigen_batch),
        torch.tensor(delta_g_batch, dtype=torch.float32)
    )
# 
class CustomDataset(Dataset):
    def __init__(self, light_chains, heavy_chains, antigen, delta_g):
        self.light_chains = light_chains
        self.heavy_chains = heavy_chains
        self.antigen = antigen
        self.delta_g = delta_g

    def __len__(self):
        return len(self.delta_g)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index out of range")
        return (self.light_chains[idx], self.heavy_chains[idx], self.antigen[idx], self.delta_g[idx])

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

class EnhancedModelTrainer:
    def __init__(self, model, train_loader, val_loader, params, device, max_lengths):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.params = params
        self.device = device
        self.max_lengths = max_lengths
        # 损失权重配置
        self.mse_weight = params.get('mse_weight', 1.0)
        self.pearson_weight = params.get('pearson_weight', 1.0)

        # 初始化损失函数和优化器
        self.mse_loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['lr'],
            weight_decay=params.get('weight_decay', 1e-7)
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=params['patience'], factor=0.5, verbose=True
        )

        # 最佳模型跟踪
        self.best_metrics = {
            'mse': np.inf, 'rmse': np.inf, 'mae': np.inf,
            'r2': -np.inf, 'pearson': -np.inf
        }
        self.best_model_state = None

    def pearson_loss(self, pred, target):
        """可微分皮尔逊相关系数损失"""
        pred = pred.squeeze()
        target = target.squeeze()

        # 计算统计量
        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)
        cov = torch.mean((pred - pred_mean) * (target - target_mean))
        pred_std = torch.std(pred)
        target_std = torch.std(target)

        # 计算皮尔逊系数
        pearson = cov / (pred_std * target_std + 1e-8)
        return 1.0 - pearson  # 转换为损失形式

    def train(self, num_epochs=50):
        early_stop_counter = 0

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            mse_loss_total = 0.0
            pearson_loss_total = 0.0

            for ab_light, ab_heavy, antigen, delta_g in self.train_loader:
                # 数据准备
                inputs = self._prepare_inputs(ab_light, ab_heavy, antigen)
                labels = delta_g.to(self.device).float()

                # 梯度清零
                self.optimizer.zero_grad()

                # 前向传播
                outputs = self.model(*inputs).squeeze()

                # 计算损失
                mse_loss = self.mse_loss(outputs, labels)
                p_loss = self.pearson_loss(outputs, labels)
                total_loss = self.mse_weight * mse_loss + self.pearson_weight * p_loss

                # 反向传播
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                # 记录损失
                total_train_loss += total_loss.item()
                mse_loss_total += mse_loss.item()
                pearson_loss_total += p_loss.item()

            # 计算平均损失
            avg_total_loss = total_train_loss / len(self.train_loader)
            avg_mse = mse_loss_total / len(self.train_loader)
            avg_pearson_loss = pearson_loss_total / len(self.train_loader)

            # 验证阶段
            val_metrics = self.validate()
            self.scheduler.step(val_metrics['mse'])

            # # 早停机制
            # if val_metrics['mse'] < self.best_metrics['mse'] and val_metrics['pearson'] > self.best_metrics['pearson']:
            #     self.best_metrics = val_metrics
            #     self.best_model_state = self.model.state_dict()
            #     early_stop_counter = 0
            # else:
            #     early_stop_counter += 1
            #     if early_stop_counter >= self.params['patience']:
            #         print(f"Early stopping at epoch {epoch}")
            #         break

            # 打印训练信息
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Total Loss: {avg_total_loss:.4f} | MSE: {avg_mse:.4f} | Pearson Loss: {avg_pearson_loss:.4f}")
            print(f"Val MSE: {val_metrics['mse']:.4f} | Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.4f} | Val R2: {val_metrics['r2']:.4f}")
            print(f"Val Pearson: {val_metrics['pearson']:.4f}")
            print("--------------------------")

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        return self.model

    def validate(self):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for ab_light, ab_heavy, antigen, delta_g in self.val_loader:
                inputs = self._prepare_inputs(ab_light, ab_heavy, antigen)
                labels = delta_g.to(self.device).float()

                outputs = self.model(*inputs).squeeze()

                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        return self._calculate_metrics(np.concatenate(all_preds), np.concatenate(all_labels))

    def _prepare_inputs(self, ab_light, ab_heavy, antigen):
        def _process_tensor(tensor, max_len):
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(-1)
            return tensor[:, :max_len, :]

        return (
            _process_tensor(ab_light.to(self.device), self.max_lengths['ab_light']),
            _process_tensor(ab_heavy.to(self.device), self.max_lengths['ab_heavy']),
            _process_tensor(antigen.to(self.device), self.max_lengths['antigen'])
        )

    @staticmethod
    def _calculate_metrics(preds, labels):
        valid_mask = ~np.isnan(preds) & ~np.isnan(labels)
        preds = preds[valid_mask]
        labels = labels[valid_mask]

        if len(preds) < 2:
            return {k: np.nan for k in ['mse', 'rmse', 'mae', 'r2', 'pearson']}

        return {
            'mse': mean_squared_error(labels, preds),
            'rmse': np.sqrt(mean_squared_error(labels, preds)),
            'mae': mean_absolute_error(labels, preds),
            'r2': r2_score(labels, preds),
            'pearson': pearsonr(labels, preds)[0]
        }



def enhanced_grid_search(train_data, params_grid):
    # 加载预处理参数
    train_params = joblib.load('/tmp/AbAgCDR/dataset/train_paramsL5.joblib')
    max_lengths = {
        'ab_light': train_params['max_ab_light_length'],
        'ab_heavy': train_params['max_ab_heavy_length'],
        'antigen': train_params['max_antigen_length']
    }

    # 划分数据集
    X_a_train_val, X_a_test, X_b_train_val, X_b_test, antigen_train_val, antigen_test, y_train_val, y_test = train_test_split(
        train_data['ab_light'], train_data['ab_heavy'], train_data['antigen'], train_data['delta_g'],
        test_size=0.2, random_state=42
    )

    # 创建数据集
    train_val_dataset = CustomDataset(X_a_train_val, X_b_train_val, antigen_train_val, y_train_val)
    test_dataset = CustomDataset(X_a_test, X_b_test, antigen_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=lambda b: dynamic_pad_collate(b, max_lengths))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_params = None
    best_avg_metrics = {'mse': np.inf, 'rmse': np.inf, 'mae': np.inf, 'r2': -np.inf, 'pearson': -np.inf}
    best_model_state = None

    # 参数搜索
    for params in ParameterGrid(params_grid):
        print(f"\n=== Training with params: {params} ===")
        kf = KFold(n_splits=params['num_folds'], shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_val_dataset)):
            print(f"\nFold {fold + 1}/{params['num_folds']}")

            # 数据加载器
            train_loader = DataLoader(
                Subset(train_val_dataset, train_idx),
                batch_size=params['batch_size'],
                shuffle=True,
                collate_fn=lambda b: dynamic_pad_collate(b, max_lengths)
            )
            val_loader = DataLoader(
                Subset(train_val_dataset, val_idx),
                batch_size=params['batch_size'],
                collate_fn=lambda b: dynamic_pad_collate(b, max_lengths)
            )

            # 初始化模型
            model = CombinedModel(
                cdr_boundaries_light=[getCDRPos(s, 'chothia') for s in ['L1','L2','L3']],
                cdr_boundaries_heavy=[getCDRPos(s, 'chothia') for s in ['H1','H2','H3']],
                num_heads=params['num_heads'],
                embed_dim=532,
                antigen_embed_dim=500
            ).to(device)

            # 训练
            trainer = EnhancedModelTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                params=params,
                device=device,
                max_lengths=max_lengths
            )
            trained_model = trainer.train(num_epochs=50)

            # 验证
            val_metrics = trainer.validate()
            fold_metrics.append(val_metrics)

            # 释放资源
            del model, trainer
            torch.cuda.empty_cache()

        # 计算平均指标
        avg_metrics = {
            k: np.nanmean([m[k] for m in fold_metrics])
            for k in ['mse', 'rmse', 'mae', 'r2', 'pearson']
        }
        #
        # # 更新最佳参数
        # if avg_metrics['mse'] < best_avg_metrics['mse'] and avg_metrics['pearson'] > best_avg_metrics['pearson']:
        #     best_avg_metrics = avg_metrics
        #     best_params = params.copy()
        #     best_model_state = trained_model.state_dict()
        if (avg_metrics['pearson'] > best_avg_metrics['pearson']) and (avg_metrics['r2'] > best_avg_metrics['r2']):
            best_avg_metrics = avg_metrics
            best_params = params.copy()
            best_model_state = trained_model.state_dict()
    # 最终测试评估
    print("\n=== Final Test Evaluation ===")
    final_model = CombinedModel(
        cdr_boundaries_light=[getCDRPos(s, 'chothia') for s in ['L1', 'L2', 'L3']],
        cdr_boundaries_heavy=[getCDRPos(s, 'chothia') for s in ['H1', 'H2', 'H3']],
        num_heads=best_params['num_heads'],
        embed_dim=532,
        antigen_embed_dim=500
    ).to(device)
    final_model.load_state_dict(best_model_state)

    test_metrics = validate_model(final_model, test_loader)
    print(f"Test MSE: {test_metrics['mse']:.4f} | Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"Test MAE: {test_metrics['mae']:.4f} | Test R2: {test_metrics['r2']:.4f}")
    print(f"Test Pearson: {test_metrics['pearson']:.4f}")

    return final_model, test_metrics


def validate_model(model, loader):
    model.eval()
    device = next(model.parameters()).device
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ab_light, ab_heavy, antigen, delta_g in loader:
            inputs = (
                ab_light.to(device),
                ab_heavy.to(device),
                antigen.to(device)
            )
            labels = delta_g.to(device).float()

            outputs = model(*inputs).squeeze()

            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return EnhancedModelTrainer._calculate_metrics(
        np.concatenate(all_preds),
        np.concatenate(all_labels)
    )

# 使用示例
if __name__ == "__main__":
    # 加载预处理数据
    train_data = torch.load("/tmp/AbAgCDR/data/train_processedL5.pt")

    # 定义参数网格
    # 扩展参数网格
    params_grid = {
        'lr': [1e-3, 5e-4],
        'batch_size': [32, 64],
        'num_folds': [5],
        'patience': [10],
        'num_heads': [2, 4],
        'weight_decay': [5e-7, 1e-6],
        'dropout': [0.1, 0.2],
        'mse_weight': [0.4],     # 新增参数
        'pearson_weight': [0.6]   # 新增参数
    }

    # 执行网格搜索
    best_model, test_metrics = enhanced_grid_search(train_data, params_grid)

    print("\n=== Final Test Metrics ===")
    for metric, value in test_metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

    torch.save(best_model.state_dict(), "/tmp/AbAgCDR/model/best_modelL5.pth")


