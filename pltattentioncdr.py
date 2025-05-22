import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import numpy as np
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
# RoformerAttention 模型定义
class RoformerAttention(nn.Module):
    def __init__(self, num_heads, embed_dim, dropout_rate=0.1, init_cdr_weight=2.0):
        super(RoformerAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim % 2 != 0:
            self.head_dim += 1
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须能够被头数整除"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim // 2, 1).float() / (self.head_dim // 2)))

        self.cdr_weight = nn.Parameter(torch.tensor(init_cdr_weight), requires_grad=True)

    def forward(self, x, cdr_mask=None):
        batch_size, seq_length, embed_dim = x.size()

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)

        inv_freq = self.inv_freq.to(x.device)
        cos_pos = torch.cos(inv_freq[:, None] * torch.arange(seq_length, device=x.device)[None, :]).unsqueeze(0)
        sin_pos = torch.sin(inv_freq[:, None] * torch.arange(seq_length, device=x.device)[None, :]).unsqueeze(0)

        cos_pos = cos_pos.to(x.device)
        sin_pos = sin_pos.to(x.device)
        cos_pos = cos_pos.permute(0, 2, 1)
        sin_pos = sin_pos.permute(0, 2, 1)

        cos_pos = cos_pos.expand(batch_size, self.num_heads, seq_length, self.head_dim // 2)
        sin_pos = sin_pos.expand(batch_size, self.num_heads, seq_length, self.head_dim // 2)

        Q_rot = torch.cat([
            Q[..., :self.head_dim // 2] * cos_pos - Q[..., self.head_dim // 2:] * sin_pos,
            Q[..., :self.head_dim // 2] * sin_pos + Q[..., self.head_dim // 2:] * cos_pos
        ], dim=-1)

        K_rot = torch.cat([
            K[..., :self.head_dim // 2] * cos_pos - K[..., self.head_dim // 2:] * sin_pos,
            K[..., :self.head_dim // 2] * sin_pos + K[..., self.head_dim // 2:] * cos_pos
        ], dim=-1)

        attention_scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if cdr_mask is not None:
            attention_scores = attention_scores + (cdr_mask * self.cdr_weight)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, embed_dim)
        out = self.out(out)

        return out, attention_weights

    def plot_position_encoding(self, seq_length):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim // 2, 1).float() / (self.head_dim // 2)))
        cos_pos = torch.cos(inv_freq[:, None] * torch.arange(seq_length, device='cpu')[None, :])
        sin_pos = torch.sin(inv_freq[:, None] * torch.arange(seq_length, device='cpu')[None, :])

        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(cos_pos.numpy(), cmap='viridis', aspect='auto')
        plt.title("Cosine Positional Encoding")
        plt.colorbar()

        plt.subplot(2, 1, 2)
        plt.imshow(sin_pos.numpy(), cmap='viridis', aspect='auto')
        plt.title("Sine Positional Encoding")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    def create_cdr_mask(self, batch_size, seq_length, cdr_boundaries, device):
        mask = torch.zeros(seq_length, seq_length, dtype=torch.float32, device=device)
        for cdr in cdr_boundaries:
            for pos in cdr:
                if pos.isdigit():
                    pos_index = int(pos)
                    mask[pos_index, pos_index] = 1
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = mask.expand(batch_size, self.num_heads, seq_length, seq_length)
        return mask

# # 加载数据
# train_data = torch.load("/tmp/AbAgCDR/data/train3_data_xin.pt")
test_data = torch.load("/tmp/AbAgCDR/data/benchmark_processedL5.pt")
# # 提取数据中的序列和CDR边界
# antibody_a_embed_dim = train_data['X_a_train']  # 轻链
# antibody_b_embed_dim = train_data['X_b_train']  # 重链
# 提取数据中的序列和CDR边界
antibody_a_embed_dim = test_data['ab_light']  # 轻链 X_a_train
antibody_b_embed_dim = test_data['ab_heavy']  # 重链 X_b_train

# CDR边界
cdr_boundaries_heavy = [
    getCDRPos('H1', cdr_scheme='chothia'),
    getCDRPos('H2', cdr_scheme='chothia'),
    getCDRPos('H3', cdr_scheme='chothia')
]
cdr_boundaries_light = [
    getCDRPos('L1', cdr_scheme='chothia'),
    getCDRPos('L2', cdr_scheme='chothia'),
    getCDRPos('L3', cdr_scheme='chothia')
]

# 获取轻链和重链的序列长度
seq_length_heavy = antibody_b_embed_dim.size(1)  # 重链序列长度
seq_length_light = antibody_a_embed_dim.size(1)  # 轻链序列长度

# 初始化模型
num_heads = 2
embed_dim = 532  # 根据输入数据的嵌入维度
model = RoformerAttention(num_heads, embed_dim)

# 将模型移到正确的设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 将输入数据和CDR掩码移到正确的设备
train_x_heavy = antibody_b_embed_dim.to(device)  # 重链输入数据
train_x_light = antibody_a_embed_dim.to(device)  # 轻链输入数据

# 获取batch size
batch_size = train_x_heavy.size(0)

# 将CDR掩码应用于Roformer模型
cdr_mask_heavy = model.create_cdr_mask(batch_size, seq_length_heavy, cdr_boundaries_heavy, device)
cdr_mask_light = model.create_cdr_mask(batch_size, seq_length_light, cdr_boundaries_light, device)

# RoformerAttention 和 CDR 掩码
output_heavy, attention_weights_heavy = model(train_x_heavy, cdr_mask=cdr_mask_heavy)  # 对重链进行处理
output_light, attention_weights_light = model(train_x_light, cdr_mask=cdr_mask_light)  # 对轻链进行处理

def plot_attention_weights(attention_weights, title_suffix="", show_ticks=True, cdr_boundaries=None, vmin=None, vmax=None):
    # 获取注意力权重矩阵，并截取第一个头的前十个元素
    attention_matrix = attention_weights[0, 0, :10, :10].cpu().detach().numpy()  # 只取第一个头的前10x10矩阵

    # 对权重进行归一化
    attention_matrix_normalized = (attention_matrix - np.min(attention_matrix)) / (np.max(attention_matrix) - np.min(attention_matrix))

    # 创建热力图
    plt.figure(figsize=(8, 6))

    # 绘制热力图，并设置颜色条范围
    im = plt.imshow(attention_matrix_normalized, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax)

    # 添加颜色条
    plt.colorbar(im, label='Attention Weights')

    # 设置标题和坐标轴标签
    plt.title(f"Attention Weights for {title_suffix}",fontsize=14)
    plt.xlabel("Sequence Position",fontsize=11)
    plt.ylabel("Sequence Position",fontsize=11)

    # 如果需要显示氨基酸位置
    if show_ticks:
        plt.xticks(np.arange(10), np.arange(10), rotation=45)
        plt.yticks(np.arange(10), np.arange(10))

    # 在热力图的每个单元格内添加权重值
    for i in range(attention_matrix_normalized.shape[0]):
        for j in range(attention_matrix_normalized.shape[1]):
            plt.text(j, i, f"{attention_matrix_normalized[i, j]:.2f}",
                     ha="center", va="center", color="black" if attention_matrix_normalized[i, j] > 0.5 else "white")

    plt.tight_layout()
    plt.show()

# 绘制重链和轻链的注意力权重热力图
vmin_value = 0.0  # 设置颜色条的最小值
vmax_value = 1.0  # 设置颜色条的最大值
plot_attention_weights(attention_weights_heavy, title_suffix="Heavy Chain", show_ticks=True, vmin=vmin_value, vmax=vmax_value)
plot_attention_weights(attention_weights_light, title_suffix="Light Chain", show_ticks=True, vmin=vmin_value, vmax=vmax_value)
# plot_attention_weights(attention_weights_heavy, seq_length_heavy, title_suffix=" (Heavy Chain)")
# plot_attention_weights(attention_weights_light, seq_length_light, title_suffix=" (Light Chain)")