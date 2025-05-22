
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from models.roformercnn import CombinedModel
import torch.nn.functional as F
import numpy as np
import joblib
# 设置随机种子以确保可重复性
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# # 数据加载函数
# 数据加载函数
def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    try:
        print(f"尝试读取文件: {file_path}")
        data = torch.load(file_path)
        light_embeddings = data['X_a_test']
        heavy_embeddings = data['X_b_test']
        antigen_embeddings = data['antigen_test']
        labels = data['y_test']  # 假设标签存储在 'y_test' 键中
        print("文件加载成功。")
        return light_embeddings, heavy_embeddings, antigen_embeddings, labels
    except Exception as e:
        print(f"加载数据时出错: {e}")
        raise

# #数据加载函数无真实数据标签
# def load_data(file_path):
#     if not os.path.exists(file_path):
#         print(f"文件不存在: {file_path}")
#         return None
#     try:
#         print(f"尝试读取文件: {file_path}")
#         data = torch.load(file_path)
#         light_embeddings = data['X_a_test']
#         heavy_embeddings = data['X_b_test']
#         antigen_embeddings = data['antigen_test']
#         print("文件加载成功。")
#         return light_embeddings, heavy_embeddings, antigen_embeddings
#     except Exception as e:
#         print(f"加载数据时出错: {e}")
#         raise
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
                       '31I', '31J',
                       '32'],
                'H2F': ['33', '34', '35',
                        '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51'],
                'H2': ['52', '52A', '52B', '52C', '52D', '52E', '52F', '52G', '52H', '52I', '52J', '52K', '52L', '52M',
                       '52N', '52O',
                       '53', '54', '55', '56'],
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

# # 处理填充和生成掩码的函数
# def process_with_mask(tensor, max_length):
#     """
#     对输入张量进行零填充，并生成对应的掩码。
#     :param tensor: 输入的张量
#     :param max_length: 填充到的最大长度
#     :return: 填充后的张量和对应的掩码
#     """
#     padded_tensor = F.pad(tensor, (0, 0, 0, max_length - tensor.size(1)))
#     mask = (torch.arange(max_length, device=tensor.device).unsqueeze(0) < tensor.size(1)).float()
#     return padded_tensor, mask
#
# def predict(model, data_loader, device, threshold=-11):
#     """
#     使用模型对输入数据进行预测。
#     :param model: 已加载的模型
#     :param data_loader: 数据加载器
#     :param device: 设备（CPU 或 GPU）
#     :param threshold: 分类的阈值
#     :return: 索引列表、原始预测值和分类结果
#     """
#     model.eval()  # 确保模型处于评估模式
#     original_predictions = []
#     classified_predictions = []
#     indices_list = []
#
#     with torch.no_grad():
#         print("开始执行预测...")
#         for batch in data_loader:
#             indices, light_cdr, heavy_cdr, antigen_global = batch
#             light_cdr = light_cdr.to(device)
#             heavy_cdr = heavy_cdr.to(device)
#             antigen_global = antigen_global.to(device)
#
#             # 获取最大长度
#             max_length = max(light_cdr.size(1), heavy_cdr.size(1), antigen_global.size(1))
#
#             # 处理填充和掩码
#             light_cdr_padded, mask_light = process_with_mask(light_cdr, max_length)
#             heavy_cdr_padded, mask_heavy = process_with_mask(heavy_cdr, max_length)
#             antigen_global_padded, mask_antigen = process_with_mask(antigen_global, max_length)
#
#             # 调用模型进行预测
#             outputs = model(
#                 light_cdr_padded,
#                 heavy_cdr_padded,
#                 antigen_global_padded
#             )
#             outputs = outputs.cpu().numpy()
#
#             # 保存预测结果
#             original_predictions.extend(outputs)
#
#             # 修正分类逻辑： <= 阈值为高亲和力 1， > 阈值为低亲和力 0
#             classified_outputs = [1 if pred <= threshold else 0 for pred in outputs]
#             classified_predictions.extend(classified_outputs)
#             indices_list.extend(indices.cpu().numpy())
#
#     # 统计分类结果
#     num_high_affinity = sum(classified_predictions)  # 分类为 1 的数量
#     num_low_affinity = len(classified_predictions) - num_high_affinity  # 分类为 0 的数量
#
#     print(f"预测完成。分类为高亲和力 (1) 的数量: {num_high_affinity}")
#     print(f"分类为低亲和力 (0) 的数量: {num_low_affinity}")
#
#     return indices_list, original_predictions, classified_predictions
#
#
# # 加载模型
# def load_model(model_path, cdr_boundaries_light, cdr_boundaries_heavy, num_heads, embed_dim, antigen_embed_dim, device):
#     """
#     加载并初始化模型。
#     :param model_path: 模型权重路径
#     :param cdr_boundaries_heavy: 重链的CDR边界
#     :param cdr_boundaries_light: 轻链的CDR边界
#     :param num_heads: 注意力头数量
#     :param embed_dim: 嵌入维度
#     :param antigen_embed_dim: 抗原嵌入维度
#     :param device: 设备（CPU 或 GPU）
#     :return: 加载的模型
#     """
#     model = CombinedModel(
#         cdr_boundaries_light=cdr_boundaries_light,
#         cdr_boundaries_heavy=cdr_boundaries_heavy,
#         num_heads=num_heads,
#         embed_dim=embed_dim,
#         antigen_embed_dim=antigen_embed_dim
#     )
#     model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
#     model.to(device)
#     return model
#
# # 主函数
# def main():
#     """
#     主函数：加载数据、模型并进行预测。
#     """
#     test_data_path = "/tmp/AbAgCDR/data/benchmark1.3.pt"
#     model_path = "/tmp/AbAgCDR/model/best_model_train_L3.5_state_dict.pth"
#     output_path = "/tmp/AbAgCDR/model/benchmark1.3.5.csv"
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # 获取重链和轻链的CDR边界chothia
#     cdr_boundaries_heavy = [
#         getCDRPos('H1', cdr_scheme='chothia'),
#         getCDRPos('H2', cdr_scheme='chothia'),
#         getCDRPos('H3', cdr_scheme='chothia')
#     ]
#
#     cdr_boundaries_light = [
#         getCDRPos('L1', cdr_scheme='chothia'),
#         getCDRPos('L2', cdr_scheme='chothia'),
#         getCDRPos('L3', cdr_scheme='chothia')
#     ]
#
#     # 模型参数
#     num_heads = 2
#     embed_dim = 532
#     antigen_embed_dim = 500
#
#     # 加载数据
#     light_embeddings, heavy_embeddings, antigen_embeddings = load_data(test_data_path)
#     if heavy_embeddings is None:
#         print("加载数据失败！")
#         return
#
#     indices = torch.arange(heavy_embeddings.size(0))
#     test_dataset = TensorDataset(indices, light_embeddings, heavy_embeddings, antigen_embeddings)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
#     # 加载模型
#     model = load_model(model_path, cdr_boundaries_light, cdr_boundaries_heavy, num_heads, embed_dim, antigen_embed_dim, device)
#
#     # 预测
#     threshold = -11
#     indices, original_predictions, classified_predictions = predict(model, test_loader, device, threshold)
#
#     # 保存预测结果到CSV文件
#     predictions_df = pd.DataFrame({
#         'Index': indices,
#         'Original_Predicted_Affinity': original_predictions,
#         'Classified_Predicted_Affinity': classified_predictions,
#     })
#     predictions_df.to_csv(output_path, index=False)
#     print(f"预测结果已保存到 {output_path}")
#
# if __name__ == "__main__":
#     main()

# 针对有真实标签的
# 处理填充和生成掩码的函数
def process_with_mask(tensor, max_length):
    """
    对输入张量进行零填充，并生成对应的掩码。
    :param tensor: 输入的张量
    :param max_length: 填充到的最大长度
    :return: 填充后的张量和对应的掩码
    """
    padded_tensor = F.pad(tensor, (0, 0, 0, max_length - tensor.size(1)))
    mask = (torch.arange(max_length, device=tensor.device).unsqueeze(0) < tensor.size(1)).float()
    return padded_tensor, mask

#预测函数
def predict(model, data_loader, device):    #, threshold=-12
    model.eval()
    original_predictions = []
    # classified_predictions = []  # 用于存储分类后的预测值
    label_list = []  # 用于存储真实值
    with torch.no_grad():
        print("开始执行预测...")
        for light_cdr, heavy_cdr, antigen_global, labels in data_loader:
            light_cdr = light_cdr.to(device)
            heavy_cdr = heavy_cdr.to(device)
            antigen_global = antigen_global.to(device)
            labels = labels.to(device)  # 确保标签也移到正确的设备

            # 确保所有张量在拼接的维度上具有相同的尺寸
            max_length = max(light_cdr.size(1), heavy_cdr.size(1), antigen_global.size(1))
            heavy_cdr_padded = F.pad(heavy_cdr, (0, 0, 0, max_length - heavy_cdr.size(1)))
            light_cdr_padded = F.pad(light_cdr, (0, 0, 0, max_length - light_cdr.size(1)))
            antigen_global_padded = F.pad(antigen_global, (0, 0, 0, max_length - antigen_global.size(1)))

            outputs = model(light_cdr_padded, heavy_cdr_padded, antigen_global_padded)
            outputs = outputs.cpu().numpy()  # 将预测值转换为numpy数组
            original_predictions.extend(outputs)  # 存储原始预测值
            # classified_outputs = [0 if pred > threshold else 1 for pred in outputs]  # 根据阈值分类
            # classified_predictions.extend(classified_outputs)  # 存储分类后的预测值
            label_list.extend(labels.cpu().numpy())  # 存储真实值
    print("预测完成。")
    return original_predictions, label_list  #classified_predictions,
# 加载模型
def load_model(model_path, cdr_boundaries_light, cdr_boundaries_heavy, num_heads, embed_dim, antigen_embed_dim, device):
    print(f"加载模型: {model_path}")
    model = CombinedModel(
        cdr_boundaries_light=cdr_boundaries_light,
        cdr_boundaries_heavy=cdr_boundaries_heavy,
        num_heads=num_heads,
        embed_dim=embed_dim,
        antigen_embed_dim=antigen_embed_dim
    )
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    print("模型加载完成。")
    return model

# 主函数
def main():
    """
    主函数：加载数据、模型并进行预测。/tmp/AbAgCDR/data/sabdab1_test_data.pt /tmp/AbAgCDR/data/pairs_seq_benchmark1.1_test.pt
    """
    test_data_path = "/tmp/AbAgCDR/data/sabdab1_test_data.pt"
    model_path = "/tmp/AbAgCDR/model/best_model_train_L3.7_state_dict.pth"
    # model_path = "/tmp/AbAgCDR/model/best_model_Abdab3_state_dict.pth"
    output_path = "/tmp/AbAgCDR/model/sabdab1.10_test_data.csv"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 获取重链和轻链的CDR边界chothia
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

    # 模型参数
    num_heads = 2
    embed_dim = 532
    antigen_embed_dim = 500

    light_embeddings, heavy_embeddings, antigen_embeddings, labels = load_data(test_data_path)
    if heavy_embeddings is None:
        return  # 如果文件不存在，终止程序

    test_dataset = TensorDataset(light_embeddings, heavy_embeddings, antigen_embeddings, labels)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = load_model(model_path, cdr_boundaries_light, cdr_boundaries_heavy, num_heads, embed_dim, antigen_embed_dim,
                       device)

    # threshold = -12  # 设置阈值
    # original_predictions, classified_predictions, true_values = predict(model, test_loader, device, threshold)
    original_predictions, true_values = predict(model, test_loader, device)

    predictions_df = pd.DataFrame({
        'Original_Predicted_Affinity': original_predictions,
        'True_Affinity': true_values
    })
    # predictions_df = pd.DataFrame({
    #     'Original_Predicted_Affinity': original_predictions,
    #     'Classified_Predicted_Affinity': classified_predictions,
    #     'True_Affinity': true_values
    # })
    predictions_df.to_csv(output_path, index=False)
    print(f"预测结果和真实值已保存到 {output_path}")


if __name__ == "__main__":
    main()

# # 下面的预测函数含有归一化
# def predict(model, data_loader, device, scaler_delta_g):
#     model.eval()
#     original_predictions = []
#     label_list = []  # 用于存储真实值
#     with torch.no_grad():
#         print("开始执行预测...")
#         for light_cdr, heavy_cdr, antigen_global, labels in data_loader:
#             light_cdr = light_cdr.to(device)
#             heavy_cdr = heavy_cdr.to(device)
#             antigen_global = antigen_global.to(device)
#             labels = labels.to(device)  # 确保标签也移到正确的设备
#
#             # 确保所有张量在拼接的维度上具有相同的尺寸
#             max_length = max(light_cdr.size(1), heavy_cdr.size(1), antigen_global.size(1))
#             heavy_cdr_padded = F.pad(heavy_cdr, (0, 0, 0, max_length - heavy_cdr.size(1)))
#             light_cdr_padded = F.pad(light_cdr, (0, 0, 0, max_length - light_cdr.size(1)))
#             antigen_global_padded = F.pad(antigen_global, (0, 0, 0, max_length - antigen_global.size(1)))
#
#             outputs = model(light_cdr_padded, heavy_cdr_padded, antigen_global_padded)
#             outputs = outputs.cpu().numpy()  # 将预测值转换为numpy数组
#
#             # 确保输出是二维数组
#             if outputs.ndim == 1:
#                 outputs = outputs.reshape(-1, 1)
#
#             # 进行逆变换
#             outputs = scaler_delta_g.inverse_transform(outputs).flatten()
#
#             original_predictions.extend(outputs)  # 存储原始预测值
#
#             # 将真实标签转换为 numpy 数组并确保是二维数组
#             labels_np = labels.cpu().numpy()
#             if labels_np.ndim == 1:
#                 labels_np = labels_np.reshape(-1, 1)
#
#             # 对真实标签进行逆变换
#             labels_inverse = scaler_delta_g.inverse_transform(labels_np).flatten()
#
#             label_list.extend(labels_inverse)  # 存储逆变换后的真实值
#     print("预测完成。")
#     return original_predictions, label_list
#
# # 加载模型
# def load_model(model_path, cdr_boundaries_light, cdr_boundaries_heavy, num_heads, embed_dim, antigen_embed_dim, device):
#     print(f"加载模型: {model_path}")
#     model = CombinedModel(
#         cdr_boundaries_light=cdr_boundaries_light,
#         cdr_boundaries_heavy=cdr_boundaries_heavy,
#         num_heads=num_heads,
#         embed_dim=embed_dim,
#         antigen_embed_dim=antigen_embed_dim
#     )
#     model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
#     model.to(device)
#     print("模型加载完成。")
#     return model
#
# # 主函数
# def main():
#     """
#     主函数：加载数据、模型并进行预测。/tmp/AbAgCDR/data/sabdab1_test_data.pt /tmp/AbAgCDR/data/pairs_seq_benchmark1.1_test.pt   final_dataset_train.tsv
#     """
#     test_data_path = "/tmp/AbAgCDR/data/sabdab1_test_data.pt"
#     model_path = "/tmp/AbAgCDR/model/best_model_train_L3.10_state_dict.pth"
#     output_path = "/tmp/AbAgCDR/model/sabdab1.10_test_data.csv"
#     scaler_path = "/tmp/AbAgCDR/model/scaler_sabdab1delta_g.pkl"  # 替换为实际的文件路径
#     # test_data_path = "/tmp/AbAgCDR/data/pairs_seq_benchmark1.1_test.pt"
#     # model_path = "/tmp/AbAgCDR/model/best_model_train_L3.11_state_dict.pth"
#     # output_path = "/tmp/AbAgCDR/model/pairs_seq_benchmark1.11_test.csv"
#     # scaler_path = "/tmp/AbAgCDR/model/scaler_pairs_seq_benchmark1.1delta_g.pkl"  # 替换为实际的文件路径
#     # test_data_path = "/tmp/AbAgCDR/data/pairs_seq_skempi.pt"
#     # model_path = "/tmp/AbAgCDR/model/best_model_train_L3.10_state_dict.pth"
#     # output_path = "/tmp/AbAgCDR/model/pairs_seq_skempi3.10.csv"
#     # scaler_path = "/tmp/AbAgCDR/model/pairs_seq_skempidelta_g.pkl"  # 替换为实际的文件路径
#     # test_data_path = "/tmp/AbAgCDR/data/pairs_seq_abbind2.pt"
#     # model_path = "/tmp/AbAgCDR/model/best_model_train_L3.10_state_dict.pth"
#     # output_path = "/tmp/AbAgCDR/model/pairs_seq_abbind23.10.csv"
#     # scaler_path = "/tmp/AbAgCDR/model/pairs_seq_abbind2delta_g.pkl"  # 替换为实际的文件路径
#
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # 获取重链和轻链的CDR边界chothia
#     cdr_boundaries_heavy = [
#         getCDRPos('H1', cdr_scheme='chothia'),
#         getCDRPos('H2', cdr_scheme='chothia'),
#         getCDRPos('H3', cdr_scheme='chothia')
#     ]
#
#     cdr_boundaries_light = [
#         getCDRPos('L1', cdr_scheme='chothia'),
#         getCDRPos('L2', cdr_scheme='chothia'),
#         getCDRPos('L3', cdr_scheme='chothia')
#     ]
#
#     # 模型参数
#     num_heads = 2
#     embed_dim = 532
#     antigen_embed_dim = 500
#
#     light_embeddings, heavy_embeddings, antigen_embeddings, labels = load_data(test_data_path)
#     if heavy_embeddings is None:
#         return  # 如果文件不存在，终止程序
#
#     test_dataset = TensorDataset(light_embeddings, heavy_embeddings, antigen_embeddings, labels)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
#
#     model = load_model(model_path, cdr_boundaries_light, cdr_boundaries_heavy, num_heads, embed_dim, antigen_embed_dim,
#                        device)
#
#     # 加载保存的 StandardScaler 对象
#     try:
#         scaler_delta_g = joblib.load(scaler_path)
#     except FileNotFoundError:
#         print(f"未找到文件: {scaler_path}")
#         return
#
#     original_predictions, true_values = predict(model, test_loader, device, scaler_delta_g)
#
#     predictions_df = pd.DataFrame({
#         'Original_Predicted_Affinity': original_predictions,
#         'True_Affinity': true_values
#     })
#     predictions_df.to_csv(output_path, index=False)
#     print(f"预测结果和真实值已保存到 {output_path}")
#
#
# if __name__ == "__main__":
#     main()