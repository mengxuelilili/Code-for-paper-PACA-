import torch
import numpy as np
from roformercnn import CombinedModel
from xiaorong import *  # 导入您的数据加载函数

def evaluate_seed_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载验证集（和训练时一样的划分）
    paths = {
        "paddle": "/root/autodl-tmp/AbAgCDR/data_split/paddle_test.pt",
        "abbind": "/root/autodl-tmp/AbAgCDR/data_split/abbind_test.pt",
        "sabdab": "/root/autodl-tmp/AbAgCDR/data_split/sabdab_test.pt",
        "skempi": "/root/autodl-tmp/AbAgCDR/data_split/skempi_test.pt"
    }
    
    all_val_samples = []
    for name, path in paths.items():
        if not os.path.exists(path): continue
        data = load_dataset(path)
        # 使用相同的划分方式（seed=42）
        split = split_dataset(data, seed=42)
        va = split["val"]
        for i in range(len(va[3])):
            all_val_samples.append((va[0][i], va[1][i], va[2][i], va[3][i]))
    
    val_loader = DataLoader(
        ListDataset(all_val_samples),
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # 2. 评估5个种子
    seeds = [0, 1, 2, 3, 42]
    results = {}
    
    model_dir = "/tmp/AbAgCDR/model"  # 改成您的实际路径
    
    for seed in seeds:
        model = CombinedModel(
            [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
            [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
            num_heads=2, embed_dim=532, antigen_embed_dim=500
        )
        
        model_path = os.path.join(model_dir, f"PWAARPEbest_model_seed_{seed}.pth")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            metrics = evaluate(model, val_loader, device)
        
        results[seed] = metrics
        print(f"Seed {seed}: MSE={metrics['MSE']:.4f}, RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}, PCC={metrics['PCC']:.4f}")
    
    # 3. 计算均值和标准差
    mse_values = [results[s]['MSE'] for s in seeds]
    rmse_values = [results[s]['RMSE'] for s in seeds]
    mae_values = [results[s]['MAE'] for s in seeds]
    r2_values = [results[s]['R2'] for s in seeds]
    pcc_values = [results[s]['PCC'] for s in seeds]
    
    print(f"\n📊 Summary on Validation Set:")
    print(f"MSE:  {np.mean(mse_values):.4f} ± {np.std(mse_values, ddof=1):.4f}")
    print(f"RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values, ddof=1):.4f}")
    print(f"MAE:  {np.mean(mae_values):.4f} ± {np.std(mae_values, ddof=1):.4f}")
    print(f"R²:   {np.mean(r2_values):.4f} ± {np.std(r2_values, ddof=1):.4f}")
    print(f"PCC:  {np.mean(pcc_values):.4f} ± {np.std(pcc_values, ddof=1):.4f}")
    
    return results

if __name__ == "__main__":
    evaluate_seed_models()


#     seeds = [0, 1, 2, 3, 43]
#     results = {}
    
#     for seed in seeds:
#         # 加载模型
#         model = CombinedModel(
#             [getCDRPos("H1"), getCDRPos("H2"), getCDRPos("H3")],
#             [getCDRPos("L1"), getCDRPos("L2"), getCDRPos("L3")],
#             num_heads=2, embed_dim=532, antigen_embed_dim=500
#         )
        
#         model_path = f"/tmp/AbAgCDR/model/best_model_seed_{seed}.pth"  # 改成您的实际路径
#         checkpoint = torch.load(model_path, map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         model.to(device)
#         model.eval()
        
#         # 在验证集上评估
#         metrics = evaluate(model, val_loader, device)
#         results[seed] = metrics
#         print(f"Seed {seed}: R²={metrics['R2']:.4f}, PCC={metrics['PCC']:.4f}")
    
#     # 3. 计算均值和标准差
#     pcc_values = [results[s]['PCC'] for s in seeds]
#     r2_values = [results[s]['R2'] for s in seeds]
    
#     print(f"\n📊 Summary on Validation Set:")
#     print(f"PCC: {np.mean(pcc_values):.4f} ± {np.std(pcc_values, ddof=1):.4f}")
#     print(f"R²:  {np.mean(r2_values):.4f} ± {np.std(r2_values, ddof=1):.4f}")
    
#     return results

# if __name__ == "__main__":
#     evaluate_seed_models()