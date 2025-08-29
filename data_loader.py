import os
import pandas as pd
from datasets import Dataset

def load_local_imdb(csv_path="imdb_master.csv", save_dir="./data"):
    """
    加载本地IMDb CSV数据集，先过滤unsup样本，仅保留pos/neg，再处理标签和划分子集
    """
    # 1. 检查CSV文件是否存在
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在：{csv_path}")
    
    # 2. 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    print(f"数据集处理目录：{save_dir}")
    
    # 3. 加载CSV并**立即过滤unsup样本**
    print(f"加载CSV数据并过滤unsup样本...")
    try:
        
        df = pd.read_csv(csv_path, encoding="latin-1")
        df_filtered = df[df["label"].str.lower().isin(["pos", "neg"])]
        total_raw = len(df)
        total_filtered = len(df_filtered)
        print(f"原始样本数：{total_raw}，过滤后（仅pos/neg）：{total_filtered}，"
              f"移除unsup等无效样本：{total_raw - total_filtered}")
        data_list = df_filtered.to_dict("records")
    except Exception as e:
        raise RuntimeError(f"加载或过滤CSV失败：{str(e)}")
    
    # 4. 转换标签（pos→1，neg→0）
    processed_list = []
    for item in data_list:
        try:
            label_lower = item["label"].lower()  
            processed_item = {
                "review": item.get("review", ""),  
                "label": 1 if label_lower == "pos" else 0
            }
            processed_list.append(processed_item)
        except Exception as e:
            print(f"样本处理警告（已跳过）：{str(e)}") 
    
    # 5. 检查有效样本数量
    total_valid = len(processed_list)
    if total_valid < 1200:
        raise ValueError(f"有效样本不足1200条（仅{total_valid}条），无法划分子集")
    
    # 6. 转换为Dataset格式并打乱
    processed_dataset = Dataset.from_list(processed_list)
    shuffled_dataset = processed_dataset.shuffle(seed=42)  
    
    # 7. 划分训练集和验证集
    small_train = shuffled_dataset.select(range(1000))     # 1000条训练
    small_val = shuffled_dataset.select(range(1000, 1200))  # 200条验证
    
    # 8. 保存子集
    train_save_path = os.path.join(save_dir, "train.csv")
    val_save_path = os.path.join(save_dir, "val.csv")
    small_train.to_csv(train_save_path, index=False)
    small_val.to_csv(val_save_path, index=False)
    print(f"子集保存完成：\n- 训练集：{train_save_path}（1000条）\n- 验证集：{val_save_path}（200条）")
    
    return small_train, small_val, train_save_path, val_save_path

if __name__ == "__main__":
    try:
        train_ds, val_ds, train_path, val_path = load_local_imdb()
        
        # 验证结果
        print("\n===== 数据验证 =====")
        print(f"训练集标签分布：1（pos）={sum(1 for x in train_ds if x['label']==1)}条，"
              f"0（neg）={sum(1 for x in train_ds if x['label']==0)}条")
        print(f"验证集标签分布：1（pos）={sum(1 for x in val_ds if x['label']==1)}条，"
              f"0（neg）={sum(1 for x in val_ds if x['label']==0)}条")
        print("\n数据处理成功：仅保留pos/neg样本，标签已转换为0/1")
    except Exception as e:
        print(f"处理失败：{str(e)}")
