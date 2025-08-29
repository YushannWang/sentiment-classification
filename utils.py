from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def preprocess_function(examples, tokenizer):
    """文本预处理：分词、截断、填充"""
    return tokenizer(
        examples["review"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt"
    )

def compute_metrics(pred):
    """计算评估指标（准确率、精确率、召回率、F1分数）"""
    labels = pred.label_ids  
    preds = pred.predictions.argmax(-1) 
    
    # 计算各指标
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')  
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')
    
    # 返回所有指标
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
