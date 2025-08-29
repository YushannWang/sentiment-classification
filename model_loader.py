# model_loader.py 
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_tokenizer(model_name="distilbert-base-uncased"):
    """加载预训练分词器"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_model(model_name="distilbert-base-uncased", num_labels=2):
    """加载带分类头的预训练模型"""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels  
    )
    return model

if __name__ == "__main__":
    # 测试模型和分词器加载
    tokenizer = load_tokenizer()
    model = load_model()
    print("分词器和模型加载成功")

