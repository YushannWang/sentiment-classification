from data_loader import load_local_imdb
from model_loader import load_tokenizer, load_model
from utils import preprocess_function, compute_metrics
from datasets import load_dataset
import os
import shutil
import numpy as np
from transformers import TrainingArguments, Trainer
from plot_utils import MetricsCallback, plot_metrics

def main():
    # 1. 加载数据
    save_dir = "./data"
    train_subset_path = os.path.join(save_dir, "train.csv")
    val_subset_path = os.path.join(save_dir, "val.csv")
    
    if os.path.exists(train_subset_path) and os.path.exists(val_subset_path):
        print("加载已保存的训练集和验证集子集...")
        train_dataset = load_dataset("csv", data_files=train_subset_path, encoding="latin-1")["train"]
        val_dataset = load_dataset("csv", data_files=val_subset_path, encoding="latin-1")["train"]
    else:
        print("处理原始数据并生成子集...")
        train_dataset, val_dataset, _, _ = load_local_imdb(
            csv_path="imdb_master.csv", 
            save_dir=save_dir
        )
    
    # 2. 加载分词器和模型
    tokenizer = load_tokenizer()
    model = load_model()
    
    # 3. 数据预处理
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["review"]
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=["review"]
    )
    
    # 4. 确保标签格式正确
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_val = tokenized_val.rename_column("label", "labels")
    tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    tokenized_val.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # 初始化自定义回调实例
    metrics_callback = MetricsCallback()
    
    # 5. 定义训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5, 
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,     
        save_strategy="epoch",
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy"
    )
    
    # 6. 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[metrics_callback]  
    )
    
    # 7. 开始训练
    print("开始训练...")
    trainer.train()
    
    # 8. 调用绘图工具绘制并保存曲线
    plot_metrics(metrics_callback)
    
    # 9. 分析结果，找到准确率最高的epoch
    if metrics_callback.epochs:
        best_idx = np.argmax(metrics_callback.val_accuracies)
        best_epoch = metrics_callback.epochs[best_idx]
        best_accuracy = metrics_callback.val_accuracies[best_idx]
        best_loss = metrics_callback.val_losses[best_idx]
        
        
        print("\n===== 各epoch准确率汇总 =====")
        for epoch, acc, loss in zip(metrics_callback.epochs, metrics_callback.val_accuracies, metrics_callback.val_losses):
            print(f"Epoch {epoch}: 准确率 = {acc:.4f}, 损失 = {loss:.4f}")
        
        
        print(f"\n===== 最佳Epoch是第 {best_epoch} 个 =====")
        print(f"最佳准确率: {best_accuracy:.4f}")
        print(f"最佳验证损失: {best_loss:.4f}")
        
       
        best_model_dir = "./best_epoch_model"
        
        steps_per_epoch = len(tokenized_train) // training_args.per_device_train_batch_size
        best_checkpoint = os.path.join("./results", f"checkpoint-{best_epoch * steps_per_epoch}")
        
       
        if not os.path.exists(best_checkpoint):
            possible_checkpoints = [f for f in os.listdir("./results") if f.startswith("checkpoint-")]
            if possible_checkpoints:
                best_checkpoint = os.path.join("./results", possible_checkpoints[best_idx])
        
        if os.path.exists(best_model_dir):
            shutil.rmtree(best_model_dir)
        
        if os.path.exists(best_checkpoint):
            shutil.copytree(best_checkpoint, best_model_dir)
            print(f"最佳epoch的模型已复制到: {best_model_dir}")
        else:
            print(f"警告: 未找到最佳模型的checkpoint路径: {best_checkpoint}")

if __name__ == "__main__":
    main()
    