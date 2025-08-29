import os
import matplotlib.pyplot as plt
import numpy as np
from transformers import TrainerCallback


plt.rcParams["font.family"] = ["sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  


class MetricsCallback(TrainerCallback):
    """自定义回调类用于记录训练过程中的指标"""
    def __init__(self):
        self.train_losses = []  
        self.train_steps = []   
        self.val_losses = []    
        self.val_accuracies = []
        self.epochs = []        

    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录训练过程中的损失"""
        if logs is not None and "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(state.global_step)

    def on_evaluate(self, args, state, control, metrics=None,** kwargs):
        """记录每个epoch的验证指标"""
        if metrics is not None and state.epoch is not None:
            epoch = int(state.epoch)
            self.epochs.append(epoch)
            self.val_losses.append(metrics.get("eval_loss", 0))
            self.val_accuracies.append(metrics.get("eval_accuracy", 0))
            print(f"Epoch {epoch} 评估结果: 准确率 = {metrics.get('eval_accuracy', 0):.4f}, 损失 = {metrics.get('eval_loss', 0):.4f}")

def plot_metrics(callback, save_dir="./plots"):
    """绘制训练过程中的损失和准确率曲线"""
    # 创建保存图表的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    # 子图1: 训练损失和验证损失
    plt.subplot(1, 2, 1)
    steps_per_epoch = len(callback.train_steps) // len(callback.epochs) if len(callback.epochs) > 0 else 1    
    plt.plot(callback.train_steps, callback.train_losses, label="Training Loss", color='blue', alpha=0.7)
    val_steps = [i * steps_per_epoch for i in range(1, len(callback.val_losses)+1)]
    plt.plot(val_steps, callback.val_losses, label="Validation Loss", color='red', marker='o')
    
    plt.title("Training and Validation Loss Trends")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss Value")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 子图2: 验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(callback.epochs, callback.val_accuracies, label="Validation Accuracy", color='green', marker='s')
    plt.title("Validation Accuracy Trend")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0.5, 1.0)  
    plt.legend()
    plt.grid(alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_metrics.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"损失和准确率曲线已保存到: {save_path}")
    
    try:
        plt.show()
    except:
        pass