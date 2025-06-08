import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """Vẽ biểu đồ loss trong quá trình training"""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label='Train Loss (MSE)', color='#1e88e5', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss (MSE)', color='#e53935', linewidth=2)
    plt.title('Biểu đồ hội tụ Loss vs. Epoch', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_vs_epoch.png', dpi=300)
    plt.show()

def plot_model_comparison():
    """Vẽ biểu đồ so sánh các mô hình"""
    models = ['NCF', 'Matrix Factorization', 'Item-based CF', 'Popularity-based']
    precision = [0.42, 0.35, 0.30, 0.25]
    rmse = [0.85, 0.95, 1.02, 1.10]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(models))
    
    bars1 = ax1.bar(index, precision, bar_width, label='Precision@5', color='#1e88e5')
    ax2 = ax1.twinx()
    bars2 = ax2.bar(index + bar_width, rmse, bar_width, label='RMSE', color='#e53935', alpha=0.7)

    ax1.set_xlabel('Mô hình', fontsize=12)
    ax1.set_ylabel('Precision@5', fontsize=12, color='#1e88e5')
    ax2.set_ylabel('RMSE', fontsize=12, color='#e53935')
    ax1.set_title('Biểu đồ so sánh NCF với mô hình truyền thống', fontsize=14)
    ax1.set_xticks(index + bar_width / 2)
    ax1.set_xticklabels(models, rotation=15)
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show() 