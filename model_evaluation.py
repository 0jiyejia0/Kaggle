import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def rmse(y_true, y_pred):
    """计算RMSE"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae(y_true, y_pred):
    """计算MAE"""
    return mean_absolute_error(y_true, y_pred)

def mape(y_true, y_pred):
    """计算MAPE"""
    return mean_absolute_percentage_error(y_true, y_pred)

def r2_score_custom(y_true, y_pred):
    """计算R²分数"""
    return r2_score(y_true, y_pred)

def calculate_metrics(y_true, y_pred):
    """计算各种评估指标"""
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_val = mean_absolute_error(y_true, y_pred)
    r2_val = r2_score(y_true, y_pred)
    mape_val = mean_absolute_percentage_error(y_true, y_pred)
    
    # 自定义指标
    median_ae_val = np.median(np.abs(y_true - y_pred))
    max_error_val = np.max(np.abs(y_true - y_pred))
    
    return {
        'rmse': rmse_val,
        'mae': mae_val,
        'r2': r2_val,
        'mape': mape_val,
        'median_ae': median_ae_val,
        'max_error': max_error_val
    }

def plot_prediction_error(y_true, y_pred, title='预测误差分析'):
    """绘制预测误差分析图"""
    plt.figure(figsize=(15, 5))
    
    # 预测值vs实际值散点图
    plt.subplot(131)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title('预测值 vs 实际值')
    
    # 误差分布直方图
    plt.subplot(132)
    errors = y_pred - y_true
    sns.histplot(errors, kde=True)
    plt.xlabel('预测误差')
    plt.ylabel('频数')
    plt.title('误差分布')
    
    # QQ图
    plt.subplot(133)
    stats.probplot(errors, dist="norm", plot=plt)
    plt.title('Q-Q图')
    
    plt.tight_layout()
    plt.savefig('prediction_error_analysis.png')
    plt.close()

def plot_residuals(y_true, y_pred, title='残差分析'):
    """绘制残差分析图"""
    plt.figure(figsize=(15, 5))
    
    # 残差vs预测值
    plt.subplot(131)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    plt.title('残差 vs 预测值')
    
    # 标准化残差
    plt.subplot(132)
    standardized_residuals = residuals / np.std(residuals)
    plt.scatter(y_pred, standardized_residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.axhline(y=2, color='g', linestyle='--')
    plt.axhline(y=-2, color='g', linestyle='--')
    plt.xlabel('预测值')
    plt.ylabel('标准化残差')
    plt.title('标准化残差')
    
    # 残差的核密度估计
    plt.subplot(133)
    sns.kdeplot(residuals)
    plt.xlabel('残差')
    plt.ylabel('密度')
    plt.title('残差分布')
    
    plt.tight_layout()
    plt.savefig('residuals_analysis.png')
    plt.close()

def calculate_confidence_intervals(y_true, y_pred, confidence=0.95):
    """计算预测的置信区间"""
    # 计算预测误差
    errors = y_true - y_pred
    
    # 计算标准误差
    std_error = np.std(errors)
    
    # 计算置信区间
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin_of_error = z_score * std_error
    
    return {
        'mean_error': np.mean(errors),
        'std_error': std_error,
        'confidence_interval': margin_of_error,
        'lower_bound': y_pred - margin_of_error,
        'upper_bound': y_pred + margin_of_error
    }

def evaluate_predictions(y_true, y_pred, model_name='Model'):
    """综合评估函数"""
    # 计算基本指标
    metrics = calculate_metrics(y_true, y_pred)
    
    # 计算置信区间
    ci = calculate_confidence_intervals(y_true, y_pred)
    
    # 生成可视化
    plot_prediction_error(y_true, y_pred, title=f'{model_name} 预测误差分析')
    plot_residuals(y_true, y_pred, title=f'{model_name} 残差分析')
    
    # 打印评估结果
    print(f"\n=== {model_name} 评估结果 ===")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"MAPE: {metrics['mape']:.4f}")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"\n95%置信区间: ±{ci['confidence_interval']:.4f}")
    print(f"平均误差: {ci['mean_error']:.4f}")
    print(f"标准误差: {ci['std_error']:.4f}")
    
    return {
        'metrics': metrics,
        'confidence_intervals': ci
    }
