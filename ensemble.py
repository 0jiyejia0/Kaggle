import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from scipy.stats import rankdata
import pandas as pd
import os

def calculate_weights(metrics_dict):
    """
    基于多个评估指标计算模型权重
    """
    weights = {}
    for model_name, metrics in metrics_dict.items():
        if metrics is not None:
            # 归一化各个指标
            rmse_norm = 1 / (metrics['rmse'] + 1e-6)  # 越小越好
            r2_norm = max(0, metrics['r2'])           # 越大越好
            mape_norm = 1 / (metrics['mape'] + 1e-6)  # 越小越好
            
            # 综合评分
            score = (0.4 * r2_norm + 0.3 * rmse_norm + 0.3 * mape_norm)
            weights[model_name] = score
        else:
            weights[model_name] = 1.0  # 基础权重
    
    # 归一化权重
    total = sum(weights.values())
    weights = {m: w/total for m, w in weights.items()}
    
    return weights

def weighted_average(predictions_list, weights_list):
    """
    加权平均融合
    """
    if not predictions_list or len(predictions_list) != len(weights_list):
        return None
    
    # 检查预测值是否有异常值
    predictions_array = np.array(predictions_list)
    weights_array = np.array(weights_list)
    
    # 计算每个模型预测值的z-score
    z_scores = np.abs((predictions_array - np.mean(predictions_array, axis=0)) / 
                      (np.std(predictions_array, axis=0) + 1e-6))
    
    # 对异常预测进行惩罚
    penalty = np.exp(-z_scores)
    adjusted_weights = weights_array.reshape(-1, 1) * penalty
    
    # 重新归一化权重
    adjusted_weights = adjusted_weights / np.sum(adjusted_weights, axis=0)
    
    # 计算加权平均
    weighted_pred = np.sum(predictions_array * adjusted_weights, axis=0)
    
    return weighted_pred

def blend_predictions(predictions_list, weights_list=None, method='weighted'):
    """
    综合融合函数，支持多种融合方法
    """
    if method == 'weighted':
        return weighted_average(predictions_list, weights_list)
    elif method == 'mean':
        return np.mean(predictions_list, axis=0)
    elif method == 'median':
        return np.median(predictions_list, axis=0)
    else:
        raise ValueError(f"Unknown blending method: {method}")

def stacking_predictions(oof_predictions_log: dict, test_predictions_log: dict, y_train_log, model_dir, random_state=42):
    """
    使用Stacking方法融合预测。
    
    参数:
    - oof_predictions_log: 字典，键为模型名，值为OOF预测 (对数尺度, np.array)
    - test_predictions_log: 字典，键为模型名，值为测试集预测 (对数尺度, np.array)
    - y_train_log: 训练集的目标变量 (对数转换后)
    - model_dir: 保存元模型的目录
    - random_state: 随机种子

    返回:
    - Stacking后的最终测试集预测
    """
    print("\n🔀 === 开始 Stacking 融合 ===")
    from sklearn.linear_model import LassoCV

    # 1. 创建元模型的训练集和测试集
    # 使用对数尺度的OOF预测作为元模型的训练特征
    meta_X_train = pd.DataFrame(oof_predictions_log)
    # 使用对数尺度的测试集预测作为元模型的测试特征
    meta_X_test = pd.DataFrame(test_predictions_log)

    print(f"元模型训练集形状: {meta_X_train.shape}")
    print(f"元模型测试集形状: {meta_X_test.shape}")

    # 2. 训练元模型 (LassoCV)
    # LassoCV通过交叉验证自动选择最佳的alpha正则化参数
    print("开始训练元模型 (LassoCV)...")
    meta_model = LassoCV(
        cv=5, 
        random_state=random_state, 
        n_jobs=-1,
        alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10] # 指定alpha范围
    )
    
    # LassoCV在对数转换后的目标上训练，以保持一致性
    meta_model.fit(meta_X_train, y_train_log)
    
    print(f"✅ 元模型训练完成。最佳Alpha: {meta_model.alpha_:.6f}")

    # 显示元模型学到的权重
    print("\n⚖️ 元模型权重 (Lasso Coefficients):")
    for name, coef in zip(meta_X_train.columns, meta_model.coef_):
        print(f"   🔸 {name}: {coef:.4f}")
        if abs(coef) < 1e-6:
            print(f"      (模型 {name} 的权重接近于0，可能被Lasso剔除)")

    # 3. 使用元模型进行最终预测
    print("使用元模型进行最终预测...")
    final_stacking_pred_log = meta_model.predict(meta_X_test)
    
    # 转换回原始尺度
    final_stacking_pred = np.expm1(final_stacking_pred_log)
    
    print("✅ Stacking 融合完成")
    
    # 保存元模型以备分析
    try:
        import joblib
        meta_model_path = os.path.join(model_dir, 'stacking_meta_model.pkl')
        joblib.dump(meta_model, meta_model_path)
        print(f"元模型已保存到: {meta_model_path}")
    except Exception as e:
        print(f"❌ 保存元模型时出错: {e}")
        
    return final_stacking_pred
