import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import KFold
from scipy.stats import rankdata

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

def rank_average(predictions_list, weights_list=None):
    """
    排序平均融合
    """
    if not predictions_list:
        return None
        
    # 将每个模型的预测转换为排名
    ranks_list = [rankdata(pred) for pred in predictions_list]
    ranks_array = np.array(ranks_list)
    
    if weights_list is not None:
        weights_array = np.array(weights_list).reshape(-1, 1)
        # 计算加权平均排名
        average_ranks = np.sum(ranks_array * weights_array, axis=0)
    else:
        # 计算简单平均排名
        average_ranks = np.mean(ranks_array, axis=0)
    
    # 将平均排名转换回原始尺度
    final_pred = np.mean(predictions_list, axis=0)
    final_pred[np.argsort(average_ranks)] = np.sort(final_pred)
    
    return final_pred

def stack_ensemble(train_preds, train_y, test_preds, final_estimator=None):
    """
    使用stacking方法融合模型
    """
    if final_estimator is None:
        final_estimator = Ridge(alpha=1.0, random_state=42)
    
    # 使用5折交叉验证生成meta特征
    n_folds = 5
    n_train = train_preds.shape[0]
    n_test = test_preds.shape[0]
    n_models = train_preds.shape[1]
    
    # 初始化存储数组
    meta_train = np.zeros((n_train,))
    meta_test = np.zeros((n_test,))
    
    # 使用K折交叉验证训练meta模型
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(train_preds):
        # 训练集分割
        X_train = train_preds[train_idx]
        y_train = train_y[train_idx]
        X_val = train_preds[val_idx]
        
        # 训练meta模型
        meta_model = Ridge(alpha=1.0, random_state=42)
        meta_model.fit(X_train, y_train)
        
        # 生成验证集和测试集预测
        meta_train[val_idx] = meta_model.predict(X_val)
        meta_test += meta_model.predict(test_preds) / n_folds
    
    # 训练最终meta模型
    final_estimator.fit(train_preds, train_y)
    final_pred = final_estimator.predict(test_preds)
    
    # 将stacking预测与原始模型预测结合
    final_pred = 0.7 * final_pred + 0.3 * meta_test
    
    return final_estimator, final_pred

def blend_predictions(predictions_list, weights_list=None, method='weighted'):
    """
    综合融合函数，支持多种融合方法
    """
    if method == 'weighted':
        return weighted_average(predictions_list, weights_list)
    elif method == 'rank':
        return rank_average(predictions_list, weights_list)
    elif method == 'mean':
        return np.mean(predictions_list, axis=0)
    elif method == 'median':
        return np.median(predictions_list, axis=0)
    else:
        raise ValueError(f"Unknown blending method: {method}")
