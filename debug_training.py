import pandas as pd
import numpy as np
from data_preprocessing import load_and_preprocess_data
from feature_engineering import add_features, select_features
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

def diagnose_data_quality(X, y):
    """诊断数据质量"""
    print("=== 数据质量诊断 ===")
    print(f"数据形状: {X.shape}")
    print(f"缺失值总数: {X.isnull().sum().sum()}")
    print(f"目标变量统计:")
    print(y.describe())
    
    # 检查异常值
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    outliers = y[(y < Q1 - 1.5 * IQR) | (y > Q3 + 1.5 * IQR)]
    print(f"异常值数量: {len(outliers)} ({len(outliers)/len(y)*100:.2f}%)")
    
    # 检查特征分布
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    print(f"数值特征数量: {len(numeric_cols)}")
    print(f"类别特征数量: {len(X.columns) - len(numeric_cols)}")
    
    return True

def test_lightgbm_params(X, y):
    """测试不同的LightGBM参数组合"""
    print("\n=== 测试 LightGBM 参数 ===")
    
    # 分割小样本用于快速测试
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y[X_sample.index]
    
    param_configs = [
        {
            "name": "保守参数",
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_child_samples": 20,
                "min_split_gain": 0.0,
                "verbose": -1
            }
        },
        {
            "name": "优化参数",
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 10,
                "min_split_gain": 0.01,
                "reg_alpha": 0.01,
                "reg_lambda": 0.01,
                "verbose": -1
            }
        }
    ]
    
    for config in param_configs:
        print(f"\n测试 {config['name']}:")
        try:
            model = LGBMRegressor(**config['params'])
            model.fit(X_sample, y_sample)
            pred = model.predict(X_sample)
            score = model.score(X_sample, y_sample)
            print(f"  训练成功，R² 分数: {score:.4f}")
        except Exception as e:
            print(f"  训练失败: {str(e)}")

def test_catboost_params(X, y):
    """测试不同的CatBoost参数组合"""
    print("\n=== 测试 CatBoost 参数 ===")
    
    # 分割小样本用于快速测试
    sample_size = min(1000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y[X_sample.index]
    
    param_configs = [
        {
            "name": "CPU模式",
            "params": {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "task_type": "CPU",
                "verbose": False,
                "allow_writing_files": False
            }
        },
        {
            "name": "优化CPU模式",
            "params": {
                "iterations": 100,
                "learning_rate": 0.05,
                "depth": 6,
                "task_type": "CPU",
                "verbose": False,
                "bootstrap_type": 'Bernoulli',
                "subsample": 0.8,
                "allow_writing_files": False
            }
        }
    ]
    
    for config in param_configs:
        print(f"\n测试 {config['name']}:")
        try:
            model = CatBoostRegressor(**config['params'])
            model.fit(X_sample, y_sample)
            pred = model.predict(X_sample)
            score = model.score(X_sample, y_sample)
            print(f"  训练成功，R² 分数: {score:.4f}")
        except Exception as e:
            print(f"  训练失败: {str(e)}")

def create_optimized_config():
    """创建优化后的配置"""
    config = {
        "random_seed": 42,
        "cv_folds": 3,  # 减少折数以加快调试
        "use_gpu": False,  # 暂时禁用GPU
        "feature_selection": {
            "n_features": 30,  # 减少特征数量
            "method": "selectkbest"
        },
        "models": {
            "xgb": {
                "enabled": True,
                "params": {
                    "n_estimators": 1000,  # 减少树的数量
                    "learning_rate": 0.05,
                    "max_depth": 6
                }
            },
            "lgb": {
                "enabled": True,
                "params": {
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_child_samples": 10,
                    "min_split_gain": 0.01,
                    "reg_alpha": 0.01,
                    "reg_lambda": 0.01
                }
            },
            "cat": {
                "enabled": True,
                "params": {
                    "iterations": 500,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "task_type": "CPU",
                    "bootstrap_type": "Bernoulli",
                    "subsample": 0.8
                }
            },
            "tabnet": {
                "enabled": False,  # 暂时禁用TabNet
                "params": {
                    "n_d": 24,
                    "n_a": 24,
                    "n_steps": 5
                }
            }
        },
        "ensemble": {
            "methods": ["weighted"],  # 只使用加权平均
            "weights_method": "auto"
        }
    }
    
    # 保存到文件
    import json
    with open("optimized_config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    print("\n=== 优化配置已保存到 optimized_config.json ===")
    return config

def main():
    print("=== 开始诊断训练问题 ===")
    
    # 加载数据
    print("加载数据...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 特征工程
    print("特征工程...")
    X_train, X_test = add_features(X_train, X_test)
    
    # 诊断数据质量
    diagnose_data_quality(X_train, y_train)
    
    # 对目标变量进行对数转换
    y_train_log = np.log1p(y_train)
    
    # 测试模型参数
    test_lightgbm_params(X_train, y_train_log)
    test_catboost_params(X_train, y_train_log)
    
    # 创建优化配置
    config = create_optimized_config()
    
    print("\n=== 诊断完成 ===")
    print("建议:")
    print("1. 使用优化后的参数配置")
    print("2. 从CPU模式开始，确保稳定性")
    print("3. 逐步增加模型复杂度")
    print("4. 监控内存使用情况")

if __name__ == "__main__":
    main() 