import numpy as np
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from model_evaluation import rmse, mae, r2_score_custom
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error as mape
import lightgbm as lgb

def load_saved_model(model_name, model_dir):
    """
    加载已保存的模型。
    如果模型文件不存在，返回None。
    """
    model = None
    base_model_file = f"{model_name}_model"
    
    try:
        if model_name == "xgb":
            model_path = os.path.join(model_dir, f"{base_model_file}.json")
            if os.path.exists(model_path):
                model = XGBRegressor()
                model.load_model(model_path)
        elif model_name == "lgb":
            model_path = os.path.join(model_dir, f"{base_model_file}.txt")
            if os.path.exists(model_path):
                model = LGBMRegressor()
                model.booster_ = lgb.Booster(model_file=model_path) # 使用 lgb.Booster 加载
        elif model_name == "cat":
            model_path = os.path.join(model_dir, f"{base_model_file}.cbm")
            if os.path.exists(model_path):
                model = CatBoostRegressor()
                model.load_model(model_path)
        elif model_name == "tabnet":
            # TabNet保存时可能没有后缀，或者是一个 .zip 文件
            # 我们先尝试不带后缀，如果找不到，再尝试 .zip
            model_path_no_suffix = os.path.join(model_dir, base_model_file)
            model_path_zip = os.path.join(model_dir, f"{base_model_file}.zip")
            if os.path.exists(model_path_no_suffix):
                model = TabNetRegressor()
                model.load_model(model_path_no_suffix)
            elif os.path.exists(model_path_zip):
                model = TabNetRegressor()
                model.load_model(model_path_zip)      
    except Exception as e:
        print(f"加载模型 {model_name} 从 {model_dir} 时出错: {str(e)}")
        model = None
    
    return model

def save_feature_names(model_name, feature_names, model_dir):
    """保存特征名称到文件"""
    try:
        model_file_path = os.path.join(model_dir, f"{model_name}_model_features.txt")
        with open(model_file_path, "w") as f:
            f.write("\n".join(feature_names))
    except Exception as e:
        print(f"保存特征名称到 {model_dir} 时出错: {str(e)}")

def load_feature_names(model_name, model_dir):
    """加载特征名称"""
    try:
        model_file_path = os.path.join(model_dir, f"{model_name}_model_features.txt")
        if os.path.exists(model_file_path):
            with open(model_file_path, "r") as f:
                return f.read().splitlines()
        else:
            print(f"特征文件 {model_file_path} 未找到")
            return None
    except Exception as e:
        print(f"加载特征名称从 {model_dir} 时出错: {str(e)}")
        return None

def train_model_cv(model_name, X, y_log, y, X_test=None, output_model_dir=None, n_splits=5, use_saved_model=True, skip_cv=False):
    """
    使用K折交叉验证训练指定模型，并评估性能。
    """
    print(f"\n=== 开始处理模型: {model_name} ===")
    if output_model_dir:
        os.makedirs(output_model_dir, exist_ok=True)
        print(f"模型及特征将保存到: {output_model_dir}")
    else:
        print("警告: 未提供 output_model_dir，模型将保存在当前工作目录。")
        output_model_dir = "." # 默认为当前目录

    # 初始化返回值
    oof_preds = None
    cv_rmse = None
    cv_mae = None
    cv_r2 = None
    cv_mape = None
    final_model = None
    final_test_pred = None
    
    # 保存特征名称（在开始时就保存，确保所有模型都有）
    feature_names = X.columns.tolist()
    save_feature_names(model_name, feature_names, output_model_dir)
    
    # 如果不跳过CV，执行完整的交叉验证评估
    if not skip_cv:
        print("执行交叉验证评估...")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        n_samples = X.shape[0]
        oof_preds_log = np.zeros(n_samples)
        test_preds_accum_log = np.zeros(X_test.shape[0]) if X_test is not None else None

        fold_metrics = {"rmse": [], "mae": [], "r2": [], "mape": []}

        # 遍历每一折进行训练
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            print(f"\n开始训练第 {fold} 折...")
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y_log[train_idx]
            y_val_fold = y_log[val_idx]

            # 根据模型名称初始化模型并设置参数
            if model_name == "xgb":
                model = XGBRegressor(
                    n_estimators=1000,  # 减少以加快训练
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    tree_method='hist',
                    device='cpu',  # 使用CPU避免兼容性问题
                    enable_categorical=True,
                    early_stopping_rounds=50
                )
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                val_preds_log = model.predict(X_val_fold)
                
            elif model_name == "lgb":
                model = LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=6,
                    min_child_samples=10,
                    min_split_gain=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.01,
                    reg_lambda=0.01,
                    random_state=42,
                    verbose=-1
                )
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                val_preds_log = model.predict(X_val_fold)
                
            elif model_name == "cat":
                model = CatBoostRegressor(
                    iterations=500,  # 减少迭代数
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3,
                    loss_function="RMSE",
                    verbose=False,
                    task_type="CPU",
                    random_seed=42,
                    bootstrap_type='Bernoulli',
                    subsample=0.8,
                    allow_writing_files=False
                )
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    early_stopping_rounds=50,
                    verbose=False
                )
                val_preds_log = model.predict(X_val_fold)
                
            elif model_name == "tabnet":
                model = TabNetRegressor(
                    n_d=24, n_a=24,
                    n_steps=5,
                    gamma=1.5,
                    n_independent=2,
                    n_shared=2,
                    mask_type="sparsemax",
                    lambda_sparse=1e-3,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    scheduler_params=dict(
                        mode="min",
                        patience=5,
                        min_lr=1e-5,
                        factor=0.5,
                    ),
                    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    device_name="cpu"
                )
                
                X_train_arr = X_train_fold.values.astype(np.float32)
                X_val_arr = X_val_fold.values.astype(np.float32)
                y_train_arr = y_train_fold.to_numpy().astype(np.float32).reshape(-1, 1)
                y_val_arr = y_val_fold.to_numpy().astype(np.float32).reshape(-1, 1)
                
                model.fit(
                    X_train_arr, y_train_arr,
                    eval_set=[(X_val_arr, y_val_arr)],
                    max_epochs=1000,
                    patience=50,
                    batch_size=1024,
                    virtual_batch_size=128
                )
                val_preds_log = model.predict(X_val_arr).squeeze()
            else:
                raise ValueError(f"Unknown model name: {model_name}")

            # 存储OOF预测结果
            oof_preds_log[val_idx] = val_preds_log

            # 将预测转换回原始尺度以计算评价指标
            val_preds = np.expm1(val_preds_log)
            y_val_actual = np.expm1(y_val_fold)
            
            # 计算该折的性能指标
            rmse_val = rmse(y_val_actual, val_preds)
            mae_val = mae(y_val_actual, val_preds)
            r2_val = r2_score_custom(y_val_actual, val_preds)
            mape_val = mape(y_val_actual, val_preds)
            
            fold_metrics["rmse"].append(rmse_val)
            fold_metrics["mae"].append(mae_val)
            fold_metrics["r2"].append(r2_val)
            fold_metrics["mape"].append(mape_val)
            
            print(f"[{model_name} Fold {fold}] RMSE: {rmse_val:.4f}, MAE: {mae_val:.4f}, R²: {r2_val:.4f}, MAPE: {mape_val:.4f}")

            # 如果提供了测试集，累积测试集预测
            if X_test is not None:
                if model_name == "tabnet":
                    X_test_arr = X_test.values.astype(np.float32)
                    fold_test_pred = model.predict(X_test_arr).squeeze()
                else:
                    fold_test_pred = model.predict(X_test)
                if test_preds_accum_log is not None:
                    test_preds_accum_log += fold_test_pred / n_splits

        # 计算交叉验证整体性能
        oof_preds = np.expm1(oof_preds_log)
        y_actual = np.expm1(y_log)
        cv_rmse = rmse(y_actual, oof_preds)
        cv_mae = mae(y_actual, oof_preds)
        cv_r2 = r2_score_custom(y_actual, oof_preds)
        cv_mape = mape(y_actual, oof_preds)
        
        print(f"\n[{model_name} CV] RMSE: {cv_rmse:.4f}, MAE: {cv_mae:.4f}, R²: {cv_r2:.4f}, MAPE: {cv_mape:.4f}")
    
    # 如果需要进行预测
    if X_test is not None:
        # 尝试加载已保存的模型
        if use_saved_model:
            final_model = load_saved_model(model_name, output_model_dir)
            if final_model is not None:
                print(f"成功加载已保存的 {model_name} 模型从 {output_model_dir}")
                
                # 加载特征名称
                saved_feature_names = load_feature_names(model_name, output_model_dir)
                if saved_feature_names is not None:
                    try:
                        # 确保测试集的特征顺序与训练时一致
                        X_test_aligned = X_test[saved_feature_names]
                        test_preds_final_log = final_model.predict(X_test_aligned)
                        final_test_pred = np.expm1(test_preds_final_log)
                        print("已完成测试集预测")
                    except Exception as e:
                        print(f"使用保存的模型预测时出错: {str(e)}")
                        final_model = None  # 重置，强制重新训练
                else:
                    print("特征名称文件未找到，将重新训练模型")
                    final_model = None
            else:
                print(f"未找到已保存的 {model_name} 模型，需要重新训练...")
        
        # 如果没有找到已保存的模型或不使用已保存模型，则训练新模型
        if final_model is None:
            print("开始训练新模型...")
            if model_name == "xgb":
                final_model = XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    min_child_weight=1,
                    gamma=0,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    tree_method='hist',
                    device='cpu',
                    enable_categorical=True
                )
            elif model_name == "lgb":
                final_model = LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    max_depth=6,
                    min_child_samples=10,
                    min_split_gain=0.01,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.01,
                    reg_lambda=0.01,
                    random_state=42,
                    verbose=-1
                )
            elif model_name == "cat":
                final_model = CatBoostRegressor(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    l2_leaf_reg=3,
                    loss_function="RMSE",
                    verbose=False,
                    task_type="CPU",
                    random_seed=42,
                    bootstrap_type='Bernoulli',
                    subsample=0.8,
                    allow_writing_files=False
                )
            elif model_name == "tabnet":
                final_model = TabNetRegressor(
                    n_d=24, n_a=24,
                    n_steps=5,
                    gamma=1.5,
                    n_independent=2,
                    n_shared=2,
                    mask_type="sparsemax",
                    lambda_sparse=1e-3,
                    optimizer_fn=torch.optim.Adam,
                    optimizer_params=dict(lr=2e-2),
                    scheduler_params=dict(
                        mode="min",
                        patience=5,
                        min_lr=1e-5,
                        factor=0.5,
                    ),
                    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                    device_name="cpu"
                )

            # 训练最终模型
            if model_name == "tabnet":
                X_arr = X.values.astype(np.float32)
                y_arr = y_log.to_numpy().astype(np.float32).reshape(-1, 1)
                final_model.fit(
                    X_arr, y_arr,
                    max_epochs=1000,
                    patience=50,
                    batch_size=1024,
                    virtual_batch_size=128
                )
            else:
                final_model.fit(X, y_log)

            # 保存模型
            base_model_file = f"{model_name}_model"
            try:
                if model_name == "xgb":
                    final_model.save_model(os.path.join(output_model_dir, f"{base_model_file}.json"))
                elif model_name == "lgb":
                    final_model.booster_.save_model(os.path.join(output_model_dir, f"{base_model_file}.txt"))
                elif model_name == "cat":
                    final_model.save_model(os.path.join(output_model_dir, f"{base_model_file}.cbm"))
                elif model_name == "tabnet":
                    # TabNet通常保存为 .zip 文件，或者不带后缀的目录
                    final_model.save_model(os.path.join(output_model_dir, base_model_file)) 
                print(f"Model saved to: {output_model_dir}")
            except Exception as e:
                print(f"保存模型时出错: {str(e)}")

            # 对测试集进行最终预测
            try:
                if model_name == "tabnet":
                    X_test_arr = X_test.values.astype(np.float32)
                    test_preds_final_log = final_model.predict(X_test_arr).squeeze()
                else:
                    test_preds_final_log = final_model.predict(X_test)
                final_test_pred = np.expm1(test_preds_final_log)
                print("已完成测试集预测")
            except Exception as e:
                print(f"预测时出错: {str(e)}")
                final_test_pred = None

    # 汇总结果
    if cv_rmse is not None:
        print(f"\nFinished {model_name}. CV Metrics -> RMSE: {cv_rmse:.4f}, MAE: {cv_mae:.4f}, R²: {cv_r2:.4f}, MAPE: {cv_mape:.4f}")
    else:
        print(f"\nFinished {model_name}. 使用已保存模型，未进行交叉验证评估。")
    
    return {
        "oof_preds": oof_preds,                   # 训练集上的OOF预测（原始尺度）
        "oof_preds_log": oof_preds_log,           # 训练集上的OOF预测（对数尺度）
        "test_preds_cv_log": test_preds_accum_log,# CV期间的测试集预测平均值（对数尺度）
        "final_model": final_model,               # 全量训练的最终模型
        "final_test_pred": final_test_pred,       # 测试集最终预测（原始尺度）
        "metrics": {
            "rmse": cv_rmse,
            "mae": cv_mae,
            "r2": cv_r2,
            "mape": cv_mape
        } if cv_rmse is not None else None
    } 