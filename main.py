import pandas as pd
import numpy as np
from data_preprocessing import load_and_preprocess_data
from feature_engineering import add_features, select_features
from model_training import train_model_cv
from ensemble import calculate_weights, blend_predictions, stacking_predictions
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入报告生成器的主函数
from experiment_report_generator import main as generate_report # 或者导入 create_performance_report
from eda_analysis import generate_eda_plots

def setup_experiment_tracking():
    """设置实验跟踪"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"experiments/{timestamp}"
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(f"{experiment_dir}/models", exist_ok=True)
    os.makedirs(f"{experiment_dir}/plots", exist_ok=True)
    os.makedirs(f"{experiment_dir}/predictions", exist_ok=True)
    return experiment_dir

def save_experiment_config(config, experiment_dir):
    """保存实验配置"""
    with open(f"{experiment_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

def plot_feature_importance(feature_importance, experiment_dir):
    """绘制特征重要性图"""
    try:
        plt.figure(figsize=(12, 6))
        importance_df = pd.DataFrame(feature_importance.items(), columns=['Feature', 'Importance'])
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        sns.barplot(data=importance_df.head(20), x='Importance', y='Feature')
        plt.title('Top 20 Most Important Features')
        plt.tight_layout()
        plt.savefig(f"{experiment_dir}/plots/feature_importance.png")
        plt.close()
        print("✅ 特征重要性图已保存")
    except Exception as e:
        print(f"❌ 绘制特征重要性图时出错: {str(e)}")

def safe_model_training(model_name, X_train, y_train_log, y_train, X_test, experiment_dir, n_splits=3):
    """安全的模型训练，带有完整错误处理"""
    try:
        print(f"\n🚀 开始训练 {model_name} 模型...")
        model_output_dir = f"{experiment_dir}/models"
        result = train_model_cv(
            model_name=model_name,
            X=X_train,
            y_log=y_train_log,
            y=y_train,
            X_test=X_test,
            output_model_dir=model_output_dir,
            n_splits=n_splits
        )
        
        # 详细检查返回结果
        if result is None:
            print(f"❌ {model_name} 训练返回 None")
            return None
            
        if not isinstance(result, dict):
            print(f"❌ {model_name} 训练返回非字典类型: {type(result)}")
            return None
            
        # 检查必要的键
        required_keys = ["final_test_pred", "metrics"]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            print(f"❌ {model_name} 训练结果缺少键: {missing_keys}")
            return None
            
        # 检查预测结果
        if result.get("final_test_pred") is None:
            print(f"❌ {model_name} 预测结果为空")
            return None
            
        if result.get("metrics") is not None:
            metrics = result["metrics"]
            print(f"✅ {model_name} 训练成功! RMSE: {metrics.get('rmse', 'N/A'):.4f}, R²: {metrics.get('r2', 'N/A'):.4f}")
            return result
        else:
            print(f"⚠️ {model_name} 训练完成但缺少评估指标")
            return result
            
    except KeyboardInterrupt:
        print(f"\n⏹️ {model_name} 训练被用户中断")
        return None
    except Exception as e:
        print(f"❌ {model_name} 训练失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("🎯 === 开始最终版本的模型训练 ===")
    
    # 首先，执行探索性数据分析（如果图表不存在）
    generate_eda_plots()
    
    # 设置实验跟踪
    experiment_dir = setup_experiment_tracking()
    
    # 优化的配置参数
    config = {
        "random_seed": 42,
        "cv_folds": 5,  # <--- 修改为5折
        "use_gpu": False,  # 使用CPU确保稳定性
        "feature_selection": {
            "n_features": 30,  # 减少特征数量
            "method": "selectkbest"
        },
        "models": {
            "xgb": {
                "enabled": True,
                "params": {
                    "n_estimators": 1000,
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
                "enabled": True,  # <--- 启用TabNet
                "params": {
                    "n_d": 24,
                    "n_a": 24,
                    "n_steps": 5
                }
            }
        },
        "ensemble": {
            "methods": ["weighted", "stacking"],
            "weights_method": "auto"
        }
    }
    
    # 保存配置
    save_experiment_config(config, experiment_dir)
    
    print("\n📊 === 开始数据处理 ===")
    try:
        # 加载和预处理数据
        X_train, X_test, y_train, y_test, test_ids = load_and_preprocess_data()
        print(f"📈 初始数据形状: X_train={X_train.shape}, X_test={X_test.shape}")
        
        # 特征工程（使用修复版本）
        print("\n🔧 === 开始特征工程 ===")
        X_train, X_test = add_features(X_train, X_test)
        
        # 特征选择
        if config["feature_selection"]["method"] == "selectkbest":
            print("🎯 开始特征选择...")
            selected_features = select_features(X_train, y_train, k=config["feature_selection"]["n_features"])
            print(f"✅ 选择了 {len(selected_features)} 个特征")
            
            # 确保选择的特征在数据中存在
            available_features = [f for f in selected_features if f in X_train.columns]
            X_train = X_train[available_features]
            X_test = X_test[available_features]
            print(f"📋 最终使用 {len(available_features)} 个特征")
        
        # 对目标变量进行对数转换
        y_train_log = np.log1p(y_train)
        
        print(f"\n📊 处理后数据形状: X_train={X_train.shape}, X_test={X_test.shape}")
        
    except Exception as e:
        print(f"❌ 数据处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    # 训练各个模型
    print("\n🤖 === 开始模型训练 ===")
    models_results = {}
    test_predictions = {}
    model_metrics = {}
    oof_predictions = {}
    oof_predictions_log = {}
    test_predictions_log = {}
    
    enabled_models = [name for name, model_config in config["models"].items() if model_config["enabled"]]
    print(f"🎯 将训练以下模型: {enabled_models}")
    
    for i, model_name in enumerate(enabled_models, 1):
        print(f"\n🔄 进度: {i}/{len(enabled_models)} - 训练 {model_name}")
        result = safe_model_training(model_name, X_train, y_train_log, y_train, X_test, experiment_dir, config["cv_folds"])
        
        if result is not None:
            models_results[model_name] = result
            
            # 存储每个模型的预测结果
            if "final_test_pred" in result and result["final_test_pred"] is not None:
                test_predictions[model_name] = result["final_test_pred"]
                print(f"✅ {model_name} 预测结果已保存")

            # 存储OOF预测用于Stacking
            if "oof_preds" in result and result["oof_preds"] is not None:
                oof_predictions[model_name] = result["oof_preds"]
                print(f"✅ {model_name} OOF预测已保存")
            
            # 存储对数尺度的OOF和测试集预测，用于Stacking
            if "oof_preds_log" in result and result["oof_preds_log"] is not None:
                oof_predictions_log[model_name] = result["oof_preds_log"]
            if "test_preds_cv_log" in result and result["test_preds_cv_log"] is not None:
                test_predictions_log[model_name] = result["test_preds_cv_log"]

            # 存储每个模型的评估指标
            if "metrics" in result and result["metrics"] is not None:
                model_metrics[model_name] = result["metrics"]
                print(f"✅ {model_name} 评估指标已保存")
        else:
            print(f"❌ 跳过失败的模型: {model_name}")
    
    # 检查是否有成功训练的模型
    if len(test_predictions) == 0:
        print("❌ 没有模型成功训练，程序结束")
        return
    
    print(f"\n🎉 成功训练的模型: {list(test_predictions.keys())} (共 {len(test_predictions)} 个)")
    
    # 显示各模型性能
    print("\n📊 === 模型性能汇总 ===")
    for model_name, metrics in model_metrics.items():
        if metrics:
            print(f"🔸 {model_name}: RMSE={metrics.get('rmse', 'N/A'):.0f}, R²={metrics.get('r2', 'N/A'):.4f}")
    
    # 模型融合
    print("\n🔀 === 开始模型融合 ===")
    
    # 检查启用的融合方法
    ensemble_methods = config.get("ensemble", {}).get("methods", [])

    # --- 1. 加权平均融合 ---
    if "weighted" in ensemble_methods:
        try:
            print("\n--- 开始加权平均融合 ---")
            if len(model_metrics) > 0:
                # 获取各个模型的权重
                weights = calculate_weights(model_metrics)
                print("\n⚖️ 模型权重:")
                for model_name, weight in weights.items():
                    print(f"   🔸 {model_name}: {weight:.4f}")
            else:
                # 如果没有指标，使用等权重
                weights = {name: 1.0/len(test_predictions) for name in test_predictions.keys()}
                print("⚖️ 使用等权重融合")
            
            # 加权平均融合
            if len(test_predictions) > 1:
                ensemble_pred = blend_predictions(
                    list(test_predictions.values()),
                    list(weights.values()),
                    method="weighted"
                )
                print("✅ 多模型加权融合完成")
            else:
                # 如果只有一个模型，直接使用其预测
                ensemble_pred = list(test_predictions.values())[0]
                print("✅ 单模型预测")
            
            # 保存预测结果
            if ensemble_pred is not None:
                submission_df = pd.DataFrame({
                    'Id': test_ids,
                    'SalePrice': ensemble_pred
                })
                submission_path = f"{experiment_dir}/predictions/submission_weighted.csv"
                submission_df.to_csv(submission_path, index=False)
                print(f"✅ 加权平均预测结果已保存到: {submission_path}")
            
        except Exception as e:
            print(f"❌ 加权平均融合失败: {str(e)}")
            import traceback
            traceback.print_exc()

    # --- 2. Stacking 融合 ---
    if "stacking" in ensemble_methods:
        try:
            print("\n--- 开始Stacking融合 ---")
            if len(oof_predictions) < 2 or len(test_predictions) < 2:
                print("⚠️ Stacking需要至少2个模型，跳过此步骤。")
            else:
                # 确保OOF和测试预测的模型一致
                common_models = sorted(list(set(oof_predictions_log.keys()) & set(test_predictions_log.keys())))
                
                if len(common_models) < 2:
                    print("⚠️ Stacking所需模型的对数尺度预测不完整，跳过此步骤。")
                else:
                    oof_preds_aligned = {model: oof_predictions_log[model] for model in common_models}
                    test_preds_aligned = {model: test_predictions_log[model] for model in common_models}

                    print(f"用于Stacking的模型: {common_models}")

                    stacking_pred = stacking_predictions(
                        oof_predictions_log=oof_preds_aligned,
                        test_predictions_log=test_preds_aligned,
                        y_train_log=y_train_log,
                        model_dir=f"{experiment_dir}/models"
                    )
                
                    if stacking_pred is not None:
                        submission_df = pd.DataFrame({
                            'Id': test_ids,
                            'SalePrice': stacking_pred
                        })
                        submission_path = f"{experiment_dir}/predictions/submission_stacking.csv"
                        submission_df.to_csv(submission_path, index=False)
                        print(f"✅ Stacking预测结果已保存到: {submission_path}")

        except Exception as e:
            print(f"❌ Stacking融合失败: {str(e)}")
            import traceback
            traceback.print_exc()

    # 特征重要性分析
    try:
        if "xgb" in models_results and models_results["xgb"]["final_model"] is not None:
            feature_importance = dict(zip(
                X_train.columns,
                models_results["xgb"]["final_model"].feature_importances_
            ))
            plot_feature_importance(feature_importance, experiment_dir)
    except Exception as e:
        print(f"❌ 特征重要性分析失败: {str(e)}")
    
    print(f"\n🎉 === 实验完成 ===")
    print(f"📁 结果保存在: {experiment_dir}")
    print(f"🏆 成功训练的模型总数: {len(models_results)}")
    
    # 保存个别模型预测（用于分析）
    try:
        for model_name, pred in test_predictions.items():
            individual_path = f"{experiment_dir}/predictions/predictions_{model_name}.csv"
            pd.DataFrame({
                'Id': range(1, len(pred) + 1),
                'SalePrice': pred
            }).to_csv(individual_path, index=False)
            print(f"📄 {model_name} 单独预测已保存")
    except Exception as e:
        print(f"❌ 保存单独预测时出错: {str(e)}")
    
    # 最终总结
    print(f"\n🏁 === 训练总结 ===")
    print(f"✅ 成功模型数: {len(test_predictions)}")
    print(f"📊 目标模型数: {len(enabled_models)}")
    if len(test_predictions) == len(enabled_models):
        print("🎉 所有启用的模型都训练成功！")
    else:
        failed_models = set(enabled_models) - set(test_predictions.keys())
        print(f"⚠️ 失败的模型: {failed_models}")

    # 在实验结束后生成报告
    print(f"\n📋 === 开始生成实验报告 ===")
    try:
        success = generate_report(experiment_dir) # 调用报告生成器的主函数
        if success:
            print("✅ 实验报告已生成")
        else:
            print("⚠️ 实验报告生成过程中出现问题，请查看上方日志。")
    except Exception as e:
        print(f"❌ 生成实验报告时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 