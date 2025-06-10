import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import os
from datetime import datetime
import json
from model_evaluation import calculate_metrics

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

def load_predictions_from_experiment(experiment_dir):
    """从实验目录加载所有预测结果，包括加权和stacking融合。"""
    predictions = {}
    pred_dir = os.path.join(experiment_dir, "predictions")
    
    if not os.path.exists(pred_dir):
        print(f"❌ 预测目录不存在: {pred_dir}")
        return predictions

    print(f"📂 正在从 {pred_dir} 加载预测...")
    for file in os.listdir(pred_dir):
        file_path = os.path.join(pred_dir, file)
        if file.startswith('predictions_') and file.endswith('.csv'):
            model_name = file.replace('predictions_', '').replace('.csv', '')
            df = pd.read_csv(file_path)
            predictions[model_name] = df['SalePrice'].values
        elif file == 'submission_weighted.csv':
            df_final = pd.read_csv(file_path)
            predictions['ensemble_weighted'] = df_final['SalePrice'].values
        elif file == 'submission_stacking.csv':
            df_final = pd.read_csv(file_path)
            predictions['ensemble_stacking'] = df_final['SalePrice'].values

    for name, pred in predictions.items():
        print(f"✅ 已加载 '{name}' 预测结果: {len(pred)} 条")
    
    return predictions

def analyze_prediction_distribution(predictions, output_dir):
    """分析预测分布"""
    print("\n📊 === 预测分布分析 ===")
    
    ensemble_keys = [k for k in predictions.keys() if 'ensemble' in k]
    individual_keys = [k for k in predictions.keys() if 'ensemble' not in k]

    num_plots = 2 + len(ensemble_keys)
    fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6))
    
    # 1. 各模型预测分布对比
    ax1 = axes[0]
    for model_name in individual_keys:
        sns.kdeplot(predictions[model_name], ax=ax1, label=model_name, fill=True, alpha=0.2)
    ax1.set_title('各基模型预测分布对比 (KDE)')
    ax1.legend()
    
    # 2. 模型相关性
    ax2 = axes[1]
    pred_matrix = np.array([predictions[name] for name in individual_keys]).T
    corr_matrix = pd.DataFrame(pred_matrix, columns=individual_keys).corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax2, fmt=".3f")
    ax2.set_title('基模型预测相关性')

    # 3. 各融合模型分布
    for i, key in enumerate(ensemble_keys):
        ax = axes[2 + i]
        sns.kdeplot(predictions[key], ax=ax, label=key, fill=True, color='red')
        ax.set_title(f'{key} 预测分布')
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prediction_analysis.png"), dpi=200)
    plt.close()

    # 统计表
    stats_data = []
    for model_name, pred in predictions.items():
        stats = {
            '模型': model_name,
            '均值': f"{pred.mean():.0f}", '中位数': f"{np.median(pred):.0f}",
            '标准差': f"{pred.std():.0f}", '最小值': f"{pred.min():.0f}",
            '最大值': f"{pred.max():.0f}", '偏度': f"{pd.Series(pred).skew():.3f}"
        }
        stats_data.append(stats)
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(os.path.join(output_dir, "prediction_statistics.csv"), index=False)
    print(f"✅ 预测分析图和统计表已保存。")
    return stats_df

def analyze_price_segments(predictions, output_dir):
    """分析不同价格段的预测表现"""
    print("\n🏘️ === 价格段分析 ===")
    
    # 定义价格段
    price_segments = {
        '低价位': (0, 150000),
        '中低价位': (150000, 200000),
        '中价位': (200000, 300000),
        '中高价位': (300000, 400000),
        '高价位': (400000, float('inf'))
    }
    
    segment_analysis = []
    
    for segment_name, (min_price, max_price) in price_segments.items():
        segment_stats = {'价格段': segment_name, '价格范围': f"{min_price:,} - {max_price:,}" if max_price != float('inf') else f"{min_price:,}+"}
        
        for model_name, pred in predictions.items():
            if model_name != 'ensemble':
                mask = (pred >= min_price) & (pred < max_price)
                count = mask.sum()
                percentage = count / len(pred) * 100
                avg_price = pred[mask].mean() if count > 0 else 0
                
                segment_stats[f'{model_name}_数量'] = count
                segment_stats[f'{model_name}_占比'] = f"{percentage:.1f}%"
                segment_stats[f'{model_name}_均价'] = f"{avg_price:.0f}"
        
        segment_analysis.append(segment_stats)
        print(f"🔸 {segment_name} ({min_price:,}-{max_price:,}): {sum((predictions[name] >= min_price) & (predictions[name] < max_price) for name in predictions if name != 'ensemble')} 个预测")
    
    # 保存价格段分析
    segment_df = pd.DataFrame(segment_analysis)
    segment_df.to_csv(f"{output_dir}/price_segment_analysis.csv", index=False)
    print(f"✅ 价格段分析已保存到: {output_dir}/price_segment_analysis.csv")
    
    return segment_df

def create_performance_report(experiment_dir):
    """为主实验目录创建性能报告"""
    print("📋 === 创建性能报告 ===")
    
    report_dir = os.path.join(experiment_dir, "performance_report")
    os.makedirs(report_dir, exist_ok=True)
    
    predictions = load_predictions_from_experiment(experiment_dir)
    
    if not predictions:
        print("❌ 警告: 未找到任何预测结果，无法生成报告。")
        return False

    stats_df = analyze_prediction_distribution(predictions, report_dir)
    
    ensemble_keys = [k for k in predictions.keys() if 'ensemble' in k]
    best_ensemble = ""
    if 'ensemble_stacking' in ensemble_keys:
        best_ensemble = 'ensemble_stacking'
    elif 'ensemble_weighted' in ensemble_keys:
        best_ensemble = 'ensemble_weighted'
    
    report_text = f"""# 模型性能评估报告

## 🎯 实验信息
- **实验目录**: `{experiment_dir}`
- **模型数量**: {len([k for k in predictions.keys() if 'ensemble' not in k])}
- **预测样本数**: {len(list(predictions.values())[0])}

## 📊 模型预测统计
{stats_df.to_markdown(index=False)}

## 🎉 结论
"""
    if best_ensemble:
        report_text += f"- **最终融合方法**: `{best_ensemble}`\n"
        report_text += f"- **预测范围**: `${min(predictions[best_ensemble]):,.0f} - ${max(predictions[best_ensemble]):,.0f}`\n"
        report_text += f"- **平均预测价格**: `${np.mean(predictions[best_ensemble]):,.0f}`\n"
    else:
        report_text += "- 未找到融合模型的预测结果。\n"

    report_text += "\n## 📈 可视化分析\n"
    report_text += "![预测分析图](prediction_analysis.png)\n"
    
    report_file_path = os.path.join(report_dir, "performance_report.md")
    with open(report_file_path, "w", encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ 完整性能报告已保存到: {report_file_path}")
    return True

def main(experiment_dir=None):
    """
    为指定的实验目录生成报告。
    如果未提供目录，则尝试自动查找最新的目录。
    """
    print("🎯 === 模型性能全面评估 ===")
    
    if experiment_dir is None:
        print("-- 未提供实验目录，尝试自动查找最新目录 --")
        experiments_root = "experiments"
        if os.path.exists(experiments_root):
            all_dirs = [d for d in os.listdir(experiments_root) if os.path.isdir(os.path.join(experiments_root, d)) and d != 'eda_plots']
            if all_dirs:
                experiment_dir = os.path.join(experiments_root, max(all_dirs))
                print(f"INFO: 自动选择最新实验目录: '{experiment_dir}'")
            else:
                print(f"❌ 错误: 在 '{experiments_root}' 目录下没有找到有效的实验子目录。")
                return False
        else:
            print(f"❌ 错误: 'experiments' 目录不存在。")
            return False

    if experiment_dir:
        success = create_performance_report(experiment_dir)
        if success:
            print("\n🎉 === 报告生成成功 ===")
        else:
            print("\n❌ === 报告生成失败 ===")
        return success
    return False

if __name__ == "__main__":
    # 作为一个独立脚本运行时，可以手动指定目录或让它自动查找
    # import sys
    # if len(sys.argv) > 1:
    #     main(sys.argv[1])
    # else:
    #     main()
    main() 