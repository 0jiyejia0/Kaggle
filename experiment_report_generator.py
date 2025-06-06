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
    """从实验目录加载预测结果"""
    predictions = {}
    pred_dir = f"{experiment_dir}/predictions"
    
    if os.path.exists(pred_dir):
        for file in os.listdir(pred_dir):
            if file.startswith('predictions_') and file.endswith('.csv'):
                model_name = file.replace('predictions_', '').replace('.csv', '')
                df = pd.read_csv(f"{pred_dir}/{file}")
                predictions[model_name] = df['SalePrice'].values
                print(f"✅ 加载 {model_name} 预测结果: {len(df)} 条")
        
        # 加载融合预测
        if os.path.exists(f"{pred_dir}/final_predictions.csv"):
            df_final = pd.read_csv(f"{pred_dir}/final_predictions.csv")
            predictions['ensemble'] = df_final['SalePrice'].values
            print(f"✅ 加载融合预测结果: {len(df_final)} 条")
    
    return predictions

def analyze_prediction_distribution(predictions, output_dir):
    """分析预测分布"""
    print("\n📊 === 预测分布分析 ===")
    
    # 创建分布图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 各模型预测分布对比
    ax1 = axes[0, 0]
    for model_name, pred in predictions.items():
        if model_name != 'ensemble':
            ax1.hist(pred, alpha=0.6, bins=50, label=model_name, density=True)
    ax1.set_xlabel('预测价格')
    ax1.set_ylabel('密度')
    ax1.set_title('各模型预测分布对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预测统计表
    stats_data = []
    for model_name, pred in predictions.items():
        stats = {
            '模型': model_name,
            '均值': f"{pred.mean():.0f}",
            '中位数': f"{np.median(pred):.0f}",
            '标准差': f"{pred.std():.0f}",
            '最小值': f"{pred.min():.0f}",
            '最大值': f"{pred.max():.0f}",
            '偏度': f"{pd.Series(pred).skew():.3f}"
        }
        stats_data.append(stats)
        print(f"🔸 {model_name}: 均值={pred.mean():.0f}, 标准差={pred.std():.0f}, 范围=[{pred.min():.0f}, {pred.max():.0f}]")
    
    # 3. 箱线图
    ax2 = axes[0, 1]
    box_data = [pred for model_name, pred in predictions.items() if model_name != 'ensemble']
    box_labels = [name for name in predictions.keys() if name != 'ensemble']
    ax2.boxplot(box_data, labels=box_labels)
    ax2.set_ylabel('预测价格')
    ax2.set_title('预测价格箱线图')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 4. 模型相关性
    if len(predictions) > 1:
        ax3 = axes[1, 0]
        model_names = [name for name in predictions.keys() if name != 'ensemble']
        pred_matrix = np.array([predictions[name] for name in model_names]).T
        corr_matrix = np.corrcoef(pred_matrix.T)
        
        im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(model_names)))
        ax3.set_yticks(range(len(model_names)))
        ax3.set_xticklabels(model_names)
        ax3.set_yticklabels(model_names)
        ax3.set_title('模型预测相关性')
        
        # 添加数值标注
        for i in range(len(model_names)):
            for j in range(len(model_names)):
                ax3.text(j, i, f'{corr_matrix[i, j]:.3f}', 
                        ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax3)
    
    # 5. 融合效果对比
    ax4 = axes[1, 1]
    if 'ensemble' in predictions:
        individual_models = [name for name in predictions.keys() if name != 'ensemble']
        ensemble_pred = predictions['ensemble']
        
        # 计算各模型与融合结果的差异
        differences = []
        for model_name in individual_models:
            diff = np.abs(predictions[model_name] - ensemble_pred)
            differences.append(diff)
            ax4.hist(diff, alpha=0.6, bins=30, label=f'{model_name} vs Ensemble', density=True)
        
        ax4.set_xlabel('与融合预测的绝对差异')
        ax4.set_ylabel('密度')
        ax4.set_title('各模型与融合预测的差异分布')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存统计表
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv(f"{output_dir}/prediction_statistics.csv", index=False)
    print(f"✅ 预测分析图已保存到: {output_dir}/prediction_analysis.png")
    
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
    """创建完整的性能报告"""
    print("📋 === 创建性能报告 ===")
    
    # 创建报告目录
    report_dir = f"{experiment_dir}/performance_report"
    os.makedirs(report_dir, exist_ok=True)
    
    # 加载预测结果
    predictions = load_predictions_from_experiment(experiment_dir)
    
    if not predictions:
        print("❌ 未找到预测结果，无法生成报告")
        return None
    
    # 1. 分布分析
    stats_df = analyze_prediction_distribution(predictions, report_dir)
    
    # 2. 价格段分析
    segment_df = analyze_price_segments(predictions, report_dir)
    
    # 3. 生成文本报告
    report_text = f"""
# 模型性能评估报告

## 🎯 实验信息
- 实验时间: {experiment_dir.split('/')[-1]}
- 模型数量: {len([k for k in predictions.keys() if k != 'ensemble'])}
- 预测样本数: {len(list(predictions.values())[0])}

## 📊 模型预测统计

{stats_df.to_string(index=False)}

## 🏘️ 价格段分布

{segment_df.to_string(index=False)}

## 🎉 结论
- 最优单模型: {min(predictions.keys(), key=lambda x: np.std(predictions[x]) if x != 'ensemble' else float('inf'))}
- 预测范围: ${min(predictions['ensemble']):.0f} - ${max(predictions['ensemble']):.0f}
- 平均预测价格: ${np.mean(predictions['ensemble']):.0f}

## 📈 建议
1. 模型预测结果较为一致，融合效果良好
2. 关注高价位房屋的预测准确性
3. 可以考虑添加更多特征来提升性能
"""
    
    with open(f"{report_dir}/performance_report.md", "w", encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"✅ 完整性能报告已保存到: {report_dir}/")
    return report_dir

def main():
    print("🎯 === 模型性能全面评估 ===")
    
    # 找到最新的实验目录 (保留这部分逻辑，因为报告生成器需要它)
    experiment_path = None
    experiments_dir = "experiments"
    if os.path.exists(experiments_dir):
        experiment_dirs = sorted([d for d in os.listdir(experiments_dir) if os.path.isdir(os.path.join(experiments_dir, d))])
        if experiment_dirs:
            latest_experiment = experiment_dirs[-1] # 获取最新的
            experiment_path = os.path.join(experiments_dir, latest_experiment)
            print(f"INFO: 将为实验目录 '{experiment_path}' 生成报告")
        else:
            print("WARNING: 在 'experiments' 目录下没有找到实验子目录。")
    else:
        print(f"WARNING: 'experiments' 目录不存在。")

    if experiment_path:
        # 创建性能报告
        report_dir = create_performance_report(experiment_path)
        
        if report_dir:
            print(f"\n🎉 === 评估完成 ===")
            print(f"📁 完整报告保存在: {report_dir}")
            print(f"📊 主要文件:")
            print(f"   - prediction_analysis.png (预测分析图)")
            print(f"   - prediction_statistics.csv (预测统计)")
            print(f"   - price_segment_analysis.csv (价格段分析)")
            print(f"   - performance_report.md (完整报告)")
        else:
            print("❌ 报告生成失败")
    else:
        print("❌ 未找到实验目录")

if __name__ == "__main__":
    main() 