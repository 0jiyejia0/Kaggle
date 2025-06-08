import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 使用 'Agg' 后端，避免在无图形界面的服务器上出错
import matplotlib
matplotlib.use('Agg')

def plot_saleprice_distribution(df, output_dir):
    """
    绘制目标变量 SalePrice 的分布图（原始和对数转换后）
    """
    plt.figure(figsize=(16, 6))

    # 原始分布
    plt.subplot(1, 2, 1)
    sns.histplot(df['SalePrice'], kde=True, bins=50)
    plt.title('Original SalePrice Distribution')

    # 对数转换后分布
    plt.subplot(1, 2, 2)
    # 创建一个临时列用于绘图
    sale_price_log = np.log1p(df['SalePrice'])
    sns.histplot(sale_price_log, kde=True, bins=50)
    plt.title('Log-transformed SalePrice Distribution')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/saleprice_distribution.png")
    plt.close()
    print("✅ SalePrice distribution plot saved.")

    # 绘制 Q-Q plot 进一步验证正态性
    plt.figure(figsize=(16, 6))
    
    # 原始数据的Q-Q plot
    plt.subplot(1, 2, 1)
    stats.probplot(df['SalePrice'], plot=plt)
    plt.title('Probability Plot for Original SalePrice')

    # 对数转换后数据的Q-Q plot
    plt.subplot(1, 2, 2)
    stats.probplot(sale_price_log, plot=plt)
    plt.title('Probability Plot for Log-transformed SalePrice')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/saleprice_qq_plot.png")
    plt.close()
    print("✅ SalePrice Q-Q plot saved.")


def plot_correlation_heatmap(df, output_dir):
    """
    绘制数值特征之间的相关性热力图
    """
    corrmat = df.corr(numeric_only=True)
    k = 15  # 热力图中显示的变量数
    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
    cm = np.corrcoef(df[cols].values.T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                yticklabels=cols.values, xticklabels=cols.values, cmap='viridis')
    plt.title(f'Top {k} Correlated Features Heatmap')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_heatmap.png")
    plt.close()
    print("✅ Correlation heatmap saved.")


def plot_scatter_plots(df, output_dir):
    """
    绘制关键特征与SalePrice的散点图以识别异常值
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'])
    plt.title('GrLivArea vs SalePrice (shows outliers)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/grlivarea_vs_saleprice.png")
    plt.close()
    print("✅ GrLivArea vs SalePrice scatter plot saved.")


def plot_box_plots(df, output_dir):
    """
    绘制关键类别特征与SalePrice的箱线图
    """
    plt.figure(figsize=(20, 8))
    # 排序使得箱线图更有序
    order = df.groupby('Neighborhood')['SalePrice'].median().sort_values().index
    sns.boxplot(x='Neighborhood', y='SalePrice', data=df, order=order)
    plt.xticks(rotation=45, ha='right')
    plt.title('SalePrice Distribution by Neighborhood')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/neighborhood_boxplot.png")
    plt.close()
    print("✅ Neighborhood vs SalePrice box plot saved.")


def generate_eda_plots():
    """
    主函数，执行所有EDA绘图功能。
    如果绘图目录已存在，则会跳过生成以节省时间。
    若要重新生成，请手动删除 'experiments/eda_plots' 目录。
    """
    print("\n📊 === 开始探索性数据分析 (EDA) ===")
    
    data_path = 'data/train.csv'
    output_dir = 'experiments/eda_plots'
    
    if os.path.exists(output_dir):
        print(f"📂 Directory '{output_dir}' already exists. Skipping EDA plot generation.")
        print("💡 To regenerate plots, please delete the directory first.")
        return

    os.makedirs(output_dir)
    print(f"📂 Created directory: {output_dir}")
    
    try:
        df_train = pd.read_csv(data_path)
        print(f"📄 Data loaded successfully from {data_path}. Shape: {df_train.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Data file not found at {data_path}. Please ensure it exists.")
        return
        
    try:
        plot_saleprice_distribution(df_train.copy(), output_dir)
        plot_correlation_heatmap(df_train.copy(), output_dir)
        plot_scatter_plots(df_train.copy(), output_dir)
        plot_box_plots(df_train.copy(), output_dir)
        print(f"\n🎉 EDA plots successfully generated in: '{output_dir}'")
    except Exception as e:
        print(f"❌ An error occurred during plot generation: {e}") 