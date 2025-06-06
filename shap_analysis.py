import numpy as np
import shap
import matplotlib.pyplot as plt

def shap_analysis(model, X_sample, feature_names=None, output_plot="shap_summary.png"):
    """
    对给定模型进行SHAP解释性分析。
    model: 已训练的模型对象（支持predict方法）。
    X_sample: 用于计算SHAP值的一批样本特征（DataFrame或numpy数组）。
    feature_names: 特征名称列表（可选，如果X_sample为DataFrame则可留空）。
    output_plot: 输出的SHAP总结图文件名。
    """
    # 如果提供了DataFrame且未指定特征名，则使用DataFrame的列名
    if feature_names is None:
        if hasattr(X_sample, "columns"):
            feature_names = X_sample.columns.tolist()
        else:
            # 若X_sample为numpy数组则生成默认特征名
            feature_names = [f"Feature_{i}" for i in range(X_sample.shape[1])]

    # 针对树模型使用TreeExplainer，加速计算；否则使用通用Explainer
    model_type = model.__class__.__name__
    try:
        if model_type in ["XGBRegressor", "LGBMRegressor", "CatBoostRegressor"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.Explainer(model, X_sample)
            shap_values = explainer(X_sample).values
    except Exception as e:
        # 如果一般Explainer失败，退而求其次使用TreeExplainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

    # 将计算所得的SHAP值转换为numpy数组
    shap_array = np.array(shap_values)
    if shap_array.ndim == 3:
        # 若模型输出多个目标（例如分类模型）会得到多维数组，这里仅取第一维
        shap_array = shap_array[0]
    # 计算每个特征的平均绝对SHAP值（体现整体重要性）
    mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
    # 特征按重要性排序
    importance_order = np.argsort(mean_abs_shap)[::-1]

    # 输出前10个最重要特征及其SHAP值大小
    print("Top 10 Important Features (by SHAP):")
    for idx in importance_order[:10]:
        print(f"  {feature_names[idx]}: {mean_abs_shap[idx]:.4f}")

    # 生成并保存SHAP特征重要性总结图（条形图）
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_array, X_sample, feature_names=feature_names, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(output_plot)
    print(f"SHAP summary plot saved to {output_plot}")
