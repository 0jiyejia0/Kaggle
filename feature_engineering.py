import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

def create_interaction_features(df):
    """创建特征交互"""
    # 填充缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 面积相关交互特征
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
    df['Total_sqr_footage'] = (df['BsmtFinSF1'] + df['BsmtFinSF2'] +
                              df['1stFlrSF'] + df['2ndFlrSF'])
    df['Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) +
                            df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))
    df['Total_porch_sf'] = (df['OpenPorchSF'] + df['3SsnPorch'] +
                           df['EnclosedPorch'] + df['ScreenPorch'])
    
    # 质量和条件交互
    df['Overall_quality_cond'] = df['OverallQual'] * df['OverallCond']
    
    # 安全的质量映射
    garage_qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
    garage_cond_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
    
    df['GarageQual_num'] = df['GarageQual'].map(garage_qual_map).fillna(0)
    df['GarageCond_num'] = df['GarageCond'].map(garage_cond_map).fillna(0)
    df['Garage_quality_cond'] = df['GarageQual_num'] * df['GarageCond_num']
    
    # 外部质量映射
    extern_qual_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
    extern_cond_map = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1}
    
    df['ExterQual_num'] = df['ExterQual'].map(extern_qual_map).fillna(3)
    df['ExterCond_num'] = df['ExterCond'].map(extern_cond_map).fillna(3)
    df['External_quality_cond'] = df['ExterQual_num'] * df['ExterCond_num']
    
    # 年份相关特征
    df['House_age'] = df['YrSold'] - df['YearBuilt']
    df['Remod_age'] = df['YrSold'] - df['YearRemodAdd']
    df['Garage_age'] = (df['YrSold'] - df['GarageYrBlt']).fillna(0)
    df['Recent_remodel'] = (df['YrSold'] - df['YearRemodAdd'] <= 5).astype(int)
    
    # 面积比率特征（避免除零）
    df['Living_area_ratio'] = df['GrLivArea'] / (df['LotArea'] + 1)
    df['Garage_ratio'] = df['GarageArea'] / (df['GrLivArea'] + 1)
    df['Basement_ratio'] = df['TotalBsmtSF'] / (df['GrLivArea'] + 1)
    
    return df

def create_polynomial_features(df, important_features, degree=2):
    """为重要特征创建多项式特征（优化性能）"""
    try:
        # 确保特征存在且无缺失值
        available_features = [f for f in important_features if f in df.columns]
        if not available_features:
            return df
            
        # 填充缺失值
        for feat in available_features:
            df[feat] = df[feat].fillna(df[feat].mean())
        
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[available_features])
        
        # 创建新特征名
        new_features = {}
        feature_idx = len(available_features)  # 跳过原始特征
        
        for i in range(len(available_features)):
            for j in range(i, len(available_features)):
                if feature_idx < poly_features.shape[1]:
                    if i == j:
                        name = f"{available_features[i]}_pow_2"
                    else:
                        name = f"{available_features[i]}_x_{available_features[j]}"
                    new_features[name] = poly_features[:, feature_idx]
                    feature_idx += 1
        
        # 使用pd.concat一次性添加所有新特征
        new_df = pd.DataFrame(new_features, index=df.index)
        df = pd.concat([df, new_df], axis=1)
        
    except Exception as e:
        print(f"创建多项式特征时出错: {str(e)}")
        print("跳过多项式特征创建")
    
    return df

def create_binned_features(df):
    """创建分箱特征（优化性能）"""
    try:
        # 使用字典收集新特征，然后一次性添加
        new_features = {}
        
        # 对重要的数值特征进行分箱
        if 'OverallQual' in df.columns:
            qual_bins = pd.cut(df['OverallQual'], 
                              bins=5, 
                              labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh'])
            qual_dummies = pd.get_dummies(qual_bins, prefix='OverallQual_bin')
            new_features.update(qual_dummies.to_dict('series'))
        
        if 'GrLivArea' in df.columns:
            area_bins = pd.cut(df['GrLivArea'], 
                              bins=5, 
                              labels=['VerySmall', 'Small', 'Medium', 'Large', 'VeryLarge'])
            area_dummies = pd.get_dummies(area_bins, prefix='GrLivArea_bin')
            new_features.update(area_dummies.to_dict('series'))
        
        if 'TotalSF' in df.columns:
            total_bins = pd.cut(df['TotalSF'], 
                               bins=5, 
                               labels=['VerySmall', 'Small', 'Medium', 'Large', 'VeryLarge'])
            total_dummies = pd.get_dummies(total_bins, prefix='TotalSF_bin')
            new_features.update(total_dummies.to_dict('series'))
        
        # 一次性添加所有新特征
        if new_features:
            new_df = pd.DataFrame(new_features, index=df.index)
            df = pd.concat([df, new_df], axis=1)
            
    except Exception as e:
        print(f"创建分箱特征时出错: {str(e)}")
        print("跳过分箱特征创建")
    
    return df

def select_features(X, y, k=50):
    """使用SelectKBest选择最重要的特征"""
    try:
        # 确保没有NaN值
        X_clean = X.copy()
        
        # 填充数值型特征的缺失值
        numeric_cols = X_clean.select_dtypes(include=[np.number]).columns
        X_clean[numeric_cols] = X_clean[numeric_cols].fillna(X_clean[numeric_cols].mean())
        
        # 填充类别型特征的缺失值
        categorical_cols = X_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X_clean[col] = X_clean[col].fillna('Missing')
        
        # 对类别特征进行编码
        X_encoded = pd.get_dummies(X_clean, columns=categorical_cols)
        
        # 确保y没有缺失值
        y_clean = y.fillna(y.mean())
        
        # 调整k值以确保不超过特征数量
        k = min(k, X_encoded.shape[1])
        
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X_encoded, y_clean)
        selected_features = X_encoded.columns[selector.get_support()].tolist()
        
        # 返回原始特征名（如果可能）
        original_features = []
        for feat in selected_features:
            base_feat = feat.split('_')[0]  # 获取基础特征名
            if base_feat in X.columns:
                original_features.append(base_feat)
            else:
                original_features.append(feat)
        
        # 去重并保持原有特征
        final_features = []
        for feat in X.columns:
            if feat in original_features or any(feat in sel_feat for sel_feat in selected_features):
                final_features.append(feat)
        
        return final_features[:k] if len(final_features) > k else final_features
        
    except Exception as e:
        print(f"特征选择时出错: {str(e)}")
        print("返回数值特征")
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        return numeric_features[:k] if len(numeric_features) > k else numeric_features

def add_features(df_train, df_test):
    """主特征工程函数"""
    print("开始特征工程...")
    
    # 合并训练集和测试集以进行一致的特征工程
    train_size = df_train.shape[0]
    all_data = pd.concat([df_train, df_test], axis=0, ignore_index=True)
    
    print("创建交互特征...")
    # 创建交互特征
    all_data = create_interaction_features(all_data)
    
    print("创建多项式特征...")
    # 为重要特征创建多项式特征
    important_numeric_features = ['OverallQual', 'GrLivArea', 'TotalSF']
    all_data = create_polynomial_features(all_data, important_numeric_features)
    
    print("创建分箱特征...")
    # 创建分箱特征
    all_data = create_binned_features(all_data)
    
    # 最终清理缺失值
    numeric_cols = all_data.select_dtypes(include=[np.number]).columns
    all_data[numeric_cols] = all_data[numeric_cols].fillna(0)
    
    # 分离回训练集和测试集
    df_train_new = all_data.iloc[:train_size, :].copy()
    df_test_new = all_data.iloc[train_size:, :].copy()
    
    print(f"特征工程完成。训练集形状: {df_train_new.shape}, 测试集形状: {df_test_new.shape}")
    
    return df_train_new, df_test_new 