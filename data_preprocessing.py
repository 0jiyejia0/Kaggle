import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def load_data(train_path="data/train.csv", test_path="data/test.csv"):
    """加载数据"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def handle_missing_values(df):
    """处理缺失值"""
    # 数值型特征用中位数填充
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_imputer = SimpleImputer(strategy='median')
    df[numeric_features] = numeric_imputer.fit_transform(df[numeric_features])
    
    # 类别型特征用众数填充
    categorical_features = df.select_dtypes(include=['object']).columns
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_features] = categorical_imputer.fit_transform(df[categorical_features])
    
    return df

def encode_categorical_features(df_train, df_test):
    """编码类别特征"""
    categorical_features = df_train.select_dtypes(include=['object']).columns
    
    # 创建编码器字典
    encoders = {}
    
    # 对每个类别特征进行标签编码
    for feature in categorical_features:
        encoders[feature] = LabelEncoder()
        # 合并训练集和测试集的唯一值进行拟合
        unique_values = pd.concat([df_train[feature], df_test[feature]]).unique()
        encoders[feature].fit(unique_values)
        
        # 转换训练集和测试集
        df_train[feature] = encoders[feature].transform(df_train[feature])
        df_test[feature] = encoders[feature].transform(df_test[feature])
    
    return df_train, df_test, encoders

def scale_numeric_features(df_train, df_test):
    """标准化数值特征"""
    numeric_features = df_train.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    
    # 对训练集进行拟合和转换
    df_train[numeric_features] = scaler.fit_transform(df_train[numeric_features])
    # 对测试集仅进行转换
    df_test[numeric_features] = scaler.transform(df_test[numeric_features])
    
    return df_train, df_test, scaler

def handle_skewed_features(df, threshold=0.75):
    """处理偏态特征"""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    skewed_features = []
    
    for feature in numeric_features:
        # 计算偏度
        skewness = df[feature].skew()
        if abs(skewness) > threshold:
            skewed_features.append(feature)
            # 对偏态特征进行对数转换
            df[feature] = np.log1p(df[feature] - df[feature].min() + 1)
    
    return df, skewed_features

def load_and_preprocess_data(train_path="data/train.csv", test_path="data/test.csv"):
    """加载和预处理数据的主函数"""
    print("加载数据...")
    train_df, test_df = load_data(train_path, test_path)
    
    # 保存测试集的ID
    test_ids = test_df['Id']

    # 分离目标变量
    y_train = train_df['SalePrice']
    X_train = train_df.drop('SalePrice', axis=1)
    X_test = test_df.copy()
    
    print("处理缺失值...")
    X_train = handle_missing_values(X_train)
    X_test = handle_missing_values(X_test)
    
    print("处理偏态特征...")
    X_train, skewed_features = handle_skewed_features(X_train)
    X_test, _ = handle_skewed_features(X_test)
    
    print("编码类别特征...")
    X_train, X_test, _ = encode_categorical_features(X_train, X_test)
    
    print("标准化数值特征...")
    X_train, X_test, _ = scale_numeric_features(X_train, X_test)
    
    print("数据预处理完成。")
    return X_train, X_test, y_train, None, test_ids  # 返回 test_ids