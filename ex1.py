from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import numpy as np


def sel_data():
    # 初始化 Spark 会话
    spark = SparkSession.builder \
        .appName("Airline Data") \
        .getOrCreate()

    # 读取CSV 文件
    file_path_pattern = './data/flights.csv'
    df = spark.read.csv(file_path_pattern, header=True, inferSchema=True)

    # 筛选航空公司代号为 'DL' 的行
    df_filtered = df.filter(df.Carrier == 'DL')

    # 根据需要重新分区（我只有2核心）
    df_filtered = df_filtered.repartition(2)

    # 保存处理后的数据到新文件（或进行其他操作）
    output_path = 'data/filtered_data'
    df_filtered.write.csv(output_path, header=True)

    # 结束 Spark 会话
    spark.stop()


def data_clean():
    from pyspark.sql.functions import col

    # 初始化 Spark 会话
    spark = SparkSession.builder \
        .appName("Data Cleaning") \
        .getOrCreate()

    # 加载数据（替换为您的文件路径）
    df = spark.read.csv('./data/filtered_data/*.csv', header=True, inferSchema=True)

    # 1. 删除重复数据
    df = df.dropDuplicates()

    # 2. 处理缺失值
    df = df.dropna()

    # 4. 过滤无效或异常数据
    df = df.filter(col("DayofMonth") <= 31)
    df = df.filter(col("DayOfWeek") <= 7)

    # 保存清洗后的数据
    df.write.csv('data/DL_Cleaning.csv', header=True)

    # 结束 Spark 会话
    spark.stop()


def corr():
    spark = SparkSession.builder \
        .appName("Feature Correlation Analysis") \
        .getOrCreate()

    # 读取数据
    df = spark.read.csv('./data/DL_Cleaning.csv/*.csv', header=True, inferSchema=True)

    # 计算每个特征与 ArrDelay 的相关性
    features = ['DayofMonth', 'DayOfWeek', 'OriginAirportID', 'DestAirportID', 'DepDelay']
    correlations = {}
    for feature in features:
        correlation = df.stat.corr(feature, 'ArrDelay')
        correlations[feature] = correlation

    # 关闭 Spark 会话
    spark.stop()

    # 将相关性数据转换为图表所需格式
    labels = list(correlations.keys())
    values = list(correlations.values())

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='blue')
    plt.xlabel('Features')
    plt.ylabel('Correlation with ArrDelay')
    plt.title('Feature Correlation with ArrDelay')
    plt.savefig('Feature-Correlation.png')


def knn():
    from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error

    spark = SparkSession.builder \
        .appName("KNN Model Preparation") \
        .getOrCreate()

    # 读取数据
    df = spark.read.csv('./data/DL_Cleaning.csv/*.csv', header=True, inferSchema=True)

    # 选择特征和标签
    df_selected = df.select('OriginAirportID', 'DestAirportID', 'DepDelay', 'ArrDelay')

    # 将 PySpark DataFrame 转换为 Pandas DataFrame
    pandas_df = df_selected.toPandas()

    # 关闭 Spark 会话
    spark.stop()

    # 分割数据集，添加随机种子
    X = pandas_df[['OriginAirportID', 'DestAirportID', 'DepDelay']]
    y = pandas_df['ArrDelay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建 KNN 模型
    knn = KNeighborsRegressor()

    # 定义超参数网格
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}  # 可以调整不同的邻居数量

    # 使用 GridSearchCV 寻找最佳超参数值(分为5个)
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 打印最佳超参数值
    print("Best Hyperparameters:", grid_search.best_params_)

    # 使用最佳模型进行交叉验证
    best_model = grid_search.best_estimator_
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)
    print("Cross-Validation RMSE:", cv_rmse)

    # 预测和评估测试集
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    print(f"Test Set RMSE: {rmse}")


def rmse_knn():
    results = {
        "Best Hyperparameters": {'n_neighbors': 7},
        "Cross-Validation RMSE": [15.210596, 15.6412172, 15.16906986, 15.29447495, 15.24870332],
        "Test Set RMSE": 15.24562650862433
    }

    # 计算交叉验证的平均损失
    average_cv_rmse = np.mean(results["Cross-Validation RMSE"])
    results["Average CV RMSE"] = average_cv_rmse

    # 打印包含平均损失的字典
    print(results)


def LRegression():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score, train_test_split

    spark = SparkSession.builder \
        .appName("LRegression Model Preparation") \
        .getOrCreate()

    # 读取数据
    df = spark.read.csv('./data/DL_Cleaning.csv/*.csv', header=True, inferSchema=True)

    # 选择特征和标签
    df_selected = df.select('OriginAirportID', 'DestAirportID', 'DepDelay', 'ArrDelay')

    # 将 PySpark DataFrame 转换为 Pandas DataFrame
    pandas_df = df_selected.toPandas()

    # 关闭 Spark 会话
    spark.stop()

    # 最大多项式次数
    max_degree = 5

    # 初始化损失列表
    losses = []

    # 分割数据集
    X = pandas_df[['OriginAirportID', 'DestAirportID', 'DepDelay']]
    y = pandas_df['ArrDelay']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 循环尝试不同次数的多项式模型
    for degree in range(1, max_degree + 1):
        # 使用多项式特征变换
        poly = PolynomialFeatures(degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        # 创建线性回归模型
        model = LinearRegression()
        model.fit(X_train_poly, y_train)

        # 使用交叉验证评估模型性能
        scores = cross_val_score(model, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=5)

        # 计算测试集上的均方误差
        y_pred = model.predict(X_test_poly)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # 输出交叉验证得分和测试集均方根误差
        print(f"Degree: {degree}, Cross-Validation Score: {-np.mean(scores)}, Test RMSE: {rmse}")



        # 添加到损失列表
        losses.append(mse)

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_degree + 1), losses, marker='o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Polynomial Degree')
    plt.savefig("rmse vs degree.png")


# 调用函数
LRegression()
