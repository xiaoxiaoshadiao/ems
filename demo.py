def quick_demo():
    # 初始化框架
    framework = AM2FFramework()

    # 加载预训练模型（实际使用时从文件加载）
    framework.famm.load_pretrained('weights/famm_weights.pth')
    framework.ramm.load_pretrained('weights/ramm_weights.pth')

    # 模拟实时数据流
    real_time_window = np.random.randn(100)  # 100个时间步的实时数据

    # 进行预测
    prediction, decision, confidence = framework.predict(real_time_window)

    print(f"决策: {decision}")
    print(f"预测值: {prediction:.4f}")
    print(f"融合权重: {confidence:.2f}")


if __name__ == '__main__':
    quick_demo()