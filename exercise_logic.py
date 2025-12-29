import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# ==========================================
# 0. 環境初期化
# ==========================================
def initialize_environment():
    """
    Colab/Local環境を自動判別し、必要なセットアップを完結させる。
    subprocess.run はコマンドの完了まで Python の処理をブロック（待機）します。
    """
    print("--- Environment Initializing ---")
    
    # 1. Colab判定
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("[Status] Google Colab detected. Installing dependencies...")
        # check=True を指定することで、bashのwaitのように終了を確実に待ちます
        subprocess.run([
            "pip", "install", 
            "medmnist", 
            "tensorflow", 
            "scikit-learn", 
            "matplotlib", 
            "seaborn",
            "-q"
        ], check=True)
        print("[Status] Installation finished.")
    else:
        print("[Status] Local environment detected.")

    # 2. パスの自動設定
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
    src_path = os.path.abspath(current_dir)  # src/ ディレクトリ自体をパスに追加
    if src_path not in sys.path:
        sys.path.append(src_path)

    # 3. TensorFlow / GPU の設定
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Found and configured: {gpus}")
        else:
            print("[GPU] Not found. Using CPU.")
            
    except Exception as e:
        print(f"[Warning] TensorFlow/GPU configuration issue: {e}")

    print("--- Setup Complete ---\n")

# --- 修正: インポートエラーを防ぐため、トップレベルの medmnist インポートを削除しました ---

# ==========================================
# 1. データ管理ロジック
# ==========================================
def load_and_preprocess(data_flag='pathmnist', as_rgb=True):
    """
    MedMNISTデータをロードし、正規化と必要に応じた3チャンネル化を行う。
    この関数内でインポートすることで、セットアップ完了を保証します。
    """
    # 遅延インポート: initialize_environment() の後に実行されることを想定
    import medmnist
    from medmnist import INFO

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # データのロード
    train_dataset = DataClass(split='train', download=True)
    test_dataset = DataClass(split='test', download=True)

    # 0-1に正規化
    x_train = train_dataset.imgs.astype('float32') / 255.0
    x_test = test_dataset.imgs.astype('float32') / 255.0
    
    # 転移学習モデル（MobileNet等）のために3チャンネル化が必要な場合
    if as_rgb and len(x_train.shape) == 3:
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
    elif len(x_train.shape) == 3:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    y_train = train_dataset.labels.astype('float32').flatten()
    y_test = test_dataset.labels.astype('float32').flatten()

    return (x_train, y_train), (x_test, y_test), info

# ==========================================
# 2. モデル構築ロジック
# ==========================================
def build_model(input_shape, num_classes, model_type='simple', multi_label=False):
    """
    指定されたタイプ（Simple CNN または Transfer Learning）のモデルを構築する。
    """
    if model_type == 'simple':
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2)
        ])
    else:
        # 転移学習（MobileNetV2）
        base_model = applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
        base_model.trainable = False
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Resizing(32, 32),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2)
        ])

    # 出力層の設定
    if multi_label:
        model.add(layers.Dense(num_classes, activation='sigmoid'))
        loss = 'binary_crossentropy'
    else:
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy'

    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    return model

# ==========================================
# 3. 評価・可視化ロジック
# ==========================================
def plot_history(history):
    """学習曲線の表示"""
    plt.figure(figsize=(12, 4))
    for i, metrics in enumerate(['loss', 'accuracy']):
        plt.subplot(1, 2, i+1)
        plt.plot(history.history[metrics], label='train')
        plt.plot(history.history[f'val_{metrics}'], label='val')
        plt.title(metrics.capitalize())
        plt.legend()
    plt.show()

def show_evaluation_reports(model, x_test, y_test, labels_dict, multi_label=False):
    """混同行列や精度指標を表示"""
    y_pred_prob = model.predict(x_test)
    
    if multi_label:
        auc = roc_auc_score(y_test, y_pred_prob)
        print(f"Overall AUC Score: {auc:.4f}")
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = y_test.flatten()
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels_dict.values(), yticklabels=labels_dict.values())
        plt.title('Confusion Matrix')
        plt.show()
        
        print(classification_report(y_true, y_pred, target_names=labels_dict.values()))

# ==========================================
# 4. 高度な分析ロジック (Grad-CAM)
# ==========================================
def compute_gradcam(model, img_array, last_conv_layer_name):
    """Grad-CAMヒートマップの生成"""
    grad_model = models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, np.argmax(preds[0])]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., np.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()