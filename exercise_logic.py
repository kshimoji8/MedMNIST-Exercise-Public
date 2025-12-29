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
    src_path = os.path.abspath(current_dir)
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


# ==========================================
# 1. データ管理ロジック
# ==========================================
def load_and_preprocess(data_flag='pathmnist', as_rgb=True, binary_classification=False):
    """
    MedMNISTデータをロードし、正規化と必要に応じた3チャンネル化を行う。
    
    Parameters
    ----------
    data_flag : str
        MedMNISTデータセット名（例: 'pathmnist', 'chestmnist'）
    as_rgb : bool
        3チャンネル化するかどうか
    binary_classification : bool
        Trueの場合、マルチラベルデータを2値分類（正常=0 vs 異常=1）に変換
        
    Returns
    -------
    tuple
        ((x_train, y_train), (x_test, y_test), info)
    """
    import medmnist
    from medmnist import INFO

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    train_dataset = DataClass(split='train', download=True)
    test_dataset = DataClass(split='test', download=True)

    x_train = train_dataset.imgs.astype('float32') / 255.0
    x_test = test_dataset.imgs.astype('float32') / 255.0
    
    if as_rgb and len(x_train.shape) == 3:
        x_train = np.repeat(x_train[..., np.newaxis], 3, axis=-1)
        x_test = np.repeat(x_test[..., np.newaxis], 3, axis=-1)
    elif len(x_train.shape) == 3:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    y_train = train_dataset.labels.astype('float32')
    y_test = test_dataset.labels.astype('float32')
    
    # 2値分類モード: マルチラベルを「正常(0) vs 異常(1)」に変換
    if binary_classification:
        # 2次元配列の場合（マルチラベル）
        if len(y_train.shape) == 2 and y_train.shape[1] > 1:
            # いずれかのラベルが1なら異常(1)、全て0なら正常(0)
            y_train = (y_train.sum(axis=1) > 0).astype('float32')
            y_test = (y_test.sum(axis=1) > 0).astype('float32')
            
            # info を更新して2値分類用に変更
            info = dict(info)  # コピーを作成
            info['label'] = {'0': 'normal', '1': 'abnormal'}
            info['n_classes'] = 2
            
            print(f"[Info] Converted to binary classification: Normal vs Abnormal")
            print(f"  Training: {int((y_train == 0).sum())} normal, {int((y_train == 1).sum())} abnormal")
            print(f"  Test: {int((y_test == 0).sum())} normal, {int((y_test == 1).sum())} abnormal")
        else:
            print("[Warning] binary_classification=True but data is not multi-label. Ignored.")
    else:
        # 通常の処理
        if len(y_train.shape) == 1:
            y_train = y_train.reshape(-1, 1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1, 1)
        
        if y_train.shape[1] == 1:
            y_train = y_train.flatten()
        if y_test.shape[1] == 1:
            y_test = y_test.flatten()

    return (x_train, y_train), (x_test, y_test), info


# ==========================================
# 2. モデル構築ロジック
# ==========================================
def build_model(input_shape, num_classes, model_type='simple', multi_label=False):
    """
    指定されたタイプ（Simple CNN または Transfer Learning）のモデルを構築する。
    
    Parameters
    ----------
    input_shape : tuple
        入力画像のshape（例: (28, 28, 3)）
    num_classes : int
        クラス数（2値分類の場合は1を指定）
    model_type : str
        'simple' または 'transfer'
    multi_label : bool
        マルチラベル分類かどうか（2値分類の場合もTrue）
    """
    if model_type == 'simple':
        layers_list = [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', name='last_conv_layer'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2)
        ]
        if multi_label or num_classes == 1:
            # マルチラベルまたは2値分類
            layers_list.append(layers.Dense(num_classes, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            layers_list.append(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        model = models.Sequential(layers_list)
    else:
        base_model = applications.MobileNetV2(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
        base_model.trainable = False
        layers_list = [
            layers.Input(shape=input_shape),
            layers.Resizing(32, 32),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2)
        ]
        if multi_label or num_classes == 1:
            layers_list.append(layers.Dense(num_classes, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            layers_list.append(layers.Dense(num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        model = models.Sequential(layers_list)

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
# 4. Grad-CAM ロジック（Keras 3 対応版）
# ==========================================
def compute_gradcam(model, img_array, last_conv_layer_name='last_conv_layer', class_index=None):
    """
    Grad-CAMヒートマップを計算する。
    
    【技術詳細】Keras 3 (TensorFlow 2.16+) では model.input への直接アクセスが
    制限されているため、Functional APIで中間モデルを再構築している。
    
    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    img_array : np.ndarray
        入力画像（shape: (1, H, W, C)）
    last_conv_layer_name : str
        Grad-CAMを計算する畳み込み層の名前
    class_index : int, optional
        対象クラスのインデックス。Noneの場合は最も確率が高いクラス（または2値分類では出力そのもの）
        
    Returns
    -------
    np.ndarray
        ヒートマップ（元画像と同じサイズ）
    """
    # 入力テンソルを新規作成してFunctional APIで再構築
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
    
    if conv_output is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")
    
    grad_model = tf.keras.Model(inputs=inputs, outputs=[conv_output, x])
    
    # 勾配を計算
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        
        # 2値分類（出力が1ユニット）の場合
        if preds.shape[-1] == 1:
            class_channel = preds[:, 0]
        # 多クラス分類の場合
        elif class_index is not None:
            class_channel = preds[:, class_index]
        else:
            class_idx = tf.argmax(preds[0])
            class_channel = preds[:, class_idx]
    
    grads = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # ヒートマップ生成
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    
    # 元画像サイズにリサイズ
    heatmap_resized = tf.image.resize(
        heatmap[..., tf.newaxis], 
        (img_array.shape[1], img_array.shape[2])
    )
    return tf.squeeze(heatmap_resized).numpy()


def show_gradcam(model, image, last_conv_layer_name='last_conv_layer', 
                 title_original="Original Image", title_gradcam="Grad-CAM",
                 class_index=None):
    """
    Grad-CAMヒートマップを可視化する（高レベルAPI）。
    
    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    image : np.ndarray
        入力画像（shape: (H, W, C) または (1, H, W, C)）
    last_conv_layer_name : str
        畳み込み層の名前（デフォルト: 'last_conv_layer'）
    title_original : str
        元画像のタイトル
    title_gradcam : str
        Grad-CAM画像のタイトル
    class_index : int, optional
        対象クラスのインデックス
    
    Returns
    -------
    np.ndarray
        計算されたヒートマップ
    """
    # バッチ次元を追加
    if len(image.shape) == 3:
        img_array = image[np.newaxis, ...]
    else:
        img_array = image
    
    # ヒートマップ計算
    heatmap = compute_gradcam(model, img_array, last_conv_layer_name, class_index)
    
    # 可視化
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img_array[0])
    plt.title(title_original)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_array[0])
    plt.imshow(heatmap, cmap='jet', alpha=0.4)
    plt.title(title_gradcam)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return heatmap


def show_gradcam_comparison(model, images, labels=None, last_conv_layer_name='last_conv_layer', 
                            cols=4, figsize=None, class_index=None):
    """
    複数画像のGrad-CAMを比較表示する。
    
    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    images : np.ndarray
        入力画像の配列（shape: (N, H, W, C)）
    labels : list, optional
        各画像のラベル（タイトル用）
    last_conv_layer_name : str
        畳み込み層の名前
    cols : int
        1行あたりの列数
    figsize : tuple, optional
        図のサイズ
    class_index : int, optional
        対象クラスのインデックス
    
    Returns
    -------
    list
        計算されたヒートマップのリスト
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    if figsize is None:
        figsize = (cols * 4, rows * 4)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()
    
    heatmaps = []
    
    for i, img in enumerate(images):
        img_array = img[np.newaxis, ...] if len(img.shape) == 3 else img
        heatmap = compute_gradcam(model, img_array, last_conv_layer_name, class_index)
        heatmaps.append(heatmap)
        
        axes[i].imshow(img_array[0])
        axes[i].imshow(heatmap, cmap='jet', alpha=0.4)
        if labels is not None and i < len(labels):
            axes[i].set_title(labels[i])
        axes[i].axis('off')
    
    # 余分なサブプロットを非表示
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return heatmaps
