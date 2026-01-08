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
    
    【技術解説】
    この関数は以下の3つの処理を行う：
    
    1. 環境判定: sys.modulesに'google.colab'が含まれているかで
       Google Colab環境かローカル環境かを判別する。
       
    2. 依存関係のインストール: Colab環境の場合、pipで必要なライブラリを
       インストールする。subprocess.runはコマンドの完了まで処理をブロック
       （待機）するため、インストール完了後に次の処理に進む。
       
    3. GPU設定: TensorFlowのGPUメモリ成長（memory_growth）を有効化する。
       これにより、必要な分だけGPUメモリを確保し、他のプロセスと
       GPUを共有できるようになる。
    
    【補足】
    TF_CPP_MIN_LOG_LEVEL='2': TensorFlowの警告メッセージを抑制
    TF_ENABLE_ONEDNN_OPTS='0': oneDNN最適化を無効化（再現性のため）
    """
    print("--- 環境を初期化中 ---")
    
    # 1. Colab判定
    IN_COLAB = 'google.colab' in sys.modules
    
    if IN_COLAB:
        print("[状態] Google Colabを検出しました。依存関係をインストール中...")
        subprocess.run([
            "pip", "install", 
            "medmnist", 
            "tensorflow", 
            "scikit-learn", 
            "matplotlib", 
            "seaborn",
            "-q"
        ], check=True)
        print("[状態] インストールが完了しました。")
    else:
        print("[状態] ローカル環境を検出しました。")

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
            print(f"[GPU] 検出・設定完了: {gpus}")
        else:
            print("[GPU] 検出されませんでした。CPUを使用します。")
            
    except Exception as e:
        print(f"[警告] TensorFlow/GPU設定の問題: {e}")

    print("--- セットアップ完了 ---\n")


# ==========================================
# 1. データ管理ロジック
# ==========================================
def load_and_preprocess(data_flag='pathmnist', as_rgb=True, binary_classification=False):
    """
    MedMNISTデータをロードし、正規化と必要に応じた3チャンネル化を行う。
    
    【技術解説】
    1. データロード: MedMNISTライブラリからデータセットを取得する。
       MedMNISTは医用画像を28×28ピクセルに標準化したデータセット群で、
       病理画像（PathMNIST）、皮膚画像（DermaMNIST）、胸部X線（ChestMNIST）
       など10種類以上のデータセットが含まれる。
    
    2. 正規化: ピクセル値を0-255から0-1の範囲に変換する（/255.0）。
       ニューラルネットワークは入力値が小さい範囲にある方が学習が安定する。
    
    3. チャンネル変換: グレースケール画像（1チャンネル）を3チャンネルに
       複製する。これは転移学習で使用するImageNet事前学習モデルが
       RGB（3チャンネル）入力を期待しているため。
    
    4. 二値分類変換: ChestMNISTなどのマルチラベルデータを「正常vs異常」の
       二値分類に変換する。いずれかの疾患ラベルが1なら「異常」とする。
    
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
        - x_train, x_test: 画像データ（float32, 0-1正規化済み）
        - y_train, y_test: ラベルデータ
        - info: データセットのメタ情報（ラベル名、クラス数など）
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
            
            print(f"[情報] 二値分類に変換しました: 正常 vs 異常")
            print(f"  訓練データ: 正常 {int((y_train == 0).sum())}件, 異常 {int((y_train == 1).sum())}件")
            print(f"  テストデータ: 正常 {int((y_test == 0).sum())}件, 異常 {int((y_test == 1).sum())}件")
        else:
            print("[警告] binary_classification=Trueですが、データがマルチラベルではありません。無視されました。")
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
    
    【技術解説】
    'simple'モード（基本的なCNN）:
    - Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Flatten → Dense(64) → 出力
    - 畳み込み層（Conv2D）: 3×3のフィルタで局所的な特徴を抽出する。
      活性化関数ReLU（Rectified Linear Unit）は負の値を0にし、
      非線形性を導入することで複雑なパターンを学習可能にする。
    - MaxPooling: 2×2領域の最大値を取り、空間解像度を半減させる。
      位置の微小なずれに対するロバスト性を向上させる。
    - Dropout(0.2): 学習時に20%のユニットをランダムに無効化し、
      過学習を抑制する。
    
    'transfer'モード（転移学習）:
    - MobileNetV2をベースモデルとして使用する。
    - MobileNetV2はImageNet（1000クラス、約120万枚）で事前学習済みの
      軽量モデルで、Depthwise Separable Convolutionにより
      パラメータ数を抑えながら高精度を実現する。
    - include_top=Falseで分類層を除外し、特徴抽出部分のみを使用する。
    - base_model.trainable=Falseで事前学習済みの重みを固定する。
    - GlobalAveragePooling2Dで空間次元を集約し、全結合層で分類する。
    
    【損失関数の選択】
    - 多クラス分類: sparse_categorical_crossentropy（ラベルが整数の場合）
    - 二値/マルチラベル分類: binary_crossentropy
    
    Parameters
    ----------
    input_shape : tuple
        入力画像のshape（例: (28, 28, 3)）
    num_classes : int
        クラス数（2値分類の場合は1を指定）
    model_type : str
        'simple': 基本的なCNN、'transfer': MobileNetV2ベースの転移学習
    multi_label : bool
        マルチラベル分類かどうか（2値分類の場合もTrue）
    
    Returns
    -------
    keras.Model
        コンパイル済みのKerasモデル（optimizer='adam', metrics=['accuracy']）
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
    """
    学習曲線（損失と精度の推移）を表示する。
    
    【技術解説】
    学習曲線は過学習の診断に重要な指標である：
    - 訓練損失が下がり続け、検証損失も下がる → 正常な学習
    - 訓練損失が下がるが、検証損失が上がり始める → 過学習の兆候
    - 両方の損失が高いまま → 学習不足（underfitting）
    
    history.historyには以下のキーが含まれる：
    - 'loss': 訓練データでの損失
    - 'val_loss': 検証データでの損失
    - 'accuracy': 訓練データでの精度
    - 'val_accuracy': 検証データでの精度
    
    Parameters
    ----------
    history : keras.callbacks.History
        model.fit()の戻り値
    """
    plt.figure(figsize=(12, 4))
    for i, metrics in enumerate(['loss', 'accuracy']):
        plt.subplot(1, 2, i+1)
        plt.plot(history.history[metrics], label='train')
        plt.plot(history.history[f'val_{metrics}'], label='val')
        plt.title(metrics.capitalize())
        plt.legend()
    plt.show()


def show_evaluation_reports(model, x_test, y_test, labels_dict, multi_label=False):
    """
    混同行列や精度指標を表示する。
    
    【技術解説】
    評価指標の意味：
    - 混同行列: 実際のラベル（行）と予測ラベル（列）のクロス集計。
      対角成分が正解数、非対角成分が誤分類数を示す。
    - Precision（適合率）: 陽性と予測したもののうち、実際に陽性の割合
    - Recall（再現率/感度）: 実際の陽性のうち、陽性と予測できた割合
    - F1-score: PrecisionとRecallの調和平均
    - AUC: ROC曲線下の面積。1に近いほど分類性能が高い。
    
    医療AIでは特にRecall（感度）が重要：
    - 疾患の見逃し（偽陰性）は治療の遅れにつながる
    - ただし、Recallを上げすぎると偽陽性が増え、
      不要な検査や患者の不安につながる
    
    Parameters
    ----------
    model : keras.Model
        学習済みモデル
    x_test : np.ndarray
        テスト画像
    y_test : np.ndarray
        テストラベル
    labels_dict : dict
        ラベル番号→ラベル名の辞書
    multi_label : bool
        マルチラベル分類の場合True（AUCのみ表示）
    """
    y_pred_prob = model.predict(x_test)
    
    if multi_label:
        auc = roc_auc_score(y_test, y_pred_prob)
        print(f"全体AUCスコア: {auc:.4f}")
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
# 3.5 特徴マップ可視化ロジック
# ==========================================
def visualize_feature_maps(model, image, layer_name=None, max_features=16):
    """
    CNNの中間層（特徴マップ）を可視化する。
    
    【技術詳細】Keras 3 (TensorFlow 2.16+) では model.input への直接アクセスが
    制限されているため、Functional APIで中間モデルを再構築している。
    
    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    image : np.ndarray
        入力画像（shape: (H, W, C) または (1, H, W, C)）
    layer_name : str, optional
        可視化する層の名前。Noneの場合は最初の畳み込み層を使用
    max_features : int
        表示する特徴マップの最大数（デフォルト: 16）
    
    Returns
    -------
    np.ndarray
        特徴マップの配列
    """
    # バッチ次元を追加
    if len(image.shape) == 3:
        img_array = image[np.newaxis, ...]
    else:
        img_array = image
    
    # 畳み込み層を探す
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
    
    if not conv_layers:
        print("[警告] 畳み込み層が見つかりません。")
        return None
    
    if layer_name is None:
        target_layer = conv_layers[0]
    else:
        target_layer = model.get_layer(layer_name)
    
    # Keras 3対応: 入力テンソルを新規作成してFunctional APIで再構築
    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    target_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == target_layer.name:
            target_output = x
            break
    
    if target_output is None:
        print(f"[警告] 層 '{target_layer.name}' が見つかりません。")
        return None
    
    intermediate_model = tf.keras.Model(inputs=inputs, outputs=target_output)
    
    # 特徴マップを取得
    feature_maps = intermediate_model.predict(img_array, verbose=0)
    
    # 可視化
    n_features = min(feature_maps.shape[-1], max_features)
    cols = 4
    rows = (n_features + cols - 1) // cols
    
    fig, axes = plt.subplots(rows + 1, cols, figsize=(12, 3 * (rows + 1)))
    axes = np.array(axes).flatten()
    
    # 元画像を最初に表示
    axes[0].imshow(img_array[0])
    axes[0].set_title('Input Image', fontsize=10)
    axes[0].axis('off')
    
    # 残りの最初の行のセルを非表示
    for i in range(1, cols):
        axes[i].axis('off')
    
    # 特徴マップを表示
    for i in range(n_features):
        ax = axes[cols + i]
        ax.imshow(feature_maps[0, :, :, i], cmap='viridis')
        ax.set_title(f'Feature {i+1}', fontsize=9)
        ax.axis('off')
    
    # 余分なセルを非表示
    for i in range(cols + n_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps from Layer "{target_layer.name}" ({n_features} shown)', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return feature_maps


def visualize_cnn_flow(model, image, labels_dict=None):
    """
    CNNの処理の流れを可視化する（入力→特徴マップ→予測）。
    
    【技術詳細】Keras 3 (TensorFlow 2.16+) では model.input への直接アクセスが
    制限されているため、Functional APIで中間モデルを再構築している。
    
    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    image : np.ndarray
        入力画像（shape: (H, W, C) または (1, H, W, C)）
    labels_dict : dict, optional
        ラベル名の辞書
    
    Returns
    -------
    None
    """
    # バッチ次元を追加
    if len(image.shape) == 3:
        img_array = image[np.newaxis, ...]
    else:
        img_array = image
    
    # 畳み込み層を探す
    conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
    
    if len(conv_layers) < 2:
        print("[警告] 可視化には少なくとも2つの畳み込み層が必要です。")
        return
    
    # Keras 3対応: 各畳み込み層の出力を取得
    layer_outputs = []
    for target_layer in conv_layers[:2]:  # 最初の2層のみ
        inputs = tf.keras.Input(shape=img_array.shape[1:])
        x = inputs
        target_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == target_layer.name:
                target_output = x
                break
        
        if target_output is not None:
            intermediate_model = tf.keras.Model(inputs=inputs, outputs=target_output)
            output = intermediate_model.predict(img_array, verbose=0)
            layer_outputs.append((target_layer.name, output))
    
    # 予測を取得
    prediction = model.predict(img_array, verbose=0)
    
    # 可視化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 1. 入力画像
    axes[0].imshow(img_array[0])
    axes[0].set_title('1. Input', fontsize=11)
    axes[0].axis('off')
    
    # 2. 第1畳み込み層の特徴マップ（代表的な4つを合成）
    fm1 = layer_outputs[0][1]
    fm1_combined = np.mean(fm1[0, :, :, :4], axis=-1)
    axes[1].imshow(fm1_combined, cmap='viridis')
    axes[1].set_title(f'2. Layer 1\n({layer_outputs[0][0]})', fontsize=11)
    axes[1].axis('off')
    
    # 3. 第2畳み込み層の特徴マップ（代表的な4つを合成）
    fm2 = layer_outputs[1][1]
    fm2_combined = np.mean(fm2[0, :, :, :4], axis=-1)
    axes[2].imshow(fm2_combined, cmap='viridis')
    axes[2].set_title(f'3. Layer 2\n({layer_outputs[1][0]})', fontsize=11)
    axes[2].axis('off')
    
    # 4. 予測結果
    if len(prediction.shape) == 1 or prediction.shape[-1] == 1:
        # 二値分類
        prob = float(prediction[0]) if len(prediction.shape) == 1 else float(prediction[0, 0])
        result_text = f'Abnormal: {prob:.1%}'
    else:
        # 多クラス分類
        pred_class = np.argmax(prediction[0])
        pred_prob = prediction[0, pred_class]
        if labels_dict:
            class_name = labels_dict.get(str(pred_class), f'Class {pred_class}')
        else:
            class_name = f'Class {pred_class}'
        result_text = f'Pred: {class_name}\nConf: {pred_prob:.1%}'
    
    axes[3].text(0.5, 0.5, result_text, fontsize=14, ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[3].set_title('4. Prediction', fontsize=11)
    axes[3].axis('off')
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    
    plt.suptitle('CNN Flow: Input -> Features -> Prediction', fontsize=13)
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. Grad-CAM ロジック（Keras 3 対応版）
# ==========================================
def compute_gradcam(model, img_array, last_conv_layer_name='last_conv_layer', class_index=None):
    """
    Grad-CAMヒートマップを計算する。
    
    【技術解説】
    Grad-CAM（Gradient-weighted Class Activation Mapping）は、CNNが
    「画像のどの領域に注目して」分類を行ったかを可視化する手法である。
    
    アルゴリズム:
    1. 対象クラスの出力を最後の畳み込み層の出力で微分（勾配を計算）
    2. 各特徴マップに対する勾配の平均値を重みとして計算
    3. 重み付き和でヒートマップを生成（Global Average Pooling的な処理）
    4. ReLUを適用して負の値を除去（正の寄与のみを可視化）
    5. 元画像サイズにリサイズ
    
    Grad-CAMの特徴:
    - モデルの再学習が不要（任意の学習済みモデルに適用可能）
    - 解像度は最後の畳み込み層のサイズに依存（粗い可視化）
    - クラスごとに異なるヒートマップを生成可能
    
    【Keras 3対応】
    Keras 3 (TensorFlow 2.16+) では model.input への直接アクセスが
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
        対象クラスのインデックス。Noneの場合は最も確率が高いクラス
        
    Returns
    -------
    np.ndarray
        ヒートマップ（元画像と同じサイズ、0-1正規化済み）
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


# ==========================================
# 5. Gradioアプリ構築ロジック
# ==========================================
def create_gradio_app(model, info, data_flag):
    """
    学習済みモデルからGradio Webアプリを構築する。
    
    【技術解説】
    Gradioは機械学習モデルを簡単にWebアプリ化できるライブラリである。
    
    主な特徴:
    - 数行のコードでインタラクティブなUIを構築可能
    - share=Trueで一時的な公開URL（72時間有効）を発行できる
    - 入力・出力のコンポーネントを柔軟にカスタマイズ可能
    
    内部処理:
    1. gr.Interface: 入力→処理関数→出力のパイプラインを定義
    2. predict関数: 画像を28×28にリサイズし、正規化後にモデルで推論
    3. 出力は各クラスの確率を辞書形式で返す
    
    【注意点】
    - 公開URLは誰でもアクセス可能なため、機密データには使用しない
    - 医療目的での公開は規制（薬機法等）の確認が必要
    
    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    info : dict
        データセット情報（ラベル名など）
    data_flag : str
        データセット名（タイトル表示用）
        
    Returns
    -------
    gr.Interface
        構築されたGradioインターフェース（.launch()で起動可能）
    """
    import gradio as gr
    from PIL import Image
    
    label_names = list(info['label'].values())
    
    def predict(image):
        """アップロードされた画像を診断する関数"""
        if image is None:
            return None
        
        # 画像の前処理
        img = Image.fromarray(image).convert('RGB').resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # 予測
        preds = model.predict(img_array, verbose=0)[0]
        
        # 2値分類の場合（sigmoid出力）
        if len(preds.shape) == 0 or len(label_names) == 2:
            prob = float(preds) if len(preds.shape) == 0 else float(preds[0])
            return {label_names[0]: 1 - prob, label_names[1]: prob}
        
        # 多クラス分類の場合（softmax出力）
        return {label_names[i]: float(preds[i]) for i in range(len(label_names))}
    
    # Gradioインターフェースの構築
    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(label="Upload Image"),
        outputs=gr.Label(num_top_classes=len(label_names), label="Prediction"),
        title="Medical Image Diagnosis AI",
        description=f"Dataset: {data_flag} | Upload an image to get AI diagnosis.",
        examples=None,
        flagging_mode="never"
    )
    
    return demo


# ==========================================
# 6. サンプル画像生成ロジック
# ==========================================
def create_sample_images(x_test, y_test, info, data_flag, n_samples=10):
    """
    テストデータからサンプル画像を作成し、ZIPファイルにまとめる。
    
    【技術解説】
    ハイブリッドサンプリング:
    1. 各クラスから最低1枚を選択（全クラスが含まれることを保証）
    2. 残りはランダムに選択（重複なし）
    
    この方式により、少数クラスもサンプルに含まれ、
    Webアプリのテストで全クラスの動作を確認できる。
    
    ファイル命名規則:
    - {連番}_{クラス名}.png
    - 例: 01_adenocarcinoma.png, 02_normal.png
    
    Parameters
    ----------
    x_test : np.ndarray
        テスト画像データ
    y_test : np.ndarray
        テストラベルデータ
    info : dict
        データセット情報（ラベル名など）
    data_flag : str
        データセット名（ファイル名用）
    n_samples : int
        サンプル数（デフォルト: 10）
        
    Returns
    -------
    str
        作成されたZIPファイルのパス（/tmp/{data_flag}_samples.zip）
    """
    from PIL import Image
    import zipfile
    
    # サンプル画像を保存するディレクトリ
    sample_dir = '/tmp/sample_images'
    os.makedirs(sample_dir, exist_ok=True)
    
    # 既存ファイルをクリア
    for f in os.listdir(sample_dir):
        os.remove(os.path.join(sample_dir, f))
    
    # ハイブリッドサンプリング: 各クラス最低1枚 + 残りはランダム
    selected_indices = []
    
    # 各クラスから1枚ずつ選択
    unique_classes = np.unique(y_test)
    for cls in unique_classes:
        cls_indices = np.where(y_test == cls)[0]
        selected_indices.append(np.random.choice(cls_indices))
    
    # 残りをランダムに選択（重複なし）
    remaining = n_samples - len(selected_indices)
    if remaining > 0:
        available_indices = np.setdiff1d(np.arange(len(x_test)), selected_indices)
        additional = np.random.choice(available_indices, size=remaining, replace=False)
        selected_indices.extend(additional)
    
    # 画像をJPEGとして保存
    saved_files = []
    for i, idx in enumerate(selected_indices):
        # 0-1のfloatを0-255のuint8に変換
        img_array = (x_test[idx] * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        # ファイル名にラベル情報を含める
        if data_flag == 'chestmnist':
            label = 'abnormal' if y_test[idx] == 1 else 'normal'
        else:
            label_idx = int(y_test[idx])
            label = info['label'][str(label_idx)]
        
        # ファイル名をシンプルに（特殊文字を除去、2桁連番）
        safe_label = label.replace(' ', '_').replace('-', '_')[:20]
        filename = f'sample_{i+1:02d}_{safe_label}.jpg'
        filepath = os.path.join(sample_dir, filename)
        
        img.save(filepath, 'JPEG')
        saved_files.append(filepath)
        print(f"作成: {filename} (ラベル: {label})")
    
    # ZIPファイルにまとめる
    zip_path = '/tmp/sample_images.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for filepath in saved_files:
            zipf.write(filepath, os.path.basename(filepath))
    
    print(f"\n✓ サンプル画像を作成しました（{len(saved_files)}枚、各クラス最低1枚含む）")

    return zip_path


# ==========================================
# 7. 評価指標・ROC曲線ロジック
# ==========================================
def plot_roc_curve(model, x_test, y_test, title="ROC Curve"):
    """
    ROC曲線を描画し、AUCと最適閾値を返す。

    【技術解説】
    ROC曲線（Receiver Operating Characteristic curve）は、二値分類器の
    性能を閾値に依存しない形で評価するためのグラフである。

    - X軸: 偽陽性率（FPR = 1 - 特異度）
    - Y軸: 真陽性率（TPR = 感度）
    - 対角線（点線）: ランダム分類器（AUC = 0.5）
    - 左上に近いほど性能が良い

    AUC（Area Under the Curve）は曲線下の面積で、0.5〜1.0の範囲を取る。
    - AUC = 1.0: 完璧な分類
    - AUC = 0.5: ランダム分類と同等
    - AUC < 0.5: ランダムより悪い（ラベルが逆転している可能性）

    【医療での解釈目安】
    - 0.9-1.0: 非常に良い（Excellent）
    - 0.8-0.9: 良い（Good）
    - 0.7-0.8: まずまず（Fair）
    - 0.6-0.7: 不十分（Poor）
    - 0.5-0.6: 失敗（Fail）

    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル（二値分類）
    x_test : np.ndarray
        テスト画像データ
    y_test : np.ndarray
        テストラベル（0または1）
    title : str
        グラフのタイトル

    Returns
    -------
    dict
        {
            'auc': AUCスコア,
            'fpr': 偽陽性率の配列,
            'tpr': 真陽性率の配列,
            'thresholds': 閾値の配列,
            'optimal_threshold': Youden Indexに基づく最適閾値,
            'optimal_point': (最適FPR, 最適TPR)
        }
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    # 予測確率を取得
    y_pred_prob = model.predict(x_test, verbose=0)

    # 形状を調整（二値分類の場合）
    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] == 1:
        y_pred_prob = y_pred_prob.flatten()

    y_true = y_test.flatten()

    # ROC曲線の計算
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)

    # 最適閾値の計算（Youden Index: TPR - FPR の最大化）
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    optimal_fpr = fpr[optimal_idx]
    optimal_tpr = tpr[optimal_idx]

    # 描画
    plt.figure(figsize=(8, 8))

    # ROC曲線
    plt.plot(fpr, tpr, color='#2196F3', lw=2,
             label=f'ROC Curve (AUC = {auc_score:.3f})')

    # ランダム分類器の線
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--',
             label='Random Classifier (AUC = 0.5)')

    # 最適点をプロット
    plt.scatter([optimal_fpr], [optimal_tpr], color='#F44336', s=100,
                zorder=5, label=f'Optimal Point (threshold={optimal_threshold:.3f})')

    # 最適点から軸への補助線
    plt.plot([optimal_fpr, optimal_fpr], [0, optimal_tpr],
             color='#F44336', linestyle=':', alpha=0.5)
    plt.plot([0, optimal_fpr], [optimal_tpr, optimal_tpr],
             color='#F44336', linestyle=':', alpha=0.5)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    # AUCの領域を塗りつぶし
    plt.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')

    plt.tight_layout()
    plt.show()

    return {
        'auc': auc_score,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'optimal_threshold': optimal_threshold,
        'optimal_point': (optimal_fpr, optimal_tpr)
    }


def plot_precision_recall_curve(model, x_test, y_test, title="Precision-Recall Curve"):
    """
    Precision-Recall曲線を描画する。

    【技術解説】
    Precision-Recall曲線は、特にクラス不均衡がある場合にROC曲線よりも
    有用な評価指標となることがある。

    - Precision（適合率）: 陽性と予測したもののうち、実際に陽性の割合
      → 「陽性予測の信頼性」を示す
    - Recall（再現率/感度）: 実際の陽性のうち、陽性と予測できた割合
      → 「陽性の取りこぼし」を示す

    【ROC曲線との使い分け】
    - ROC曲線: クラスバランスが取れている場合に適切
    - PR曲線: 陽性クラスが少ない（不均衡）場合に適切

    医療では「異常（陽性）」が少ないことが多いため、PR曲線が有用な場面が多い。

    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル（二値分類）
    x_test : np.ndarray
        テスト画像データ
    y_test : np.ndarray
        テストラベル（0または1）
    title : str
        グラフのタイトル

    Returns
    -------
    dict
        {
            'precision': Precisionの配列,
            'recall': Recallの配列,
            'thresholds': 閾値の配列,
            'average_precision': Average Precision（PR曲線下の面積）
        }
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    # 予測確率を取得
    y_pred_prob = model.predict(x_test, verbose=0)

    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] == 1:
        y_pred_prob = y_pred_prob.flatten()

    y_true = y_test.flatten()

    # PR曲線の計算
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    ap_score = average_precision_score(y_true, y_pred_prob)

    # ベースライン（ランダム分類器）= 陽性クラスの割合
    baseline = y_true.sum() / len(y_true)

    # 描画
    plt.figure(figsize=(8, 8))

    # PR曲線
    plt.plot(recall, precision, color='#4CAF50', lw=2,
             label=f'PR Curve (AP = {ap_score:.3f})')

    # ベースライン
    plt.axhline(y=baseline, color='gray', linestyle='--',
                label=f'Baseline (Prevalence = {baseline:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision (PPV)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)

    plt.fill_between(recall, precision, alpha=0.1, color='#4CAF50')

    plt.tight_layout()
    plt.show()

    return {
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds,
        'average_precision': ap_score
    }


def find_optimal_threshold(model, x_test, y_test, strategy='youden',
                           target_sensitivity=None, target_specificity=None):
    """
    指定された戦略に基づいて最適な判定閾値を探索する。

    【技術解説】
    閾値の選択は、誤分類のコストに基づいて行うべきである。

    戦略の選択指針:
    - 'youden': 感度と特異度のバランスを取りたい場合
    - 'sensitivity': 見逃しを最小化したい場合（スクリーニング検査）
    - 'specificity': 偽陽性を最小化したい場合（確定診断）
    - 'f1': Precision/Recallのバランスを重視する場合

    【医療での典型的な使い分け】
    - スクリーニング: 高感度（見逃し防止）→ target_sensitivity=0.95
    - 確定診断: 高特異度（誤診防止）→ target_specificity=0.95

    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    x_test : np.ndarray
        テスト画像データ
    y_test : np.ndarray
        テストラベル
    strategy : str
        'youden': Youden Index最大化（デフォルト）
        'sensitivity': 目標感度を達成する最小閾値
        'specificity': 目標特異度を達成する最大閾値
        'f1': F1スコア最大化
    target_sensitivity : float, optional
        strategy='sensitivity'の場合の目標感度（例: 0.95）
    target_specificity : float, optional
        strategy='specificity'の場合の目標特異度（例: 0.95）

    Returns
    -------
    dict
        {
            'threshold': 最適閾値,
            'sensitivity': その閾値での感度,
            'specificity': その閾値での特異度,
            'strategy': 使用した戦略
        }
    """
    from sklearn.metrics import roc_curve

    y_pred_prob = model.predict(x_test, verbose=0)
    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] == 1:
        y_pred_prob = y_pred_prob.flatten()
    y_true = y_test.flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    specificity = 1 - fpr

    if strategy == 'youden':
        # Youden Index = TPR - FPR = 感度 + 特異度 - 1
        youden = tpr - fpr
        optimal_idx = np.argmax(youden)

    elif strategy == 'sensitivity':
        if target_sensitivity is None:
            target_sensitivity = 0.95
        # 目標感度以上を達成する閾値のうち、最も特異度が高いもの
        valid_indices = np.where(tpr >= target_sensitivity)[0]
        if len(valid_indices) == 0:
            print(f"[警告] 感度 {target_sensitivity} を達成できません。最大感度の閾値を使用します。")
            optimal_idx = np.argmax(tpr)
        else:
            # 特異度が最大のインデックスを選択
            optimal_idx = valid_indices[np.argmax(specificity[valid_indices])]

    elif strategy == 'specificity':
        if target_specificity is None:
            target_specificity = 0.95
        # 目標特異度以上を達成する閾値のうち、最も感度が高いもの
        valid_indices = np.where(specificity >= target_specificity)[0]
        if len(valid_indices) == 0:
            print(f"[警告] 特異度 {target_specificity} を達成できません。最大特異度の閾値を使用します。")
            optimal_idx = np.argmax(specificity)
        else:
            # 感度が最大のインデックスを選択
            optimal_idx = valid_indices[np.argmax(tpr[valid_indices])]

    elif strategy == 'f1':
        # F1スコアを各閾値で計算
        precision = np.zeros_like(tpr)
        for i in range(len(thresholds)):
            tp = tpr[i] * y_true.sum()
            fp = fpr[i] * (len(y_true) - y_true.sum())
            if tp + fp > 0:
                precision[i] = tp / (tp + fp)
            else:
                precision[i] = 0
        f1 = 2 * (precision * tpr) / (precision + tpr + 1e-8)
        optimal_idx = np.argmax(f1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = specificity[optimal_idx]

    return {
        'threshold': optimal_threshold,
        'sensitivity': optimal_sensitivity,
        'specificity': optimal_specificity,
        'strategy': strategy
    }


def evaluate_at_threshold(model, x_test, y_test, threshold=0.5):
    """
    指定された閾値での全評価指標を計算・表示する。

    【技術解説】
    二値分類の評価指標:

    混同行列から導出される指標:
    - TP (True Positive): 正しく陽性と予測
    - TN (True Negative): 正しく陰性と予測
    - FP (False Positive): 誤って陽性と予測（偽陽性）
    - FN (False Negative): 誤って陰性と予測（偽陰性）

    基本指標:
    - 感度 (Sensitivity) = TP / (TP + FN): 陽性を正しく検出する割合
    - 特異度 (Specificity) = TN / (TN + FP): 陰性を正しく検出する割合
    - 正解率 (Accuracy) = (TP + TN) / 全体
    - 適合率 (Precision/PPV) = TP / (TP + FP)
    - 陰性的中率 (NPV) = TN / (TN + FN)
    - F1スコア = 2 * Precision * Recall / (Precision + Recall)

    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    x_test : np.ndarray
        テスト画像データ
    y_test : np.ndarray
        テストラベル
    threshold : float
        判定閾値（デフォルト: 0.5）

    Returns
    -------
    dict
        全評価指標を含む辞書
    """
    from sklearn.metrics import confusion_matrix

    y_pred_prob = model.predict(x_test, verbose=0)
    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] == 1:
        y_pred_prob = y_pred_prob.flatten()
    y_true = y_test.flatten()

    # 閾値で二値化
    y_pred = (y_pred_prob >= threshold).astype(int)

    # 混同行列
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # 各指標の計算
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    youden = sensitivity + specificity - 1

    # 結果の表示
    print("=" * 50)
    print(f"評価結果（閾値 = {threshold:.3f}）")
    print("=" * 50)
    print()
    print("【混同行列】")
    print(f"                予測:陰性    予測:陽性")
    print(f"  実際:正常      TN={tn:5d}    FP={fp:5d}")
    print(f"  実際:異常      FN={fn:5d}    TP={tp:5d}")
    print()
    print("【評価指標】")
    print(f"  感度 (Sensitivity/Recall) : {sensitivity:.4f}  ← 異常の検出率")
    print(f"  特異度 (Specificity)      : {specificity:.4f}  ← 正常の正答率")
    print(f"  正解率 (Accuracy)         : {accuracy:.4f}")
    print(f"  適合率 (Precision/PPV)    : {precision:.4f}  ← 陽性予測の信頼度")
    print(f"  陰性的中率 (NPV)          : {npv:.4f}  ← 陰性予測の信頼度")
    print(f"  F1スコア                  : {f1:.4f}")
    print(f"  Youden Index              : {youden:.4f}")
    print("=" * 50)

    # 混同行列の可視化
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Normal', 'Abnormal'])
    plt.title(f'Confusion Matrix (threshold={threshold:.3f})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

    return {
        'threshold': threshold,
        'confusion_matrix': cm,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'precision': precision,
        'npv': npv,
        'f1': f1,
        'youden': youden
    }


def compare_thresholds(model, x_test, y_test, thresholds=[0.3, 0.5, 0.7]):
    """
    複数の閾値での評価指標を比較する。

    【技術解説】
    閾値の違いによる性能変化を一覧表示することで、
    適切な閾値選択の判断材料を提供する。

    Parameters
    ----------
    model : keras.Model
        学習済みのKerasモデル
    x_test : np.ndarray
        テスト画像データ
    y_test : np.ndarray
        テストラベル
    thresholds : list
        比較する閾値のリスト

    Returns
    -------
    list
        各閾値での評価結果のリスト
    """
    from sklearn.metrics import confusion_matrix

    y_pred_prob = model.predict(x_test, verbose=0)
    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] == 1:
        y_pred_prob = y_pred_prob.flatten()
    y_true = y_test.flatten()

    results = []

    print("=" * 80)
    print("閾値による評価指標の比較")
    print("=" * 80)
    print(f"{'閾値':^8} {'感度':^10} {'特異度':^10} {'正解率':^10} {'適合率':^10} {'F1':^10}")
    print("-" * 80)

    for thresh in thresholds:
        y_pred = (y_pred_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

        print(f"{thresh:^8.2f} {sens:^10.4f} {spec:^10.4f} {acc:^10.4f} {prec:^10.4f} {f1:^10.4f}")

        results.append({
            'threshold': thresh,
            'sensitivity': sens,
            'specificity': spec,
            'accuracy': acc,
            'precision': prec,
            'f1': f1
        })

    print("=" * 80)

    return results
