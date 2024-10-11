# Fish Species Classification with Deep Learning

This project involves building a deep learning model to classify different species of fish using the dataset available on Kaggle named "Fish Dataset". The main objective of this project is to train a convolutional neural network (CNN) to accurately identify fish species from images. The project utilizes TensorFlow, Keras, and Transfer Learning with the MobileNetV2 model for fine-tuning.

## Dataset

The dataset used in this project is the "Fish Dataset" from Kaggle, which contains images of 31 different species of fish. The data is split into training, validation, and test sets with the following structure:
- Training set: 7047 images belonging to 31 classes
- Validation set: 1744 images belonging to 31 classes (split from training data)
- Test set: 1760 images belonging to 31 classes

The dataset is available for download from Kaggle. You will need your Kaggle API key (`kaggle.json`) to download it.

## Model Training and Architecture

### Initial CNN Model
An initial convolutional neural network (CNN) was constructed with the following architecture:
- Three convolutional layers with increasing filter sizes (32, 64, and 128) and ReLU activation functions.
- MaxPooling layers to reduce spatial dimensions.
- A fully connected layer with 512 neurons, followed by a Dropout layer to reduce overfitting.
- An output layer with 31 neurons (for each fish class) using softmax activation.

The initial model was trained for 20 epochs with an Adam optimizer and categorical cross-entropy loss. The model achieved around 84% accuracy on the test set.

### Transfer Learning with MobileNetV2
To further improve model performance, transfer learning was applied using the MobileNetV2 pre-trained model. The following steps were performed:
- The MobileNetV2 base was used without the top layers, and its weights were initially frozen.
- A new classification head was added consisting of a global average pooling layer, a dense layer with 512 units, and a Dropout layer.
- The model was trained for 10 epochs with the MobileNetV2 base frozen.
- The model was then fine-tuned by unfreezing the last few layers of MobileNetV2, allowing the model to adapt better to the fish dataset. This fine-tuning stage was also trained for 10 epochs.

### Model Performance
The final fine-tuned model achieved:
- Test Accuracy: 89%
- Test Loss: 0.40

## Evaluation
A confusion matrix was generated to understand the classification performance for each fish class. The project also includes a classification report detailing precision, recall, and F1-score for each of the 31 fish species, providing insights into which species are easier or harder to classify.

## How to Use This Project

1. **Set Up Kaggle API**: Place your `kaggle.json` file in the appropriate directory to download the dataset.
   ```sh
   !mkdir -p ~/.kaggle
   !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Download Dataset**: Use the Kaggle API to download the dataset.
   ```sh
   !kaggle datasets download -d markdaniellampa/fish-dataset
   ```

3. **Extract Dataset**: Extract the downloaded dataset.
   ```python
   import zipfile
   with zipfile.ZipFile('fish-dataset.zip', 'r') as zip_ref:
       zip_ref.extractall('/content/fish_dataset')
   ```

4. **Model Training**: Train the model using the provided training script in the project.

5. **Evaluation**: Evaluate the model on the test data and view the confusion matrix and classification report for a detailed analysis.

6. **Download Trained Model**: The trained model can be saved and downloaded as a `.keras` file for future use.
   ```python
   from google.colab import files
   files.download('/content/fish_classification_model_finetuned.keras')
   ```

## Requirements
- Python 3.x
- TensorFlow and Keras
- Kaggle API
- NumPy, Matplotlib, Seaborn
- scikit-learn for classification report and confusion matrix

## Results and Conclusion
The project successfully utilized deep learning to classify fish species with a final accuracy of 89%. The use of transfer learning with MobileNetV2 significantly improved the model's performance. The model is now capable of classifying fish species with a good level of accuracy, and further improvements could include data augmentation, hyperparameter tuning, or adding more sophisticated layers to the CNN.

## Future Work
- Implementing more advanced augmentation techniques to improve generalization.
- Using ensemble models to improve accuracy.
- Applying hyperparameter tuning to optimize the MobileNetV2 model further.

  # 魚種分類のディープラーニング

このプロジェクトでは、Kaggleで利用可能な「Fish Dataset」を使用して、異なる魚種を分類するためのディープラーニングモデルを構築します。このプロジェクトの主な目的は、畳み込みニューラルネットワーク（CNN）をトレーニングして、魚種を画像から正確に識別することです。プロジェクトでは、TensorFlow、Keras、およびMobileNetV2モデルを用いた転移学習を利用しています。

## データセット

このプロジェクトで使用されるデータセットはKaggleの「Fish Dataset」で、31種類の魚の画像が含まれています。データは以下のようにトレーニング、検証、テストセットに分割されています：
- トレーニングセット：31クラスに属する7047枚の画像
- 検証セット：31クラスに属する1744枚の画像（トレーニングデータから分割）
- テストセット：31クラスに属する1760枚の画像

データセットはKaggleからダウンロード可能です。ダウンロードにはKaggle APIキー（`kaggle.json`）が必要です。

## モデルのトレーニングとアーキテクチャ

### 初期CNNモデル
初期の畳み込みニューラルネットワーク（CNN）は以下のアーキテクチャで構築されました：
- フィルタサイズが増加する3つの畳み込み層（32、64、128）とReLU活性化関数。
- 空間次元を減らすためのMaxPooling層。
- 過学習を防ぐためのDropout層を持つ512ニューロンの全結合層。
- 各魚種に対応する31ニューロンを持つ出力層（softmax活性化関数）。

初期モデルはAdamオプティマイザとカテゴリカルクロスエントロピー損失を使用して20エポックでトレーニングされ、テストセットで約84%の精度を達成しました。

### MobileNetV2を用いた転移学習
モデルの性能をさらに向上させるため、事前学習済みのMobileNetV2モデルを使用して転移学習を行いました。以下のステップを実施しました：
- MobileNetV2のベースを使用し、トップ層は除去して重みを初期的に固定。
- グローバル平均プーリング層、512ユニットの全結合層、Dropout層からなる新しい分類ヘッドを追加。
- MobileNetV2のベースを固定した状態で10エポックのトレーニングを実施。
- MobileNetV2の最後の数層を解凍し、魚データセットに適応するように微調整。この微調整フェーズも10エポックでトレーニング。

### モデルの性能
最終的な微調整モデルの性能：
- テスト精度：89%
- テスト損失：0.40

## 評価
各魚種に対する分類性能を理解するために混同行列を生成しました。プロジェクトには、31種の魚それぞれに対する精度、再現率、F1スコアを詳述した分類レポートも含まれており、どの魚種が分類しやすく、どの魚種が分類しにくいかについての洞察が得られます。

## プロジェクトの使い方

1. **Kaggle APIの設定**: `kaggle.json`ファイルを適切なディレクトリに配置してデータセットをダウンロードします。
   ```sh
   !mkdir -p ~/.kaggle
   !cp /content/drive/MyDrive/kaggle.json ~/.kaggle/
   !chmod 600 ~/.kaggle/kaggle.json
   ```

2. **データセットのダウンロード**: Kaggle APIを使用してデータセットをダウンロードします。
   ```sh
   !kaggle datasets download -d markdaniellampa/fish-dataset
   ```

3. **データセットの解凍**: ダウンロードしたデータセットを解凍します。
   ```python
   import zipfile
   with zipfile.ZipFile('fish-dataset.zip', 'r') as zip_ref:
       zip_ref.extractall('/content/fish_dataset')
   ```

4. **モデルのトレーニング**: プロジェクト内のトレーニングスクリプトを使用してモデルをトレーニングします。

5. **評価**: テストデータでモデルを評価し、混同行列と分類レポートを表示して詳細な分析を行います。

6. **トレーニング済みモデルのダウンロード**: トレーニング済みモデルを`.keras`ファイルとして保存し、将来の使用のためにダウンロードします。
   ```python
   from google.colab import files
   files.download('/content/fish_classification_model_finetuned.keras')
   ```

## 必要条件
- Python 3.x
- TensorFlowおよびKeras
- Kaggle API
- NumPy、Matplotlib、Seaborn
- scikit-learn（分類レポートと混同行列のため）

## 結果と結論
このプロジェクトは、ディープラーニングを用いて魚種を分類することに成功し、最終的な精度は89%に達しました。MobileNetV2を用いた転移学習により、モデルの性能が大幅に向上しました。現在、このモデルは良好な精度で魚種を分類することができ、さらに改善するためにはデータ拡張、ハイパーパラメータのチューニング、またはCNNにより高度な層を追加することが考えられます。

## 今後の作業
- より高度なデータ拡張技術を導入して汎化性能を向上させる。
- アンサンブルモデルを使用して精度を向上させる。
- MobileNetV2モデルを最適化するためにハイパーパラメータのチューニングを行う。

