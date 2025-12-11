> このドキュメントは、書籍スキャン画像の補正作業を自動化するための、検証済みで安全なワークフローを解説します。

# 書籍スキャン補正 実装ガイド

このガイドでは、以下の3つのステップからなる高精度な補正ワークフローの具体的な実行方法を解説します。

1.  **前処理 (Unpaper)**: 両開きスキャン画像のページ分割、傾き補正、黒縁除去
2.  **歪み補正 (py-reform)**: ページの湾曲や歪みを最新のAIモデルで補正
3.  **指除去 (OpenCV)**: 画像に写り込んだ指を手動または半自動で除去

## 0. 事前準備

macOS環境で、以下のツールをインストールします。

### 1. Homebrew

macOS用のパッケージマネージャーです。ターミナルで以下のコマンドを実行してインストールします。

```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Python 3.11+

Homebrewを使ってPythonをインストールします。

```shell
brew install python
```

### 3. Unpaper

Homebrewを使ってUnpaperをインストールします。

```shell
brew install unpaper
```

### 4. 必要なPythonライブラリ

`pip`コマンドで、歪み補正と画像処理に必要なライブラリをインストールします。

```shell
pip3 install py-reform opencv-python numpy
```

--- 

## 方法1: 完全自動化スクリプト (`complete_workflow.py`)

提供された`complete_workflow.py`スクリプトを使うと、コマンド一つで前処理から歪み補正、指除去までを一貫して実行できます。

### 基本的な使い方

```shell
# 単一ページのスキャン画像を補正
python3 complete_workflow.py --input scan.jpg --output result.jpg
```

### 両開きページを分割して補正

`--layout double`オプションを付けると、1枚の画像に含まれる左右2ページを自動で分割し、それぞれを補正します。

```shell
# 両開きスキャンを2つのファイル（page_1.jpg, page_2.jpg）に出力
python3 complete_workflow.py --input book.jpg --output "page_%d.jpg" --layout double
```

### 指除去を含めて補正

`--mask`オプションで指の部分を白く塗りつぶしたマスク画像を指定すると、指除去（Inpainting）も同時に実行されます。

> **重要**: マスク画像は、GIMPやPhotoshopなどの画像編集ソフトを使い、**指の部分を白、それ以外を黒**で塗りつぶして作成してください。

```shell
# マスク画像（finger_mask.png）を使って指を除去
python3 complete_workflow.py --input scan.jpg --output result.jpg --mask finger_mask.png
```

### M4 ProのGPUを活用する

`--device mps`オプションを指定すると、Apple SiliconのGPU（Metal Performance Shaders）を使って`py-reform`の処理を高速化できます。

```shell
python3 complete_workflow.py --input scan.jpg --output result.jpg --device mps
```

--- 

## 方法2: インタラクティブ指除去ツール (`interactive_inpaint.py`)

マスク画像を手動で作成するのが面倒な場合は、`interactive_inpaint.py`ツールが便利です。マウスで指の部分をなぞるだけで、簡単に除去できます。

### 使い方

1.  **ツールの起動**

    ```shell
    python3 interactive_inpaint.py scan_with_finger.jpg
    ```

2.  **マスクの作成**

    -   画像ウィンドウ上で、除去したい指の部分を**マウスでドラッグ**して緑色に塗りつぶします。
    -   `+` / `-` キーでブラシのサイズを調整できます。

3.  **Inpaintingの実行**

    -   `i`キーを押すと、修復処理が実行され、結果が「Result」ウィンドウに表示されます。

4.  **結果の保存**

    -   `s`キーを押すと、`inpaint_result.jpg`という名前で結果が保存されます。

### 操作キー一覧

| キー | 機能 |
| :--- | :--- |
| マウスドラッグ | 修復したい領域を塗りつぶす |
| `i` | 修復を実行（高速なTELEA法） |
| `n` | 修復を実行（高品質なNS法） |
| `r` | リセット（塗りつぶしをクリア） |
| `s` | 結果を`inpaint_result.jpg`に保存 |
| `+` / `=` | ブラシサイズを大きく |
| `-` / `_` | ブラシサイズを小さく |
| `q` | ツールを終了 |

--- 

## 方法3: 手動ステップ・バイ・ステップ

各ツールを個別に実行し、処理内容を細かく制御したい場合は、以下の手順に従ってください。

### ステップ1: Unpaperで前処理

両開きのスキャン画像 `book.png` を2つの単一ページ `page_1.png`, `page_2.png` に分割し、傾きを補正します。

```shell
unpaper --layout double book.png page_%d.png
```

### ステップ2: py-reformで歪み補正

Unpaperで出力された `page_1.png` の歪みを補正し、`dewarped.png` として保存します。

```python
# dewarp.py
from py_reform import straighten

# M4 ProのGPUを使う場合は device='mps' を指定
straight_image = straighten("page_1.png", device="mps")
straight_image.save("dewarped.png")
```

### ステップ3: OpenCVで指除去

`dewarped.png` に写り込んだ指を、マスク画像 `mask.png` を使って除去します。

```python
# inpaint.py
import cv2

img = cv2.imread("dewarped.png")
mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

# inpaintingを実行
result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

cv2.imwrite("final_result.png", result)
```
