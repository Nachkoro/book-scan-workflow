# 書籍スキャン補正ワークフロー - 使い方ガイド

## 概要

このワークフローは、書籍スキャン画像の補正作業を自動化するためのツールセットです。以下の3つのステップから構成されています：

1. **前処理 (Unpaper)**: 両開きスキャン画像のページ分割、傾き補正、黒縁除去
2. **歪み補正 (py-reform)**: ページの湾曲や歪みを最新のAIモデルで補正
3. **指除去 (OpenCV)**: 画像に写り込んだ指を手動または半自動で除去

## 環境要件

- macOS
- Python 3.11+
- Homebrew
- 必要なツール: unpaper, py-reform, opencv-python, numpy, Pillow
- 自動指除去を利用する場合: mediapipe, （推奨）simple-lama-inpainting

## インストール

```shell
# Python 3.11のインストール
brew install python@3.11

# Unpaperのインストール
brew install unpaper

# Pythonライブラリのインストール
python3.11 -m pip install py-reform opencv-python numpy Pillow mediapipe simple-lama-inpainting
```

## ツール一覧

### 1. 完全自動化スクリプト (`complete_workflow.py`)

すべての処理を一度に実行するメインスクリプトです。

#### 基本的な使い方

```shell
# 単一ページの補正
python3.11 complete_workflow.py --input scan.jpg --output result.jpg

# 両開きページを分割して補正
python3.11 complete_workflow.py --input book.jpg --output "page_%d.jpg" --layout double

# M4 Pro GPUを使用して高速化
python3.11 complete_workflow.py --input scan.jpg --output result.jpg --device mps
```

#### 指除去を含める場合

```shell
# 自動指検出＋LaMaで高品質除去
python3.11 complete_workflow.py --input scan.jpg --output result.jpg --auto-finger --inpaint-method lama

# マスク画像を指定して指除去を実行（従来方式）
python3.11 complete_workflow.py --input scan.jpg --output result.jpg --mask finger_mask.png
```

**マスク画像の作成方法**:
- GIMPやPhotoshopなどの画像編集ソフトを使用
- 指の部分を**白**、それ以外を**黒**で塗りつぶす

> **備考**: JPEGやHEICなどUnpaper非対応形式を指定した場合でも、自動でPNGに変換してから処理します。

### 2. インタラクティブ指除去ツール (`interactive_inpaint.py`)

マウスで指の部分をなぞるだけで簡単に指を除去できます。

```shell
python3.11 interactive_inpaint.py scan_with_finger.jpg
```

#### 操作方法

| キー | 機能 |
|-----|------|
| マウスドラッグ | 修復したい領域を塗りつぶす |
| `i` | 修復を実行（高速なTELEA法） |
| `n` | 修復を実行（高品質なNS法） |
| `r` | リセット（塗りつぶしをクリア） |
| `s` | 結果を`inpaint_result.jpg`に保存 |
| `+` / `=` | ブラシサイズを大きく |
| `-` / `_` | ブラシサイズを小さく |
| `q` | ツールを終了 |

### 3. 手動ステップ・バイ・ステップ (`manual_steps.py`)

各処理を個別に実行したい場合に使用します。

```shell
# ステップ1: Unpaperで前処理
python3.11 manual_steps.py --step 1 --input book.jpg --output page_%d.png --layout double

# ステップ2: py-reformで歪み補正
python3.11 manual_steps.py --step 2 --input page_1.png --output dewarped.png --device mps

# ステップ3: 指除去
python3.11 manual_steps.py --step 3 --input dewarped.png --mask mask.png --output final.png
```

### 4. 環境テストスクリプト (`test_workflow.py`)

インストールが正しく完了したかを確認します。

```shell
python3.11 test_workflow.py
```

## 推奨ワークフロー

### 基本的な書籍スキャン補正

1. **両開きスキャン画像の補正**:
   ```shell
   python3.11 complete_workflow.py --input book_scan.jpg --output "corrected_page_%d.jpg" --layout double --device mps
   ```

2. **指がある場合**:
   ```shell
   # まずインタラクティブツールでマスクを作成
   python3.11 interactive_inpaint.py corrected_page_1.jpg
   
   # または手動でマスクを作成してから
   python3.11 complete_workflow.py --input corrected_page_1.jpg --output final_page_1.jpg --mask finger_mask.png
   ```

### 高品質な処理を行う場合

1. **各ステップを個別に実行して品質を確認**:
   ```shell
   # ステップ1: 前処理
   python3.11 manual_steps.py --step 1 --input book.jpg --output page_%d.png --layout double
   
   # ステップ2: 歪み補正（結果を確認）
   python3.11 manual_steps.py --step 2 --input page_1.png --output dewarped.png --device mps
   
   # ステップ3: 指除去（必要な場合）
   python3.11 interactive_inpaint.py dewarped.png
   ```

## トラブルシューティング

### よくある問題

1. **Unpaperが見つからない**:
   ```shell
   brew install unpaper
   ```

2. **Pythonライブラリが見つからない**:
   ```shell
   python3.11 -m pip install py-reform opencv-python numpy
   ```

3. **GPUが使用できない**:
   - `--device cpu` に切り替えてください
   - M4 Pro/M2/M3 Macの場合は `--device mps` を使用

4. **メモリ不足**:
   - 画像サイズを小さくする
   - `--device cpu` を使用する

### パフォーマンスの最適化

- **M4 Pro/M2/M3 Mac**: `--device mps` を使用してGPUアクセラレーション
- **Intel Mac**: `--device cpu` を使用
- **大量の画像**: ステップ・バイ・ステップで個別に処理

## ファイル形式

- **対応入力形式**: JPG, PNG, TIFF, BMP
- **対応出力形式**: JPG, PNG
- **マスク画像**: グレースケールPNG（白が修復対象）

## ライセンス

このツールセットは、各コンポーネントのライセンスに従います：
- Unpaper: GPL v2
- py-reform: MIT License
- OpenCV: Apache License 2.0
- 本スクリプト: MIT License
