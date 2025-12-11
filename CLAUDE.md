# Book Scan Workflow

書籍スキャン画像の補正ワークフローを自動化するPythonプロジェクト。

## プロジェクト概要

両開きスキャン画像のページ分割、傾き補正、歪み補正、指除去を行う3ステップのワークフロー。

## 技術スタック

- **Python 3.11+**
- **unpaper** - ページ分割・傾き補正（外部コマンド）
- **py-reform** - AI歪み補正（uvdocモデル）
- **OpenCV** - Inpainting（指除去）
- **MediaPipe** - 自動指検出（オプション）
- **LaMa** - 高品質インペインティング（オプション）
- **NumPy** - 画像処理

## ファイル構成

```
complete_workflow.py   # メインワークフロー（全ステップ統合）
auto_finger_remove.py  # 自動指検出・除去ツール（単体実行可）
interactive_inpaint.py # マウス操作の指除去ツール
manual_steps.py        # ステップ別手動実行スクリプト
test_workflow.py       # 環境テストスクリプト
```

## 主要コマンド

```bash
# 環境テスト
python3 test_workflow.py

# 基本的な補正
python3 complete_workflow.py --input scan.jpg --output result.jpg

# 両開きページ分割
python3 complete_workflow.py --input book.jpg --output "page_%d.jpg" --layout double

# 自動指検出・除去（推奨）
python3 complete_workflow.py --input scan.jpg --output result.jpg --auto-finger

# 自動指除去（高品質LaMa使用）
python3 complete_workflow.py --input scan.jpg --output result.jpg --auto-finger --inpaint-method lama

# Apple Silicon GPU使用
python3 complete_workflow.py --input scan.jpg --output result.jpg --device mps

# 自動指除去ツール単体実行
python3 auto_finger_remove.py scan.jpg
python3 auto_finger_remove.py scan.jpg --method lama --save-mask

# インタラクティブ指除去（手動マスク作成）
python3 interactive_inpaint.py image.jpg
```

## 処理フロー

1. **Unpaper**: ページ分割、傾き補正、黒縁除去
2. **py-reform**: 湾曲・歪み補正（uvdoc/deskewモデル）
3. **指除去**: 自動検出（MediaPipe）または手動マスク（OpenCV/LaMa）

## 依存ライブラリ

```bash
# 必須
pip install py-reform opencv-python numpy

# 自動指検出用（推奨）
pip install mediapipe

# 高品質インペインティング用（オプション）
pip install simple-lama-inpainting
```

## 開発時の注意

- `--device mps`でApple Silicon GPU加速可能
- `--auto-finger`で自動指検出・除去（MediaPipe必要）
- `--inpaint-method lama`で高品質修復（simple-lama-inpainting必要）
- マスク画像は「指=白、それ以外=黒」で作成（手動時）
- 一時ファイルは`/tmp/book_scan_workflow/`に出力
- 日本語でコメント・ドキュメント記載

## py-reformの制限事項

- 複雑な折り目や強いシワには弱い
- ページごとに結果が不安定な場合あり
- エラー時は`errors="warn"`で元画像にフォールバック推奨
