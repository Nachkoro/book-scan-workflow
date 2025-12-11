#!/usr/bin/env python3
"""
書籍スキャン補正の完全なワークフロー

使用方法:
    python complete_workflow.py --input scan.jpg --output result.jpg

ステップ:
1. Unpaper: ページ分割・傾き補正（外部コマンド）
2. py-reform: 歪み補正
3. 指除去: 自動検出（MediaPipe）またはマスク指定（OpenCV/LaMa）
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from py_reform import straighten
    import cv2
    import numpy as np
except ImportError as e:
    print(f"エラー: 必要なライブラリがインストールされていません: {e}")
    print("\n以下のコマンドでインストールしてください:")
    print("  pip install py-reform opencv-python numpy")
    sys.exit(1)

# MediaPipe（自動指検出用、オプション）
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

# LaMa（高品質インペインティング用、オプション）
LAMA_AVAILABLE = False
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    pass


def check_unpaper():
    """unpaperがインストールされているか確認"""
    try:
        subprocess.run(['unpaper', '--version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_unpaper(input_path, output_path, layout='single'):
    """
    Unpaperで前処理を実行
    
    Args:
        input_path: 入力画像のパス
        output_path: 出力画像のパス
        layout: 'single' または 'double'
    """
    print(f"[1/3] Unpaper実行中... (layout={layout})")
    
    cmd = ['unpaper', '--layout', layout, str(input_path), str(output_path)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ Unpaper完了: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Unpaperエラー: {e.stderr}")
        return False


def run_py_reform(input_path, output_path, model='uvdoc', device='cpu'):
    """
    py-reformで歪み補正を実行
    
    Args:
        input_path: 入力画像のパス
        output_path: 出力画像のパス
        model: 'uvdoc' または 'deskew'
        device: 'cpu', 'cuda', 'mps'
    """
    print(f"[2/3] py-reform実行中... (model={model}, device={device})")
    
    try:
        # 画像を補正
        result = straighten(str(input_path), model=model, device=device)
        
        # 保存
        result.save(str(output_path))
        print(f"  ✓ py-reform完了: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ py-reformエラー: {e}")
        return False


def detect_fingers(image, margin=30, min_confidence=0.5):
    """
    MediaPipeで指/手を自動検出してマスクを生成

    Args:
        image: BGR画像（numpy配列）
        margin: 検出領域の拡張マージン
        min_confidence: 検出信頼度の閾値

    Returns:
        mask: 指の部分が白（255）のマスク画像
        detected: 検出されたかどうか
    """
    if not MEDIAPIPE_AVAILABLE:
        return None, False

    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=4,
        min_detection_confidence=min_confidence
    )

    try:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return mask, False

        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])

            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            for point in points:
                cv2.circle(mask, tuple(point), margin, 255, -1)

        kernel = np.ones((margin, margin), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask, True
    finally:
        hands.close()


def inpaint_with_lama(image, mask):
    """
    LaMaで高品質インペインティング

    Args:
        image: BGR画像
        mask: マスク画像

    Returns:
        修復された画像
    """
    if not LAMA_AVAILABLE:
        return None

    from PIL import Image

    lama = SimpleLama()
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask)
    result = lama(pil_image, pil_mask)

    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def run_inpainting(input_path, mask_path, output_path, radius=3, method='telea'):
    """
    OpenCV Inpaintingで指除去を実行

    Args:
        input_path: 入力画像のパス
        mask_path: マスク画像のパス
        output_path: 出力画像のパス
        radius: 修復半径
        method: 'telea' または 'ns'
    """
    print(f"[3/3] Inpainting実行中... (method={method}, radius={radius})")

    try:
        # 画像とマスクを読み込み
        img = cv2.imread(str(input_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"  ✗ 画像を読み込めません: {input_path}")
            return False

        if mask is None:
            print(f"  ✗ マスクを読み込めません: {mask_path}")
            return False

        # サイズが一致しているか確認
        if img.shape[:2] != mask.shape[:2]:
            print(f"  ! マスクをリサイズします: {mask.shape[:2]} -> {img.shape[:2]}")
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Inpainting実行
        flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        result = cv2.inpaint(img, mask, radius, flag)

        # 保存
        cv2.imwrite(str(output_path), result)
        print(f"  ✓ Inpainting完了: {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Inpaintingエラー: {e}")
        return False


def run_auto_finger_removal(input_path, output_path, method='auto', margin=30):
    """
    自動指検出・除去を実行

    Args:
        input_path: 入力画像のパス
        output_path: 出力画像のパス
        method: 'auto', 'lama', 'opencv'
        margin: 検出マージン
    """
    print(f"[3/3] 自動指検出・除去実行中... (method={method})")

    if not MEDIAPIPE_AVAILABLE:
        print("  ✗ MediaPipeがインストールされていません")
        print("    pip install mediapipe")
        return False

    try:
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"  ✗ 画像を読み込めません: {input_path}")
            return False

        # 指を検出
        mask, detected = detect_fingers(img, margin=margin)

        if not detected:
            print("  ! 指が検出されませんでした。元の画像を使用します。")
            import shutil
            shutil.copy(input_path, output_path)
            return True

        print("  ✓ 指を検出しました")

        # 修復方法を決定
        if method == 'auto':
            method = 'lama' if LAMA_AVAILABLE else 'opencv'

        # 修復実行
        if method == 'lama' and LAMA_AVAILABLE:
            print("  → LaMaで修復中...")
            result = inpaint_with_lama(img, mask)
            if result is None:
                result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        else:
            print("  → OpenCVで修復中...")
            result = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)

        cv2.imwrite(str(output_path), result)
        print(f"  ✓ 自動指除去完了: {output_path}")
        return True

    except Exception as e:
        print(f"  ✗ 自動指除去エラー: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='書籍スキャン補正の完全なワークフロー',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用方法
  python complete_workflow.py --input scan.jpg --output result.jpg

  # 両開きスキャンを分割
  python complete_workflow.py --input scan.jpg --output page.jpg --layout double

  # 自動指検出・除去（推奨）
  python complete_workflow.py --input scan.jpg --output result.jpg --auto-finger

  # 自動指除去（高品質LaMa使用）
  python complete_workflow.py --input scan.jpg --output result.jpg --auto-finger --inpaint-method lama

  # マスク指定で指除去（従来方式）
  python complete_workflow.py --input scan.jpg --output result.jpg --mask finger_mask.png

  # M4 Pro（Apple Silicon）のGPUを使用
  python complete_workflow.py --input scan.jpg --output result.jpg --device mps
        """
    )

    parser.add_argument('--input', '-i', required=True, type=Path,
                       help='入力画像のパス')
    parser.add_argument('--output', '-o', required=True, type=Path,
                       help='出力画像のパス')
    parser.add_argument('--layout', choices=['single', 'double'], default='single',
                       help='ページレイアウト (default: single)')
    parser.add_argument('--model', choices=['uvdoc', 'deskew'], default='uvdoc',
                       help='歪み補正モデル (default: uvdoc)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                       help='使用デバイス (default: cpu)')
    parser.add_argument('--mask', type=Path,
                       help='指除去用のマスク画像（従来方式）')
    parser.add_argument('--auto-finger', action='store_true',
                       help='自動指検出・除去を有効化（MediaPipe使用）')
    parser.add_argument('--inpaint-method', choices=['auto', 'lama', 'opencv'],
                       default='auto',
                       help='インペインティング方法 (default: auto)')
    parser.add_argument('--skip-unpaper', action='store_true',
                       help='Unpaperをスキップ')
    parser.add_argument('--skip-reform', action='store_true',
                       help='py-reformをスキップ')
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not args.input.exists():
        print(f"エラー: 入力ファイルが見つかりません: {args.input}")
        sys.exit(1)
    
    # 出力ディレクトリの作成
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    # 一時ファイルのパス
    temp_dir = Path('/tmp/book_scan_workflow')
    temp_dir.mkdir(exist_ok=True)
    temp_unpaper = temp_dir / 'unpaper_output.png'
    temp_reform = temp_dir / 'reform_output.png'
    
    print("=" * 60)
    print("書籍スキャン補正ワークフロー")
    print("=" * 60)
    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    print(f"レイアウト: {args.layout}")
    print(f"モデル: {args.model}")
    print(f"デバイス: {args.device}")
    if args.auto_finger:
        print(f"自動指除去: 有効 ({args.inpaint_method})")
    elif args.mask:
        print(f"マスク: {args.mask}")
    print("=" * 60)
    
    current_file = args.input
    
    # ステップ1: Unpaper
    if not args.skip_unpaper:
        if not check_unpaper():
            print("警告: unpaperがインストールされていません")
            print("  brew install unpaper でインストールしてください")
            print("  Unpaperをスキップします...")
        else:
            if run_unpaper(current_file, temp_unpaper, args.layout):
                current_file = temp_unpaper
    else:
        print("[1/3] Unpaper: スキップ")
    
    # ステップ2: py-reform
    if not args.skip_reform:
        if run_py_reform(current_file, temp_reform, args.model, args.device):
            current_file = temp_reform
    else:
        print("[2/3] py-reform: スキップ")
    
    # ステップ3: 指除去（オプション）
    if args.auto_finger:
        # 自動指検出・除去
        run_auto_finger_removal(current_file, args.output,
                               method=args.inpaint_method)
    elif args.mask:
        # マスク指定での指除去（従来方式）
        if not args.mask.exists():
            print(f"エラー: マスクファイルが見つかりません: {args.mask}")
            sys.exit(1)

        run_inpainting(current_file, args.mask, args.output)
    else:
        print("[3/3] 指除去: スキップ（--auto-finger または --mask を指定してください）")
        # 最終結果をコピー
        import shutil
        shutil.copy(current_file, args.output)
        print(f"  ✓ 最終結果を保存: {args.output}")
    
    print("=" * 60)
    print("✓ 処理完了！")
    print(f"結果: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
