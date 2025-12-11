#!/usr/bin/env python3
"""
手動ステップ・バイ・ステップ実行用スクリプト

使用方法:
    python manual_steps.py --step 1 --input book.jpg --output page_%d.png
    python manual_steps.py --step 2 --input page_1.png --output dewarped.png
    python manual_steps.py --step 3 --input dewarped.png --mask mask.png --output final.png
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
    print("  python3.11 -m pip install py-reform opencv-python numpy")
    sys.exit(1)


def check_unpaper():
    """unpaperがインストールされているか確認"""
    try:
        subprocess.run(['unpaper', '--version'], 
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def step1_unpaper(input_path, output_path, layout='single'):
    """
    ステップ1: Unpaperで前処理
    
    Args:
        input_path: 入力画像のパス
        output_path: 出力画像のパス
        layout: 'single' または 'double'
    """
    print(f"ステップ1: Unpaper実行中... (layout={layout})")
    
    if not check_unpaper():
        print("エラー: unpaperがインストールされていません")
        print("  brew install unpaper でインストールしてください")
        return False
    
    cmd = ['unpaper', '--layout', layout, str(input_path), str(output_path)]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ Unpaper完了: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Unpaperエラー: {e.stderr}")
        return False


def step2_py_reform(input_path, output_path, model='uvdoc', device='cpu'):
    """
    ステップ2: py-reformで歪み補正
    
    Args:
        input_path: 入力画像のパス
        output_path: 出力画像のパス
        model: 'uvdoc' または 'deskew'
        device: 'cpu', 'cuda', 'mps'
    """
    print(f"ステップ2: py-reform実行中... (model={model}, device={device})")
    
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


def step3_inpainting(input_path, mask_path, output_path, radius=3, method='telea'):
    """
    ステップ3: OpenCV Inpaintingで指除去
    
    Args:
        input_path: 入力画像のパス
        mask_path: マスク画像のパス
        output_path: 出力画像のパス
        radius: 修復半径
        method: 'telea' または 'ns'
    """
    print(f"ステップ3: Inpainting実行中... (method={method}, radius={radius})")
    
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


def main():
    parser = argparse.ArgumentParser(
        description='書籍スキャン補正の手動ステップ実行',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # ステップ1: Unpaperで前処理
  python manual_steps.py --step 1 --input book.jpg --output page_%d.png --layout double
  
  # ステップ2: py-reformで歪み補正
  python manual_steps.py --step 2 --input page_1.png --output dewarped.png --device mps
  
  # ステップ3: 指除去
  python manual_steps.py --step 3 --input dewarped.png --mask mask.png --output final.png
        """
    )
    
    parser.add_argument('--step', type=int, choices=[1, 2, 3], required=True,
                       help='実行するステップ (1: Unpaper, 2: py-reform, 3: Inpainting)')
    parser.add_argument('--input', '-i', type=Path, required=True,
                       help='入力画像のパス')
    parser.add_argument('--output', '-o', type=Path, required=True,
                       help='出力画像のパス')
    parser.add_argument('--layout', choices=['single', 'double'], default='single',
                       help='ページレイアウト (ステップ1のみ, default: single)')
    parser.add_argument('--model', choices=['uvdoc', 'deskew'], default='uvdoc',
                       help='歪み補正モデル (ステップ2のみ, default: uvdoc)')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu',
                       help='使用デバイス (ステップ2のみ, default: cpu)')
    parser.add_argument('--mask', type=Path,
                       help='指除去用のマスク画像（ステップ3のみ）')
    parser.add_argument('--method', choices=['telea', 'ns'], default='telea',
                       help='Inpainting手法 (ステップ3のみ, default: telea)')
    parser.add_argument('--radius', type=int, default=3,
                       help='修復半径 (ステップ3のみ, default: 3)')
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not args.input.exists():
        print(f"エラー: 入力ファイルが見つかりません: {args.input}")
        sys.exit(1)
    
    # ステップ3の場合はマスクファイルも確認
    if args.step == 3 and args.mask and not args.mask.exists():
        print(f"エラー: マスクファイルが見つかりません: {args.mask}")
        sys.exit(1)
    
    # 出力ディレクトリの作成
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"書籍スキャン補正 - ステップ{args.step}")
    print("=" * 60)
    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    
    if args.step == 1:
        print(f"レイアウト: {args.layout}")
    elif args.step == 2:
        print(f"モデル: {args.model}")
        print(f"デバイス: {args.device}")
    elif args.step == 3:
        print(f"マスク: {args.mask}")
        print(f"手法: {args.method}")
        print(f"半径: {args.radius}")
    
    print("=" * 60)
    
    # ステップ実行
    success = False
    
    if args.step == 1:
        success = step1_unpaper(args.input, args.output, args.layout)
    elif args.step == 2:
        success = step2_py_reform(args.input, args.output, args.model, args.device)
    elif args.step == 3:
        if not args.mask:
            print("エラー: ステップ3にはマスク画像が必要です")
            sys.exit(1)
        success = step3_inpainting(args.input, args.mask, args.output, args.radius, args.method)
    
    if success:
        print("=" * 60)
        print("✓ 処理完了！")
        print(f"結果: {args.output}")
        print("=" * 60)
    else:
        print("=" * 60)
        print("✗ 処理失敗")
        sys.exit(1)


if __name__ == '__main__':
    main()
