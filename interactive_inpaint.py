#!/usr/bin/env python3
"""
インタラクティブな指除去ツール

使用方法:
    python interactive_inpaint.py image.jpg

操作方法:
    - マウスドラッグ: 指の部分を塗りつぶす
    - 'r'キー: リセット
    - 'i'キー: Inpainting実行
    - 's'キー: 結果を保存
    - '+'キー: ブラシサイズを大きく
    - '-'キー: ブラシサイズを小さく
    - 'q'キー: 終了
"""

import sys
import argparse
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("エラー: OpenCVがインストールされていません")
    print("  pip install opencv-python numpy")
    sys.exit(1)


class InpaintTool:
    def __init__(self, image_path):
        self.img = cv2.imread(str(image_path))
        if self.img is None:
            raise ValueError(f"画像を読み込めません: {image_path}")
        
        self.img_show = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.drawing = False
        self.brush_size = 20
        self.result = None
        
        # ウィンドウ名
        self.window_name = 'Inpainting Tool'
        
    def draw_callback(self, event, x, y, flags, param):
        """マウスイベントのコールバック"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
            cv2.circle(self.img_show, (x, y), self.brush_size, (0, 255, 0), -1)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cv2.circle(self.mask, (x, y), self.brush_size, 255, -1)
                cv2.circle(self.img_show, (x, y), self.brush_size, (0, 255, 0), -1)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
    
    def reset(self):
        """リセット"""
        self.img_show = self.img.copy()
        self.mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        self.result = None
        print("リセットしました")
    
    def inpaint(self, method='telea', radius=3):
        """Inpainting実行"""
        if np.sum(self.mask) == 0:
            print("警告: マスクが空です。先に指の部分を塗りつぶしてください。")
            return
        
        print(f"Inpainting実行中... (method={method}, radius={radius})")
        
        flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        self.result = cv2.inpaint(self.img, self.mask, radius, flag)
        self.img_show = self.result.copy()
        
        print("Inpainting完了！")
        cv2.imshow('Result', self.result)
    
    def save(self, output_path):
        """結果を保存"""
        if self.result is None:
            print("警告: まだInpaintingを実行していません")
            return False
        
        cv2.imwrite(str(output_path), self.result)
        print(f"結果を保存しました: {output_path}")
        return True
    
    def run(self):
        """メインループ"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.draw_callback)
        
        print("=" * 60)
        print("インタラクティブな指除去ツール")
        print("=" * 60)
        print("操作方法:")
        print("  マウスドラッグ: 指の部分を塗りつぶす")
        print("  'r'キー: リセット")
        print("  'i'キー: Inpainting実行（TELEA法）")
        print("  'n'キー: Inpainting実行（NS法、高品質）")
        print("  's'キー: 結果を保存")
        print("  '+'キー: ブラシサイズを大きく")
        print("  '-'キー: ブラシサイズを小さく")
        print("  'q'キー: 終了")
        print("=" * 60)
        print(f"現在のブラシサイズ: {self.brush_size}")
        
        while True:
            # ブラシサイズを表示
            display = self.img_show.copy()
            cv2.putText(display, f"Brush: {self.brush_size}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display)
            k = cv2.waitKey(1) & 0xFF
            
            if k == ord('r'):  # リセット
                self.reset()
            
            elif k == ord('i'):  # Inpainting（TELEA）
                self.inpaint(method='telea', radius=3)
            
            elif k == ord('n'):  # Inpainting（NS）
                self.inpaint(method='ns', radius=3)
            
            elif k == ord('s'):  # 保存
                if self.result is not None:
                    output_path = Path('inpaint_result.jpg')
                    self.save(output_path)
                else:
                    print("警告: まずInpaintingを実行してください（'i'キー）")
            
            elif k == ord('+') or k == ord('='):  # ブラシサイズを大きく
                self.brush_size = min(self.brush_size + 5, 100)
                print(f"ブラシサイズ: {self.brush_size}")
            
            elif k == ord('-') or k == ord('_'):  # ブラシサイズを小さく
                self.brush_size = max(self.brush_size - 5, 5)
                print(f"ブラシサイズ: {self.brush_size}")
            
            elif k == ord('q'):  # 終了
                break
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='インタラクティブな指除去ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('image', type=Path, help='入力画像のパス')
    
    args = parser.parse_args()
    
    if not args.image.exists():
        print(f"エラー: 画像が見つかりません: {args.image}")
        sys.exit(1)
    
    try:
        tool = InpaintTool(args.image)
        tool.run()
    except Exception as e:
        print(f"エラー: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
