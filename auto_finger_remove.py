#!/usr/bin/env python3
"""
自動指検出・除去ツール

MediaPipeで指を自動検出し、LaMa または OpenCV で除去します。

使用方法:
    python auto_finger_remove.py image.jpg
    python auto_finger_remove.py image.jpg --output result.jpg
    python auto_finger_remove.py image.jpg --method lama  # 高品質
    python auto_finger_remove.py image.jpg --method opencv  # 高速

必要なライブラリ:
    pip install mediapipe opencv-python numpy pillow
    pip install simple-lama-inpainting  # LaMa使用時
"""

import argparse
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
except ImportError:
    print("エラー: OpenCVがインストールされていません")
    print("  pip install opencv-python numpy")
    sys.exit(1)

try:
    import mediapipe as mp
except ImportError:
    print("エラー: MediaPipeがインストールされていません")
    print("  pip install mediapipe")
    sys.exit(1)

# LaMaはオプション
LAMA_AVAILABLE = False
try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except ImportError:
    pass


class AutoFingerRemover:
    """自動指検出・除去クラス"""

    def __init__(self, margin=30, min_detection_confidence=0.5):
        """
        Args:
            margin: 検出領域の拡張マージン（ピクセル）
            min_detection_confidence: 検出信頼度の閾値
        """
        self.margin = margin
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=4,
            min_detection_confidence=min_detection_confidence
        )
        self.lama = None
        if LAMA_AVAILABLE:
            self.lama = SimpleLama()

    def detect_fingers(self, image):
        """
        画像から指/手を検出してマスクを生成

        Args:
            image: BGR画像（numpy配列）

        Returns:
            mask: 指の部分が白（255）のマスク画像
            detected: 検出されたかどうか
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # RGB変換してMediaPipeで検出
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            return mask, False

        # 各手のランドマークからマスクを生成
        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])

            points = np.array(points, dtype=np.int32)

            # 凸包を計算してマージン付きで塗りつぶし
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255)

            # 各ランドマーク周辺も塗りつぶし（指先などをカバー）
            for point in points:
                cv2.circle(mask, tuple(point), self.margin, 255, -1)

        # マスクを膨張させてエッジをカバー
        kernel = np.ones((self.margin, self.margin), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask, True

    def inpaint_opencv(self, image, mask, radius=5, method='telea'):
        """
        OpenCVでインペインティング

        Args:
            image: BGR画像
            mask: マスク画像
            radius: 修復半径
            method: 'telea' または 'ns'

        Returns:
            修復された画像
        """
        flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
        return cv2.inpaint(image, mask, radius, flag)

    def inpaint_lama(self, image, mask):
        """
        LaMaでインペインティング（高品質）

        Args:
            image: BGR画像
            mask: マスク画像

        Returns:
            修復された画像
        """
        if not LAMA_AVAILABLE or self.lama is None:
            print("警告: LaMaが利用できません。OpenCVにフォールバックします。")
            return self.inpaint_opencv(image, mask)

        from PIL import Image

        # OpenCV BGR -> PIL RGB
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)

        # LaMaで修復
        result = self.lama(pil_image, pil_mask)

        # PIL RGB -> OpenCV BGR
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    def process(self, image, method='auto'):
        """
        画像から指を自動検出して除去

        Args:
            image: BGR画像
            method: 'lama', 'opencv', 'auto'
                   auto: LaMaが利用可能ならLaMa、そうでなければOpenCV

        Returns:
            result: 修復された画像
            mask: 使用したマスク
            detected: 指が検出されたかどうか
        """
        # 指を検出
        mask, detected = self.detect_fingers(image)

        if not detected:
            print("指が検出されませんでした。元の画像を返します。")
            return image.copy(), mask, False

        # インペインティング方法を決定
        if method == 'auto':
            method = 'lama' if LAMA_AVAILABLE else 'opencv'

        print(f"指を検出しました。{method}で修復中...")

        # 修復実行
        if method == 'lama':
            result = self.inpaint_lama(image, mask)
        else:
            result = self.inpaint_opencv(image, mask)

        return result, mask, True

    def close(self):
        """リソースを解放"""
        self.hands.close()


def process_image(input_path, output_path=None, method='auto',
                  save_mask=False, margin=30, confidence=0.5):
    """
    画像ファイルを処理

    Args:
        input_path: 入力画像パス
        output_path: 出力画像パス（Noneの場合は自動生成）
        method: 修復方法
        save_mask: マスク画像を保存するか
        margin: 検出マージン
        confidence: 検出信頼度

    Returns:
        成功したかどうか
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_no_finger{input_path.suffix}"
    else:
        output_path = Path(output_path)

    # 画像読み込み
    image = cv2.imread(str(input_path))
    if image is None:
        print(f"エラー: 画像を読み込めません: {input_path}")
        return False

    print(f"入力: {input_path}")
    print(f"サイズ: {image.shape[1]}x{image.shape[0]}")

    # 処理
    remover = AutoFingerRemover(margin=margin, min_detection_confidence=confidence)
    try:
        result, mask, detected = remover.process(image, method=method)
    finally:
        remover.close()

    if not detected:
        print("指が検出されなかったため、処理をスキップしました。")
        return True

    # 結果保存
    cv2.imwrite(str(output_path), result)
    print(f"出力: {output_path}")

    # マスク保存（オプション）
    if save_mask:
        mask_path = output_path.parent / f"{output_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        print(f"マスク: {mask_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description='自動指検出・除去ツール',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使い方（自動でLaMaまたはOpenCVを選択）
  python auto_finger_remove.py scan.jpg

  # 出力ファイル名を指定
  python auto_finger_remove.py scan.jpg --output clean.jpg

  # 高品質モード（LaMa）
  python auto_finger_remove.py scan.jpg --method lama

  # 高速モード（OpenCV）
  python auto_finger_remove.py scan.jpg --method opencv

  # マスク画像も保存
  python auto_finger_remove.py scan.jpg --save-mask

  # 検出感度を調整
  python auto_finger_remove.py scan.jpg --confidence 0.3 --margin 40

必要なライブラリ:
  pip install mediapipe opencv-python numpy pillow
  pip install simple-lama-inpainting  # 高品質モード用（オプション）
        """
    )

    parser.add_argument('image', type=Path, help='入力画像のパス')
    parser.add_argument('--output', '-o', type=Path, help='出力画像のパス')
    parser.add_argument('--method', '-m', choices=['auto', 'lama', 'opencv'],
                       default='auto',
                       help='修復方法 (default: auto)')
    parser.add_argument('--save-mask', action='store_true',
                       help='検出したマスク画像も保存')
    parser.add_argument('--margin', type=int, default=30,
                       help='検出領域の拡張マージン (default: 30)')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='検出信頼度の閾値 (default: 0.5)')

    args = parser.parse_args()

    if not args.image.exists():
        print(f"エラー: 画像が見つかりません: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print("自動指検出・除去ツール")
    print("=" * 60)

    if not LAMA_AVAILABLE:
        print("注意: LaMaが利用できません（OpenCVを使用）")
        print("  高品質モードを使うには: pip install simple-lama-inpainting")

    print()

    success = process_image(
        args.image,
        args.output,
        method=args.method,
        save_mask=args.save_mask,
        margin=args.margin,
        confidence=args.confidence
    )

    print("=" * 60)
    if success:
        print("✓ 処理完了！")
    else:
        print("✗ 処理失敗")
        sys.exit(1)


if __name__ == '__main__':
    main()
