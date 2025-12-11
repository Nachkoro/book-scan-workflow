#!/usr/bin/env python3
"""
書籍スキャン補正ワークフローのテストスクリプト

環境とツールの動作をテストします。
"""

import subprocess
import sys
from pathlib import Path

def test_unpaper():
    """Unpaperの動作テスト"""
    print("1. Unpaperの動作テスト...")
    try:
        result = subprocess.run(['unpaper', '--version'], 
                              capture_output=True, text=True, check=True)
        version_info = result.stdout.strip() or result.stderr.strip()
        print(f"  ✓ Unpaperが利用可能です: {version_info}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ✗ Unpaperが見つかりません")
        return False

def test_python_libraries():
    """Pythonライブラリの動作テスト"""
    print("2. Pythonライブラリの動作テスト...")
    
    libraries = ['numpy', 'cv2', 'py_reform']
    for lib in libraries:
        try:
            __import__(lib)
            print(f"  ✓ {lib} が利用可能です")
        except ImportError:
            print(f"  ✗ {lib} が見つかりません")
            return False
    return True

def test_complete_workflow():
    """完全ワークフロースクリプトのテスト"""
    print("3. complete_workflow.py のテスト...")
    
    script_path = Path(__file__).parent / 'complete_workflow.py'
    if not script_path.exists():
        print("  ✗ complete_workflow.py が見つかりません")
        return False
    
    try:
        # ヘルプを表示して引数解析が正常に動作するかテスト
        result = subprocess.run([
            'python3.11', str(script_path), '--help'
        ], capture_output=True, text=True, check=True)
        print("  ✓ complete_workflow.py が正常に動作します")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ complete_workflow.py エラー: {e.stderr}")
        return False

def test_interactive_inpaint():
    """インタラクティブ指除去ツールのテスト"""
    print("4. interactive_inpaint.py のテスト...")
    
    script_path = Path(__file__).parent / 'interactive_inpaint.py'
    if not script_path.exists():
        print("  ✗ interactive_inpaint.py が見つかりません")
        return False
    
    try:
        # ヘルプを表示して引数解析が正常に動作するかテスト
        result = subprocess.run([
            'python3.11', str(script_path), '--help'
        ], capture_output=True, text=True, check=True)
        print("  ✓ interactive_inpaint.py が正常に動作します")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ interactive_inpaint.py エラー: {e.stderr}")
        return False

def test_manual_steps():
    """手動ステップスクリプトのテスト"""
    print("5. manual_steps.py のテスト...")
    
    script_path = Path(__file__).parent / 'manual_steps.py'
    if not script_path.exists():
        print("  ✗ manual_steps.py が見つかりません")
        return False
    
    try:
        # ヘルプを表示して引数解析が正常に動作するかテスト
        result = subprocess.run([
            'python3.11', str(script_path), '--help'
        ], capture_output=True, text=True, check=True)
        print("  ✓ manual_steps.py が正常に動作します")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ manual_steps.py エラー: {e.stderr}")
        return False

def main():
    print("=" * 60)
    print("書籍スキャン補正ワークフロー - 環境テスト")
    print("=" * 60)
    
    tests = [
        test_unpaper,
        test_python_libraries,
        test_complete_workflow,
        test_interactive_inpaint,
        test_manual_steps
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"テスト結果: {passed}/{total} 通過")
    
    if passed == total:
        print("✓ すべてのテストに合格しました！")
        print("ワークフローを使用する準備が整いました。")
        print()
        print("使用例:")
        print("  python3.11 complete_workflow.py --input scan.jpg --output result.jpg")
        print("  python3.11 interactive_inpaint.py scan.jpg")
        print("  python3.11 manual_steps.py --step 1 --input book.jpg --output page_%d.png --layout double")
    else:
        print("✗ 一部のテストに失敗しました。")
        print("環境設定を確認してください。")
    
    print("=" * 60)
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
