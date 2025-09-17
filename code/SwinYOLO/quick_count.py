#!/usr/bin/env python3
import os
import glob


def count_effective_lines(file_path):
    """统计文件中的有效代码行数（排除空行和注释）"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        effective_lines = 0
        in_multiline_string = False
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行
            if not stripped:
                continue
            
            # 处理Python多行字符串
            if '"""' in stripped or "'''" in stripped:
                quote_count = stripped.count('"""') + stripped.count("'''")
                if quote_count % 2 == 1:
                    in_multiline_string = not in_multiline_string
                if in_multiline_string and quote_count == 1:
                    continue
            
            # 跳过多行字符串内容
            if in_multiline_string:
                continue
            
            # 跳过注释行
            if stripped.startswith('#'):
                continue
            
            # 统计有效代码行
            effective_lines += 1
        
        return effective_lines, len(lines)
    
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0


def main():
    """快速统计当前YOLO项目的代码行数"""
    
    # 要统计的Python文件
    python_files = glob.glob("*.py")
    
    total_effective = 0
    total_lines = 0
    
    print("YOLO项目代码行数统计")
    print("=" * 60)
    print(f"{'文件名':<30} {'有效代码行':<12} {'总行数':<10} {'比例':<10}")
    print("-" * 60)
    
    file_stats = []
    
    for file_path in sorted(python_files):
        effective, total = count_effective_lines(file_path)
        ratio = effective / total * 100 if total > 0 else 0
        
        file_stats.append((file_path, effective, total, ratio))
        total_effective += effective
        total_lines += total
        
        print(f"{file_path:<30} {effective:<12} {total:<10} {ratio:.1f}%")
    
    print("-" * 60)
    overall_ratio = total_effective / total_lines * 100 if total_lines > 0 else 0
    print(f"{'总计':<30} {total_effective:<12} {total_lines:<10} {overall_ratio:.1f}%")
    print("=" * 60)
    
    # 显示最大的几个文件
    file_stats.sort(key=lambda x: x[1], reverse=True)
    print("\n代码量最多的文件:")
    for i, (name, effective, total, ratio) in enumerate(file_stats[:5]):
        print(f"{i+1}. {name}: {effective} 行有效代码")
    
    print(f"\n项目总计: {len(python_files)} 个Python文件，{total_effective:,} 行有效代码")


if __name__ == "__main__":
    main()
