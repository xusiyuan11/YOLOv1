#!/usr/bin/env python3
"""
YOLO项目代码行数统计工具
支持YOLOv1, YOLOv3, YOLOv4, YOLOv5等各种版本
"""

import os
import glob
from pathlib import Path


def is_comment_or_empty(line, file_ext):
    """判断是否为注释行或空行"""
    line = line.strip()
    
    if not line:
        return True
    
    # Python文件
    if file_ext == '.py':
        return line.startswith('#')
    
    # C/C++文件
    elif file_ext in ['.c', '.cpp', '.h', '.hpp', '.cu', '.cuh']:
        return line.startswith('//') or line.startswith('/*') or line.startswith('*')
    
    # 配置文件
    elif file_ext in ['.cfg', '.yaml', '.yml']:
        return line.startswith('#')
    
    # Makefile
    elif file_ext in ['.mk'] or 'makefile' in line.lower():
        return line.startswith('#')
    
    # Shell脚本
    elif file_ext in ['.sh', '.bash']:
        return line.startswith('#')
    
    return False


def count_file_lines(file_path):
    """统计单个文件的代码行数"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        file_ext = Path(file_path).suffix.lower()
        total_lines = len(lines)
        code_lines = 0
        in_multiline_comment = False
        
        for line in lines:
            stripped = line.strip()
            
            # 处理多行注释
            if file_ext == '.py':
                # Python多行字符串
                if '"""' in stripped or "'''" in stripped:
                    quote_count = stripped.count('"""') + stripped.count("'''")
                    if quote_count % 2 == 1:
                        in_multiline_comment = not in_multiline_comment
                    continue
                
                if in_multiline_comment:
                    continue
            
            elif file_ext in ['.c', '.cpp', '.h', '.hpp', '.cu', '.cuh']:
                # C风格多行注释
                if '/*' in stripped:
                    in_multiline_comment = True
                if in_multiline_comment:
                    if '*/' in stripped:
                        in_multiline_comment = False
                    continue
            
            # 统计非注释非空行
            if not is_comment_or_empty(stripped, file_ext):
                code_lines += 1
        
        return code_lines, total_lines
    
    except Exception as e:
        print(f"读取文件 {file_path} 出错: {e}")
        return 0, 0


def scan_yolo_project(directory="."):
    """扫描YOLO项目目录"""
    
    # 支持的文件类型
    file_patterns = [
        "*.py",      # Python源码
        "*.c", "*.cpp", "*.h", "*.hpp",  # C/C++源码
        "*.cu", "*.cuh",  # CUDA源码
        "*.cfg",     # YOLO配置文件
        "*.yaml", "*.yml",  # YAML配置
        "*.sh",      # Shell脚本
        "Makefile", "makefile", "*.mk",  # Makefile
    ]
    
    all_files = []
    for pattern in file_patterns:
        files = glob.glob(os.path.join(directory, "**", pattern), recursive=True)
        all_files.extend(files)
    
    # 排除常见的非代码目录
    exclude_dirs = {'.git', '__pycache__', '.vscode', '.idea', 'build', 'dist', 'node_modules'}
    
    filtered_files = []
    for file_path in all_files:
        # 检查路径中是否包含排除的目录
        path_parts = set(Path(file_path).parts)
        if not path_parts.intersection(exclude_dirs):
            filtered_files.append(file_path)
    
    return sorted(filtered_files)


def analyze_yolo_project(directory="."):
    """分析YOLO项目的代码统计"""
    
    print(f"正在分析YOLO项目: {os.path.abspath(directory)}")
    print("=" * 80)
    
    files = scan_yolo_project(directory)
    
    if not files:
        print("未找到任何代码文件！")
        return
    
    # 按文件类型分组
    file_types = {}
    total_code_lines = 0
    total_file_lines = 0
    
    print(f"{'文件路径':<50} {'代码行数':<10} {'总行数':<10} {'类型':<8}")
    print("-" * 80)
    
    file_stats = []
    
    for file_path in files:
        code_lines, total_lines = count_file_lines(file_path)
        file_ext = Path(file_path).suffix.lower() or 'other'
        
        # 相对路径显示
        rel_path = os.path.relpath(file_path, directory)
        
        # 按类型统计
        if file_ext not in file_types:
            file_types[file_ext] = {'files': 0, 'code_lines': 0, 'total_lines': 0}
        
        file_types[file_ext]['files'] += 1
        file_types[file_ext]['code_lines'] += code_lines
        file_types[file_ext]['total_lines'] += total_lines
        
        total_code_lines += code_lines
        total_file_lines += total_lines
        
        file_stats.append((rel_path, code_lines, total_lines, file_ext))
        
        print(f"{rel_path:<50} {code_lines:<10} {total_lines:<10} {file_ext:<8}")
    
    print("-" * 80)
    
    # 按文件类型汇总
    print(f"\n按文件类型统计:")
    print(f"{'类型':<10} {'文件数':<8} {'代码行数':<10} {'总行数':<10} {'代码比例':<10}")
    print("-" * 50)
    
    for ext in sorted(file_types.keys()):
        stats = file_types[ext]
        ratio = stats['code_lines'] / stats['total_lines'] * 100 if stats['total_lines'] > 0 else 0
        print(f"{ext:<10} {stats['files']:<8} {stats['code_lines']:<10} {stats['total_lines']:<10} {ratio:.1f}%")
    
    # 总体统计
    print("\n" + "=" * 80)
    print(f"项目总计:")
    print(f"  文件总数: {len(files)}")
    print(f"  有效代码行数: {total_code_lines:,}")
    print(f"  文件总行数: {total_file_lines:,}")
    overall_ratio = total_code_lines / total_file_lines * 100 if total_file_lines > 0 else 0
    print(f"  代码行占比: {overall_ratio:.1f}%")
    
    # 显示代码量最多的文件
    file_stats.sort(key=lambda x: x[1], reverse=True)
    print(f"\n代码量最多的文件 (Top 10):")
    for i, (path, code_lines, total_lines, ext) in enumerate(file_stats[:10]):
        print(f"  {i+1:2d}. {path:<40} {code_lines:>6} 行 ({ext})")
    
    print("=" * 80)
    
    return {
        'total_files': len(files),
        'total_code_lines': total_code_lines,
        'total_file_lines': total_file_lines,
        'file_types': file_types,
        'file_stats': file_stats
    }


def compare_projects(*directories):
    """比较多个YOLO项目的代码量"""
    
    print("YOLO项目代码量对比")
    print("=" * 100)
    
    results = []
    for directory in directories:
        if os.path.exists(directory):
            print(f"\n分析项目: {directory}")
            result = analyze_yolo_project(directory)
            result['directory'] = directory
            results.append(result)
        else:
            print(f"目录不存在: {directory}")
    
    if len(results) > 1:
        print(f"\n项目对比总结:")
        print(f"{'项目':<30} {'文件数':<8} {'代码行数':<10} {'总行数':<10} {'代码比例':<10}")
        print("-" * 70)
        
        for result in results:
            ratio = result['total_code_lines'] / result['total_file_lines'] * 100 if result['total_file_lines'] > 0 else 0
            project_name = os.path.basename(result['directory']) or result['directory']
            print(f"{project_name:<30} {result['total_files']:<8} {result['total_code_lines']:<10} {result['total_file_lines']:<10} {ratio:.1f}%")


def main():
    """主函数"""
    print("YOLO项目代码统计工具")
    print("支持 YOLOv1, YOLOv3, YOLOv4, YOLOv5 等各版本")
    print()
    
    mode = input("选择模式:\n1. 分析单个项目\n2. 比较多个项目\n请选择 (1-2, 默认1): ").strip()
    
    if mode == '2':
        print("请输入要比较的项目目录，每行一个，空行结束:")
        directories = []
        while True:
            directory = input("项目目录: ").strip()
            if not directory:
                break
            directories.append(directory)
        
        if directories:
            compare_projects(*directories)
        else:
            print("未输入任何目录！")
    
    else:
        # 单个项目分析
        directory = input("请输入项目目录 (直接回车使用当前目录): ").strip()
        if not directory:
            directory = "."
        
        if os.path.exists(directory):
            analyze_yolo_project(directory)
        else:
            print(f"目录不存在: {directory}")


if __name__ == "__main__":
    main()

