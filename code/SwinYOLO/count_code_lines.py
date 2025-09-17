import os
import re
from pathlib import Path


def is_comment_line(line, file_ext):
    """
    判断是否为注释行
    """
    line = line.strip()
    
    # 空行
    if not line:
        return True
    
    # Python文件注释
    if file_ext in ['.py']:
        # 单行注释
        if line.startswith('#'):
            return True
        # 多行字符串（简单判断）
        if line.startswith('"""') or line.startswith("'''"):
            return True
        if line.endswith('"""') or line.endswith("'''"):
            return True
    
    # C/C++/Java/JavaScript等文件注释
    elif file_ext in ['.c', '.cpp', '.h', '.hpp', '.java', '.js', '.ts']:
        # 单行注释
        if line.startswith('//'):
            return True
        # 多行注释开始或结束
        if line.startswith('/*') or line.endswith('*/'):
            return True
        # 多行注释中间行
        if line.startswith('*'):
            return True
    
    # HTML/XML注释
    elif file_ext in ['.html', '.htm', '.xml']:
        if '<!--' in line and '-->' in line:
            return True
        if line.startswith('<!--') or line.endswith('-->'):
            return True
    
    # CSS注释
    elif file_ext in ['.css']:
        if line.startswith('/*') or line.endswith('*/'):
            return True
        if '/*' in line and '*/' in line:
            return True
    
    # YAML/YML注释
    elif file_ext in ['.yaml', '.yml']:
        if line.startswith('#'):
            return True
    
    # CFG配置文件注释
    elif file_ext in ['.cfg']:
        if line.startswith('#') or line.startswith(';'):
            return True
    
    return False


def count_code_lines_in_file(file_path):
    """
    统计单个文件的有效代码行数
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        file_ext = Path(file_path).suffix.lower()
        code_lines = 0
        total_lines = len(lines)
        in_multiline_comment = False
        
        for line in lines:
            stripped_line = line.strip()
            
            # 跳过空行
            if not stripped_line:
                continue
            
            # 处理多行注释
            if file_ext in ['.py']:
                # Python多行字符串
                if '"""' in stripped_line or "'''" in stripped_line:
                    quote_count = stripped_line.count('"""') + stripped_line.count("'''")
                    if quote_count % 2 == 1:
                        in_multiline_comment = not in_multiline_comment
                    continue
                
                if in_multiline_comment:
                    continue
            
            elif file_ext in ['.c', '.cpp', '.h', '.hpp', '.java', '.js', '.ts']:
                # C风格多行注释
                if '/*' in stripped_line:
                    in_multiline_comment = True
                if in_multiline_comment:
                    if '*/' in stripped_line:
                        in_multiline_comment = False
                    continue
            
            # 检查是否为注释行
            if not is_comment_line(stripped_line, file_ext):
                code_lines += 1
        
        return code_lines, total_lines
    
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return 0, 0


def count_code_lines_in_directory(directory, extensions=None):
    """
    统计目录下所有指定类型文件的代码行数
    """
    if extensions is None:
        extensions = ['.py', '.c', '.cpp', '.h', '.hpp', '.java', '.js', '.ts', '.html', '.css', '.cfg', '.yaml', '.yml']
    
    total_code_lines = 0
    total_file_lines = 0
    file_count = 0
    file_details = []
    
    for root, dirs, files in os.walk(directory):
        # 跳过常见的非代码目录
        dirs[:] = [d for d in dirs if d not in ['.git', '.svn', '__pycache__', 'node_modules', '.vscode', '.idea']]
        
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = Path(file).suffix.lower()
            
            if file_ext in extensions:
                code_lines, file_lines = count_code_lines_in_file(file_path)
                if file_lines > 0:  # 只统计非空文件
                    total_code_lines += code_lines
                    total_file_lines += file_lines
                    file_count += 1
                    
                    relative_path = os.path.relpath(file_path, directory)
                    file_details.append({
                        'path': relative_path,
                        'code_lines': code_lines,
                        'total_lines': file_lines,
                        'extension': file_ext
                    })
    
    return total_code_lines, total_file_lines, file_count, file_details


def print_statistics(directory, extensions=None):
    """
    打印统计结果
    """
    print(f"正在统计目录: {directory}")
    print("=" * 60)
    
    total_code_lines, total_file_lines, file_count, file_details = count_code_lines_in_directory(directory, extensions)
    
    # 按文件类型分组统计
    ext_stats = {}
    for detail in file_details:
        ext = detail['extension']
        if ext not in ext_stats:
            ext_stats[ext] = {'files': 0, 'code_lines': 0, 'total_lines': 0}
        ext_stats[ext]['files'] += 1
        ext_stats[ext]['code_lines'] += detail['code_lines']
        ext_stats[ext]['total_lines'] += detail['total_lines']
    
    # 打印按文件类型的统计
    print("\n按文件类型统计:")
    print(f"{'文件类型':<10} {'文件数':<8} {'代码行数':<10} {'总行数':<10} {'代码比例':<10}")
    print("-" * 60)
    
    for ext in sorted(ext_stats.keys()):
        stats = ext_stats[ext]
        ratio = stats['code_lines'] / stats['total_lines'] * 100 if stats['total_lines'] > 0 else 0
        print(f"{ext:<10} {stats['files']:<8} {stats['code_lines']:<10} {stats['total_lines']:<10} {ratio:.1f}%")
    
    # 打印详细文件列表（按代码行数排序）
    print(f"\n详细文件列表（按代码行数排序）:")
    print(f"{'文件路径':<50} {'代码行数':<10} {'总行数':<10}")
    print("-" * 80)
    
    file_details.sort(key=lambda x: x['code_lines'], reverse=True)
    for detail in file_details:
        print(f"{detail['path']:<50} {detail['code_lines']:<10} {detail['total_lines']:<10}")
    
    # 打印总统计
    print("\n" + "=" * 60)
    print(f"总计:")
    print(f"  文件数量: {file_count}")
    print(f"  有效代码行数: {total_code_lines:,}")
    print(f"  文件总行数: {total_file_lines:,}")
    print(f"  代码行占比: {total_code_lines/total_file_lines*100:.1f}%" if total_file_lines > 0 else "  代码行占比: 0%")
    print("=" * 60)


def main():
    """
    主函数
    """
    # 默认统计当前目录
    directory = input("请输入要统计的目录路径 (直接回车使用当前目录): ").strip()
    if not directory:
        directory = "."
    
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在！")
        return
    
    # 选择要统计的文件类型
    print("\n选择要统计的文件类型:")
    print("1. Python文件 (.py)")
    print("2. C/C++文件 (.c, .cpp, .h, .hpp)")
    print("3. Java文件 (.java)")
    print("4. JavaScript/TypeScript文件 (.js, .ts)")
    print("5. Web文件 (.html, .css, .js)")
    print("6. 所有支持的类型")
    print("7. 自定义扩展名")
    
    choice = input("请选择 (1-7, 默认为6): ").strip()
    
    extensions_map = {
        '1': ['.py'],
        '2': ['.c', '.cpp', '.h', '.hpp'],
        '3': ['.java'],
        '4': ['.js', '.ts'],
        '5': ['.html', '.css', '.js'],
        '6': ['.py', '.c', '.cpp', '.h', '.hpp', '.java', '.js', '.ts', '.html', '.css'],
    }
    
    if choice in extensions_map:
        extensions = extensions_map[choice]
    elif choice == '7':
        ext_input = input("请输入文件扩展名，用逗号分隔 (如: .py,.js,.cpp): ")
        extensions = [ext.strip() for ext in ext_input.split(',')]
    else:
        extensions = extensions_map['6']  # 默认所有类型
    
    print(f"\n将统计以下类型的文件: {', '.join(extensions)}")
    
    # 执行统计
    print_statistics(directory, extensions)


if __name__ == "__main__":
    main()
