#!/usr/bin/env python3
"""
修复项目中的相对导入
此脚本会递归遍历项目目录，将所有Python文件中的相对导入（以..开头）
替换为绝对导入（直接从项目根目录开始）
"""

import os
import re
import sys

def fix_relative_imports(directory):
    """递归遍历目录，修复所有Python文件中的相对导入"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                fix_file_imports(file_path)

def fix_file_imports(file_path):
    """修复单个文件中的相对导入"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 查找相对导入（形如 from ..module import X 或 from ..module.submodule import X）
    relative_import_pattern = r'from\s+\.\.([a-zA-Z0-9_.]+)\s+import'
    
    # 替换为绝对导入
    modified_content = re.sub(relative_import_pattern, r'from \1 import', content)
    
    # 仅处理..(两个点)开头的相对导入，单个点的相对导入保留
    if content != modified_content:
        print(f"修复文件: {file_path}")
        with open(file_path, 'w') as f:
            f.write(modified_content)

def add_syspath_to_files(directory):
    """向所有Python文件添加sys.path处理代码"""
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') and '__init__.py' not in file:
                file_path = os.path.join(root, file)
                add_syspath(file_path)

def add_syspath(file_path):
    """向单个文件添加sys.path处理代码"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 检查文件是否已包含sys.path处理
    if "sys.path.insert(0, os.path.abspath(os.path.join" not in content:
        # 查找文件顶部的导入语句
        import_section_end = 0
        lines = content.split('\n')
        
        # 找到导入语句的结束位置
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_section_end = i + 1
        
        # 添加sys.path处理代码
        syspath_code = [
            "import sys",
            "import os",
            "# 添加项目根目录到sys.path",
            "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))",
        ]
        
        # 向文件中插入代码
        if "import sys" not in content and "import os" not in content:
            lines = lines[:import_section_end] + syspath_code + lines[import_section_end:]
        elif "import sys" in content and "import os" not in content:
            lines = lines[:import_section_end] + ["import os"] + [syspath_code[2], syspath_code[3]] + lines[import_section_end:]
        elif "import sys" not in content and "import os" in content:
            lines = lines[:import_section_end] + ["import sys"] + [syspath_code[2], syspath_code[3]] + lines[import_section_end:]
        else:
            lines = lines[:import_section_end] + [syspath_code[2], syspath_code[3]] + lines[import_section_end:]
        
        # 写回文件
        print(f"添加sys.path到文件: {file_path}")
        with open(file_path, 'w') as f:
            f.write('\n'.join(lines))

if __name__ == "__main__":
    # 获取项目根目录
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"正在处理项目目录: {project_dir}")
    
    # 修复相对导入
    fix_relative_imports(project_dir)
    
    # 添加sys.path处理
    add_syspath_to_files(project_dir)
    
    print("完成！") 