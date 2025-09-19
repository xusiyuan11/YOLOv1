# 部署与运行环境

## 环境要求

在部署和运行本项目之前，请确保您的系统满足以下要求：

- 操作系统：Windows 10 或更高版本 / Ubuntu 20.04 或更高版本
- Python 版本：3.8 或更高版本
- 必要的依赖库：详见 `requirements.txt`

## 部署步骤

1. **克隆代码仓库**

   ```bash
   git clone <仓库地址>
   cd <项目目录>
   ```

2. **创建虚拟环境**（推荐）

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows
   ```

3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   ```

4. **配置环境变量**

   根据项目需求，设置必要的环境变量。例如：

   ```bash
   export DATASET_PATH=/path/to/dataset  # Linux/macOS
   set DATASET_PATH=C:\path\to\dataset  # Windows
   ```



## 注意事项

- 确保您的 GPU 驱动和 CUDA 环境已正确安装（如果使用 GPU 加速）。
- 如果遇到依赖冲突或安装问题，请参考 `requirements.txt` 中的版本说明。
- 数据集路径和其他配置请根据实际情况修改。

## 常见问题

1. **依赖安装失败**
   - 确保您已激活虚拟环境。
   - 检查 `pip` 是否为最新版本：
     ```bash
     pip install --upgrade pip
     ```

2. **无法找到数据集路径**
   - 确保环境变量 `DATASET_PATH` 设置正确。
   - 检查路径是否存在并具有正确的权限。

3. **模型运行报错**
   - 检查是否正确安装了所有依赖。
   - 确保使用的 Python 版本与项目要求一致。

如有其他问题，请联系项目维护者或查阅项目文档。
