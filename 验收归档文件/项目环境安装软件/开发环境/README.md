# 开发环境

## 环境要求

在开发本项目之前，请确保您的系统满足以下要求：

- 操作系统：Windows 10 或更高版本 / Ubuntu 20.04 或更高版本
- Python 版本：3.8 或更高版本
- 开发工具：Visual Studio Code（推荐）
- 必要的依赖库：详见 `requirements.txt`

## 开发环境配置

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

4. **安装开发工具**

   - 下载并安装 [Visual Studio Code](https://code.visualstudio.com/)。
   - 安装推荐的扩展：
     - Python
     - Pylance
     - GitLens

5. **配置代码格式化工具**

   - 安装 `black` 和 `isort`：
     ```bash
     pip install black isort
     ```
   - 在 VS Code 中配置格式化工具：
     ```json
     // settings.json
     {
         "python.formatting.provider": "black",
         "editor.formatOnSave": true,
         "python.sortImports.args": ["--profile", "black"]
     }
     ```

6. **运行开发服务器**

   根据项目模块，运行相应的开发脚本。例如：

   - 运行 YOLOv1 模型：
     ```bash
     python YOLOv1/Train_Detection.py
     ```
   - 运行 YOLOv3 模型：
     ```bash
     python YOLOv3/Train_Detection.py
     ```
   - 运行 SwinYOLO 模型：
     ```bash
     python SwinYOLO/train_swin_yolo.py
     ```

## 注意事项

- 确保您的开发环境与运行环境一致，以避免部署时出现问题。
- 定期更新依赖库：
  ```bash
  pip install --upgrade -r requirements.txt
  ```
- 使用 Git 进行版本控制，确保代码变更被正确记录。

## 常见问题

1. **依赖安装失败**
   - 确保您已激活虚拟环境。
   - 检查 `pip` 是否为最新版本：
     ```bash
     pip install --upgrade pip
     ```

2. **代码格式化失败**
   - 确保已安装 `black` 和 `isort`。
   - 检查 VS Code 的 `settings.json` 配置是否正确。

3. **无法运行开发脚本**
   - 检查是否正确安装了所有依赖。
   - 确保使用的 Python 版本与项目要求一致。

如有其他问题，请联系项目维护者或查阅项目文档。
