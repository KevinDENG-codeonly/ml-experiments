# Docker 使用说明 - ViT模型训练

本文档提供使用Docker容器进行ViT（Vision Transformer）模型训练的相关指南，使用PyTorch 2.1.0、CUDA 12.1.0和Python 3.10。我们解决了相对导入的问题，同时保持了原有的文件结构和名称。

## 构建Docker镜像

在项目根目录下，运行以下命令构建Docker镜像：

```bash
docker build -t vit-training -f Dockerfile .
```

## 运行Docker容器进行训练

要运行容器并开始训练过程：

```bash
docker run --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/outputs:/outputs \
  vit-training
```

容器启动时会自动运行训练脚本。

## 运行Docker容器进行开发

如果需要在容器内进行开发，可以使用shell交互模式：

```bash
docker run --gpus all \
  -v $(pwd)/data:/data \
  -v $(pwd)/outputs:/outputs \
  -it --entrypoint /bin/bash \
  vit-training
```

## 问题解决

### 导入错误解决方案

原始的导入错误是由以下原因引起的：

1. 使用相对导入（例如`from ..utils.logger import get_logger`）时，Python无法正确解析路径
2. PYTHONPATH环境变量没有正确设置
3. 程序执行方式不正确

我们的解决方案：

1. **保持原有文件名和结构**：不改变`ml-experiments`目录名，保持原有代码结构
2. **通过sys.path动态添加根目录**：使用`sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))` 
3. **提供导入修复脚本**：`fix_imports.py`自动处理项目中的导入问题
4. **适当设置PYTHONPATH**：通过环境变量确保Python能找到正确的模块

### 在特定GPU上运行

要在特定GPU上运行，可以使用：

```bash
docker run --gpus '"device=0,1"' \
  -v $(pwd)/data:/data \
  -v $(pwd)/outputs:/outputs \
  vit-training
```

这将只使用GPU 0和1。

## 不使用Docker的手动安装

如果不想使用Docker，可以：

1. 运行导入修复脚本以解决相对导入问题：

```bash
python fix_imports.py
```

2. 设置正确的PYTHONPATH环境变量：

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/ml-experiments
```

3. 运行训练脚本：

```bash
cd ml-experiments
python main.py train --config config_cloud.yaml
```

## 自定义数据集和模型

如果需要使用自定义数据集或模型，请参考项目的`README.md`中相关部分的说明。 