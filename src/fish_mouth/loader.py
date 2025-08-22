"""
loader.py

该模块用于加载鱼嘴行为分析系统的主配置文件（YAML），
并负责以下任务：
1. 加载并解析配置文件为 Python 字典
2. 检查配置中必须的字段是否存在（包括顶层字段和模块字段）
3. 自动创建标准输出目录结构（videos, csv, plots, logs）
4. 返回可供主流程 pipeline 使用的标准化配置对象
"""

import os
import yaml


def load_yaml(path: str) -> dict:
    """
    加载 YAML 配置文件为字典。

    参数:
        path (str): 配置文件路径

    返回:
        dict: 解析后的配置内容

    异常:
        FileNotFoundError: 如果配置文件不存在
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[配置错误] 未找到配置文件: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_required_fields(cfg: dict):
    """
    检查配置字典中必须存在的字段。

    包括：
    - 顶层字段（如 input_video, output_base）
    - 模块字段（如 model.fish_head_path 等）

    参数:
        cfg (dict): 从 YAML 加载后的配置字典

    异常:
        KeyError: 如果缺少任何必要字段
    """
    # 顶层字段
    top_level_required = ["input_video", "output_base"]
    for key in top_level_required:
        if key not in cfg:
            raise KeyError(f"[配置错误] 缺少顶层字段: '{key}'")

    # 模块字段（可扩展）
    module_required = {
        "model": ["fish_head_path", "fish_mouth_path"],
        "analysis": ["covariate_col"],
        # "preprocess": []  # 当前全为可选
        # "crop": []
    }

    for module, fields in module_required.items():
        if module in cfg:
            for f in fields:
                if f not in cfg[module]:
                    raise KeyError(f"[配置错误] 模块 '{module}' 缺少字段: '{f}'")


def ensure_output_subdirs(output_base: str, subdirs: list = ["videos", "csv", "plots", "logs"]):
    """
    在输出目录下创建标准的子目录结构。

    参数:
        output_base (str): 输出根目录路径
        subdirs (list): 子文件夹名称列表（默认包含常用 4 项）
    """
    for sub in subdirs:
        path = os.path.join(output_base, sub)
        os.makedirs(path, exist_ok=True)


def load_config(config_path: str) -> dict:
    """
    加载配置文件、校验必要字段，并创建输出目录。

    参数:
        config_path (str): YAML 配置文件路径

    返回:
        dict: 经过校验和初始化的配置对象
    """
    print(f"📄 正在加载配置文件: {config_path}")
    cfg = load_yaml(config_path)

    validate_required_fields(cfg)
    ensure_output_subdirs(cfg["output_base"])

    print(f"✅ 配置加载完成，输出目录已准备好: {cfg['output_base']}")
    return cfg