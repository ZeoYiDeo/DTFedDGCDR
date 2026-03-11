"""
recbole_cdr.utils.utils
############################
"""
import importlib

from recbole_cdr.utils.enum_type import ModelType


def get_model(model_name):
    model_submodule = ['cross_domain_recommender']
    model_file_name = model_name.lower()
    model_module = None
    for submodule in model_submodule:
        module_path = '.'.join(['recbole_cdr.model', submodule, model_file_name])
        if importlib.util.find_spec(module_path, package=None) is not None:
            model_module = importlib.import_module(module_path)
            break

    if model_module is None:
        raise ValueError(f'`model_name` [{model_name}] is not the name of an existing model.')
    model_class = getattr(model_module, model_name)
    return model_class


def get_trainer(model_type, model_name):
    """
    获取训练器类。

    [联邦改造] 新增:
    - 若 model_name 以 'Federated' 开头，返回 FederatedDGCDRTrainer（占位，实际由 run_federated_cdr.py 控制）
    - 其余逻辑与原始保持一致
    """
    # 尝试加载模型专属训练器
    trainer_submodule = 'recbole_cdr.trainer.trainer'
    trainer_module = importlib.import_module(trainer_submodule)

    # 模型专属 Trainer（优先级最高）
    model_specific_trainer = model_name + 'Trainer'
    if hasattr(trainer_module, model_specific_trainer):
        return getattr(trainer_module, model_specific_trainer)

    # 按模型类型选择通用 Trainer
    type2trainer = {
        ModelType.CROSSDOMAIN: 'CrossDomainTrainer',
    }
    trainer_name = type2trainer.get(model_type, 'Trainer')
    if hasattr(trainer_module, trainer_name):
        return getattr(trainer_module, trainer_name)

    # 降级到 RecBole 基础 Trainer
    from recbole.trainer import Trainer
    return Trainer


def get_keys_from_chainmap_by_order(chain_map):
    """
    Get all keys of the chainmap in order.
    """
    keys = []
    for mapping in chain_map.maps:
        for key in mapping.keys():
            if key not in keys:
                keys.append(key)
    return keys