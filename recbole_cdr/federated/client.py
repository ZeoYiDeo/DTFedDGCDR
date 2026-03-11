"""
recbole_cdr.federated.client
####################################
联邦学习客户端
职责：数据持有、参数打包上传、接收全局参数
训练循环由 FederatedDGCDRTrainer 统一调度
"""
from logging import getLogger


class FederatedClient:
    """
    联邦客户端。原始交互数据永不离开本节点。
    """

    def __init__(self, client_id: str, model, train_data, valid_data, config):
        self.client_id  = client_id
        self.model      = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.config     = config
        self.logger     = getLogger()
        self.data_size  = self._estimate_data_size()

    def _estimate_data_size(self) -> int:
        """
        估算本地训练集样本数（用于 FedAvg 加权）。

        CrossDomainDataLoader 的 dataset 是 CrossDomainDataset，
        没有 __len__，需要通过源域/目标域子数据集的交互矩阵行数来估算。
        优先取源域和目标域交互数之和；若仍取不到则回退到 batch 数估算。
        """
        dataset = getattr(self.train_data, 'dataset', None)
        if dataset is None:
            # train_data 本身可能就是 dataset
            dataset = self.train_data

        # 方式1：CrossDomainDataset —— 读源域 + 目标域交互数
        src_ds = getattr(dataset, 'source_domain_dataset', None)
        tgt_ds = getattr(dataset, 'target_domain_dataset', None)
        if src_ds is not None and tgt_ds is not None:
            try:
                src_size = len(src_ds.inter_feat)
                tgt_size = len(tgt_ds.inter_feat)
                return src_size + tgt_size
            except Exception:
                pass

        # 方式2：普通 Dataset，直接取交互数
        inter_feat = getattr(dataset, 'inter_feat', None)
        if inter_feat is not None:
            try:
                return len(inter_feat)
            except Exception:
                pass

        # 方式3：直接对 dataset 取 len
        try:
            return len(dataset)
        except TypeError:
            pass

        # 兜底：用 batch 数量估算
        self.logger.warning(
            f"[{self.client_id}] Cannot determine exact data size, "
            "falling back to batch-count estimate."
        )
        return len(self.train_data) * (self.config['train_batch_size'] if 'train_batch_size' in self.config else 2048)

    def get_local_update(self):
        """获取本地模型参数，准备上传服务器。"""
        state = self.model.get_local_model_state()
        return state, self.data_size

    def get_local_features(self):
        """提取解纠缠特征，用于服务器跨域对齐计算。"""
        return self.model.extract_disentangled_features()

    def apply_global_update(self, global_state: dict):
        """将服务器聚合参数注入本地模型。"""
        self.model.set_global_model_state(global_state)