"""
recbole_cdr.federated.server
####################################
联邦学习中央协调服务器

聚合策略：
  - 只聚合语义共享的参数（编码器/解码器 MLP）
  - 不聚合 embedding 层（各客户端实体ID不同，聚合无意义）
  - 不聚合图邻接矩阵（本地数据衍生）
"""
import copy
import torch
import torch.nn.functional as F
from logging import getLogger


# 只聚合包含这些关键词的参数层（MLP编解码器）
# embedding / norm_adj 等本地专属参数不参与聚合
_AGGREGATABLE_KEYWORDS = [
    'en_common_layers',
    'en_specific_layers',
    'de_layers',
    'en_item_common_layers',
    'en_item_specific_layers',
    'en_item_layers',
    'en_layers',
    'mapping',
    'item_mapping_layer',
]


def _is_aggregatable(param_name: str) -> bool:
    """判断该参数是否应参与联邦聚合。"""
    # embedding 层绝对不聚合（不同客户端实体空间不同）
    if 'embedding' in param_name:
        return False
    # 图结构相关不聚合
    if 'norm_adj' in param_name or 'interaction_matrix' in param_name:
        return False
    # 检查是否是共享的 MLP 层
    return any(kw in param_name for kw in _AGGREGATABLE_KEYWORDS)


class FederatedCentralServer:
    """
    联邦学习中央服务器。
    只聚合编码器/解码器 MLP 参数，不聚合 embedding。
    """

    def __init__(self, config):
        self.config = config
        self.logger = getLogger()
        self.num_clients = config['num_federated_clients'] if 'num_federated_clients' in config else 2
        self.global_model_state = None
        self._client_updates  = {}
        self._client_features = {}
        self.communication_rounds = 0
        self.alignment_history  = []
        self.separation_history = []

    def receive_local_update(self, client_id: str, local_state: dict, data_size: int):
        self._client_updates[client_id] = (local_state, data_size)
        self.logger.debug(
            f"[Server] Received update from '{client_id}', data_size={data_size}"
        )

    def receive_features(self, client_id: str, features: dict):
        self._client_features[client_id] = features

    def aggregate_models(self):
        """
        FedAvg 聚合：只聚合共享MLP参数，embedding保持各自独立。
        """
        if not self._client_updates:
            raise ValueError("[Server] No client updates received.")

        total_data = sum(size for _, size in self._client_updates.values())
        weights = {
            cid: size / total_data
            for cid, (_, size) in self._client_updates.items()
        }
        self.logger.debug(f"[Server] FedAvg weights: {weights}")

        # 收集所有客户端的参数
        all_states = {
            cid: state for cid, (state, _) in self._client_updates.items()
        }
        client_ids = list(all_states.keys())
        first_state = all_states[client_ids[0]]

        # 初始化聚合结果（先用第一个客户端的参数填充，作为不聚合层的默认值）
        aggregated = {k: v.float().clone() for k, v in first_state.items()}

        # 统计聚合了哪些层（用于日志）
        aggregated_params = []
        skipped_params = []

        for param_name in first_state.keys():
            if not _is_aggregatable(param_name):
                # 不聚合：各客户端保留自己的（此处全局模型存第一个客户端的，
                # 每个客户端接收后会用 set_global_model_state 选择性更新）
                skipped_params.append(param_name)
                continue

            # 检查所有客户端该参数形状是否一致
            shapes = [all_states[cid][param_name].shape
                      for cid in client_ids if param_name in all_states[cid]]
            if len(set(str(s) for s in shapes)) > 1:
                self.logger.debug(
                    f"[Server] Shape mismatch for '{param_name}': {shapes}, skip."
                )
                skipped_params.append(param_name)
                continue

            # FedAvg 加权平均
            agg_val = torch.zeros_like(first_state[param_name], dtype=torch.float32)
            for cid in client_ids:
                if param_name in all_states[cid]:
                    agg_val += weights[cid] * all_states[cid][param_name].float()
            aggregated[param_name] = agg_val
            aggregated_params.append(param_name)

        self.logger.debug(
            f"[Server] Aggregated {len(aggregated_params)} param groups, "
            f"skipped {len(skipped_params)} (embedding/local layers)."
        )

        self.global_model_state = aggregated
        self.communication_rounds += 1
        self._client_updates.clear()
        return self.global_model_state

    def compute_feature_alignment(self):
        if len(self._client_features) < 2:
            return 0.0, 0.0

        client_ids = list(self._client_features.keys())
        feat_A = self._client_features[client_ids[0]]
        feat_B = self._client_features[client_ids[1]]

        alignment_loss = 0.0
        separation_loss = 0.0
        n_terms = 0

        for key in ['source_common', 'target_common']:
            if key in feat_A and key in feat_B:
                a = F.normalize(feat_A[key].float(), dim=1)
                b = F.normalize(feat_B[key].float(), dim=1)
                min_len = min(a.shape[0], b.shape[0])
                cos_sim = torch.sum(a[:min_len] * b[:min_len], dim=1).mean().item()
                alignment_loss += (1.0 - cos_sim)
                n_terms += 1

        for key in ['source_specific', 'target_specific']:
            if key in feat_A and key in feat_B:
                a = F.normalize(feat_A[key].float(), dim=1)
                b = F.normalize(feat_B[key].float(), dim=1)
                min_len = min(a.shape[0], b.shape[0])
                cos_sim = torch.sum(a[:min_len] * b[:min_len], dim=1).mean().item()
                separation_loss += cos_sim
                n_terms += 1

        if n_terms > 0:
            alignment_loss /= max(n_terms // 2, 1)
            separation_loss /= max(n_terms // 2, 1)

        self.logger.debug(
            f"[Server] align={alignment_loss:.4f}, sep={separation_loss:.4f}"
        )
        self._client_features.clear()
        return alignment_loss, separation_loss

    def get_global_model_state(self):
        if self.global_model_state is None:
            raise ValueError("[Server] Global model not initialized.")
        return copy.deepcopy(self.global_model_state)

    def get_stats(self):
        return {
            'communication_rounds': self.communication_rounds,
            'alignment_history':    self.alignment_history,
            'separation_history':   self.separation_history,
        }