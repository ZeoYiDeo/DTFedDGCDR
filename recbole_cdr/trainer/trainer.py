r"""
recbole_cdr.trainer.trainer
################################
"""

import copy
import numpy as np
from recbole.trainer import Trainer
from recbole_cdr.utils import train_mode2state


class CrossDomainTrainer(Trainer):
    r"""Trainer for training cross-domain models."""

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']
        self._federated_epoch_override = None

    def _reinit(self, phase):
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        if self._federated_epoch_override is not None:
            self.epochs = int(self._federated_epoch_override)
        else:
            self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True,
            show_progress=False, callback_fn=None):
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
            else:
                super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result


class DCDCSRTrainer(Trainer):
    r"""Trainer for training DCDCSR models."""

    def __init__(self, config, model):
        super(DCDCSRTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']
        self._federated_epoch_override = None

    def _reinit(self, phase):
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        if self._federated_epoch_override is not None:
            self.epochs = int(self._federated_epoch_override)
        else:
            self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True,
            show_progress=False, callback_fn=None):
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if scheme == 'BOTH':
                super().fit(train_data, None, verbose, saved, show_progress, callback_fn)
            else:
                if self.split_valid_flag and valid_data is not None:
                    source_valid_data, target_valid_data = valid_data
                    if scheme == 'SOURCE':
                        super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                    else:
                        super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result


# =============================================================================
# [联���改造] 联邦学习 DGCDR 训练器
# =============================================================================

class FederatedDGCDRTrainer:
    """
    联邦DGCDR训练管理器。

    特性:
    - 5阶段联邦训练循环
    - 每个客户端独立保存自己的最佳模型参数（in-memory，不写磁盘）
    - 联邦早停：所有客户端连续 fed_stopping_step 轮无提升则停止
    - 早停/训练结束后自动将每个客户端恢复到各自最佳轮次的参数
    - 每轮打印各客户端详细验证指标并标注客户端标签
    """

    def __init__(self, config, clients, server, local_trainers,
                 client_labels: dict = None):
        """
        Args:
            config: CDRConfig
            clients: List[FederatedClient]
            server: FederatedCentralServer
            local_trainers: List[CrossDomainTrainer]
            client_labels: {client_id: 描述字符串}
        """
        import logging
        self.config         = config
        self.clients        = clients
        self.server         = server
        self.local_trainers = local_trainers
        self.logger         = logging.getLogger()

        self.num_rounds        = config['num_federated_rounds'] if 'num_federated_rounds' in config else 10
        self.local_epochs      = config['local_epochs'] if 'local_epochs' in config else 1
        self.fed_stopping_step = config['fed_stopping_step'] if 'fed_stopping_step' in config else 5

        self.client_labels = client_labels or {c.client_id: c.client_id for c in clients}

        # ── 早停状态 ──────────────────────────────────────────────────
        self._best_scores   = {c.client_id: -np.inf for c in clients}
        self._best_results  = {c.client_id: None    for c in clients}
        self._best_round    = {c.client_id: 0       for c in clients}
        self._no_improve    = {c.client_id: 0       for c in clients}

        # ── 最佳模型参数快照（in-memory deep copy，不写磁盘）─────────
        # {client_id: state_dict}
        self._best_model_states: dict = {c.client_id: None for c in clients}

        self.round_stats = []

    # ------------------------------------------------------------------
    # 早停辅助
    # ------------------------------------------------------------------
    def _update_early_stop(self, client_id: str, score: float,
                           result: dict, model) -> bool:
        """
        更新早停计数，若有提升则保存当前模型快照。

        Returns:
            bool: True 表示本轮有提升
        """
        if score > self._best_scores[client_id]:
            self._best_scores[client_id]  = score
            self._best_results[client_id] = result
            self._best_round[client_id]   = self.server.communication_rounds
            self._no_improve[client_id]   = 0
            # ★ 保存最佳模型参数快照（深拷贝到 CPU，节省显存）
            self._best_model_states[client_id] = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            return True
        else:
            self._no_improve[client_id] += 1
            return False

    def _should_stop(self) -> bool:
        """所有客户端均无提升才触发早停。"""
        return all(
            cnt >= self.fed_stopping_step
            for cnt in self._no_improve.values()
        )

    def _restore_best_models(self):
        """
        将每个客户端的模型恢复到各自最佳轮次的参数快照。
        在早停触发或训练正常结束后调用。
        """
        for client in self.clients:
            cid   = client.client_id
            label = self.client_labels.get(cid, cid)
            best_state = self._best_model_states[cid]

            if best_state is None:
                self.logger.warning(
                    f"[Federated] Client {label}: no best model snapshot, "
                    "keeping current parameters."
                )
                continue

            # 将快照参数加载回模型（移回 GPU）
            device = client.model.device
            state_on_device = {k: v.to(device) for k, v in best_state.items()}
            client.model.load_state_dict(state_on_device, strict=False)
            client.model.init_restore_e()

            self.logger.info(
                f"[Federated] Client {label}: restored best model "
                f"(round={self._best_round[cid]}, "
                f"valid_score={self._best_scores[cid]:.4f})"
            )

    # ------------------------------------------------------------------
    # 格式化工具
    # ------------------------------------------------------------------
    @staticmethod
    def _fmt_result(result: dict) -> str:
        if result is None:
            return "N/A"
        return "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in result.items()
        )

    # ------------------------------------------------------------------
    # 主训练循环
    # ------------------------------------------------------------------
    def run(self, test_data_list=None):
        self.logger.info(
            f"[Federated] Start: max {self.num_rounds} rounds × "
            f"{len(self.clients)} clients × {self.local_epochs} local epoch/round  "
            f"| early_stop_patience={self.fed_stopping_step}"
        )
        for cid, label in self.client_labels.items():
            self.logger.info(f"  {cid} => {label}")

        actual_rounds = 0
        stop_reason   = "max rounds reached"

        for fed_round in range(1, self.num_rounds + 1):
            actual_rounds = fed_round

            # ── 步骤1: 各客户端本地训练 ────────────────────────────
            round_client_info = {}

            for client, trainer in zip(self.clients, self.local_trainers):
                cid   = client.client_id
                label = self.client_labels.get(cid, cid)

                self.logger.info(
                    f"\n{'─'*70}\n"
                    f"[Fed Round {fed_round}/{self.num_rounds}] "
                    f"Local training  >>  Client {label}\n"
                    f"{'─'*70}"
                )

                trainer._federated_epoch_override = self.local_epochs
                trainer.cur_step    = 0
                trainer.start_epoch = 0

                best_score, best_result = trainer.fit(
                    client.train_data,
                    client.valid_data,
                    saved=False,
                    show_progress=False,
                )
                trainer._federated_epoch_override = None

                round_client_info[cid] = {
                    'score':  best_score,
                    'result': best_result,
                }

            # ── 步骤2: 特征提取 ─────────────────────────────────────
            for client in self.clients:
                self.server.receive_features(
                    client.client_id, client.get_local_features()
                )

            # ── 步骤3: FedAvg 聚合 ──────────────────────────────────
            for client in self.clients:
                local_state, data_size = client.get_local_update()
                self.server.receive_local_update(
                    client.client_id, local_state, data_size
                )
            global_state         = self.server.aggregate_models()
            align_loss, sep_loss = self.server.compute_feature_alignment()

            # ── 步骤4: 早停快照（广播之前，模型/分数状态完全一致）────
            # 验证分数来自本轮本地训练，模型也是本地训练后的状态，两者一致
            improved_flags = {}
            for client in self.clients:
                cid = client.client_id
                improved_flags[cid] = self._update_early_stop(
                    cid,
                    round_client_info[cid]['score'],
                    round_client_info[cid]['result'],
                    client.model,   # ← 本地训练后、广播前的模型
                )

            # ── 步骤5: 广播全局模型 + 设置 FedProx 参考点 ──────────
            for client in self.clients:
                client.apply_global_update(global_state)
                # [P1] FedProx：将全局聚合参数作为下一轮本地训练的近端约束参考
                if hasattr(client.model, 'set_fedprox_ref'):
                    client.model.set_fedprox_ref(global_state)

            # ── 本轮汇总日志 ────────────────────────────────────────
            round_info = {
                'round':                fed_round,
                'clients':              round_client_info,
                'alignment_loss':       align_loss,
                'separation_loss':      sep_loss,
                'communication_rounds': self.server.communication_rounds,
                'no_improve_counts':    dict(self._no_improve),
            }
            self.round_stats.append(round_info)

            self.logger.info(
                f"\n{'═'*70}\n"
                f"[Fed Round {fed_round:>3}/{self.num_rounds}]  "
                f"align={align_loss:.4f}  sep={sep_loss:.4f}\n"
                f"{'─'*70}"
            )
            for client in self.clients:
                cid    = client.client_id
                label  = self.client_labels.get(cid, cid)
                score  = round_client_info[cid]['score']
                result = round_client_info[cid]['result']
                no_imp = self._no_improve[cid]
                flag   = "↑ NEW BEST" if improved_flags[cid] \
                         else f"no improve {no_imp}/{self.fed_stopping_step}"
                self.logger.info(
                    f"  Client {label}\n"
                    f"    valid_score = {score:.4f}  [{flag}]"
                    f"  (best so far = {self._best_scores[cid]:.4f}"
                    f"  @ round {self._best_round[cid]})\n"
                    f"    {self._fmt_result(result)}"
                )
            self.logger.info(f"{'═'*70}")

            # ── 早停检查 ────────────────────────────────────────────
            if self._should_stop():
                stop_reason = (
                    f"early stopping: all clients no improvement "
                    f"for {self.fed_stopping_step} rounds"
                )
                self.logger.info(
                    f"\n[Federated] *** Early stopping at round {fed_round} ***\n"
                    f"  {stop_reason}"
                )
                break

        # ── 恢复最佳模型 ────────────────────────────────────────────
        self.logger.info(
            f"\n[Federated] Restoring each client's best model before testing..."
        )
        self._restore_best_models()

        # ── 训练结束摘要 ────────────────────────────────────────────
        self.logger.info(
            f"\n{'═'*70}\n"
            f"[Federated] Training finished  ({stop_reason})\n"
            f"  Actual rounds     : {actual_rounds} / {self.num_rounds}\n"
            f"  Total comm rounds : {self.server.communication_rounds}\n"
            f"{'─'*70}"
        )
        for client in self.clients:
            cid   = client.client_id
            label = self.client_labels.get(cid, cid)
            self.logger.info(
                f"  Client {label}\n"
                f"    best_valid_score = {self._best_scores[cid]:.4f}"
                f"  @ fed_round {self._best_round[cid]}\n"
                f"    {self._fmt_result(self._best_results[cid])}"
            )
        self.logger.info(f"{'═'*70}\n")

        return {
            'round_stats':   self.round_stats,
            'server_stats':  self.server.get_stats(),
            'best_scores':   dict(self._best_scores),
            'best_results':  dict(self._best_results),
            'best_rounds':   dict(self._best_round),
            'actual_rounds': actual_rounds,
            'stop_reason':   stop_reason,
        }