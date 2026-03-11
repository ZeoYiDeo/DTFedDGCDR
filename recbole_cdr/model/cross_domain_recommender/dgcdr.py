import copy
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization, xavier_normal_
from torch.nn.init import kaiming_uniform_, constant_, xavier_uniform_
from recbole.model.loss import EmbLoss
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss

from recbole.utils import InputType


class DGCDR(CrossDomainRecommender):
    # BCE Loss
    # input_type = InputType.POINTWISE

    # BPR Loss
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DGCDR, self).__init__(config, dataset)

        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'
        self.phase = 'both'

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.tem = config['temperature']

        self.drop_rate = config['drop_rate']
        self.connect_way = config['connect_way']
        self.preference_disentangle = config['preference_disentangle']
        self.fuse_mode = config['fuse_mode']
        self.loss_type = config['loss_type']
        self.attention_mode = config['attention_mode']
        self.concat_mode = config['concat_mode']
        self.cl_sim_weight = config['cl_sim_weight']
        self.cl_org_weight = config['cl_org_weight']
        self.cl_decoder_weight = config['cl_decoder_weight']
        self.item_negative = config['item_negative']
        self.item_cl_weight = config['item_cl_weight']
        self.item_mapping = config['item_mapping']
        self.item_disentangle = config['item_disentangle']
        self.feature_mapping_way = config['feature_mapping_way']
        mlp_hidden_size = config['mlp_hidden_size']
        activation_func = config['activation_func']
        init_way = config['init_way']

        # ===== [联邦改造] 新增联邦学习相关配置 =====
        self.federated_mode = config['federated_mode'] if 'federated_mode' in config else False
        # 联邦评估方向: 'target'(原始), 'source'(反向), 'both'(双向)
        self.eval_direction = config['eval_direction'] if 'eval_direction' in config else 'target'

        # [P1] FedProx 近端约束：防止客户端在本地训练时偏离全局模型过远
        self.fedprox_mu = config['fedprox_mu'] if 'fedprox_mu' in config else 0.0
        self._fedprox_global_params = None  # 由 set_fedprox_ref() 在每轮训练前注入

        # [P2] 自监督图对比学习 (SGL)：两个 edge-dropout 增强视图之间的 InfoNCE 损失
        self.ssl_weight = config['ssl_weight'] if 'ssl_weight' in config else 0.0
        self.ssl_aug_ratio = config['ssl_aug_ratio'] if 'ssl_aug_ratio' in config else 0.1
        self.ssl_temp = config['ssl_temp'] if 'ssl_temp' in config else 0.2

        # [P2] 跨域重叠用户 common 特征融合门控
        self.use_cross_domain_fusion = config['use_cross_domain_fusion'] if 'use_cross_domain_fusion' in config else False

        # define layers and loss
        self.source_user_embedding = nn.Embedding(num_embeddings=self.total_num_users,
                                                  embedding_dim=self.embedding_size, device=self.device)
        self.target_user_embedding = nn.Embedding(num_embeddings=self.total_num_users,
                                                  embedding_dim=self.embedding_size, device=self.device)

        self.source_item_embedding = nn.Embedding(num_embeddings=self.total_num_items,
                                                  embedding_dim=self.embedding_size, device=self.device)
        self.target_item_embedding = nn.Embedding(num_embeddings=self.total_num_items,
                                                  embedding_dim=self.embedding_size, device=self.device)

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        # mapping, encoding and decoding
        en_input_size = [self.embedding_size * (self.n_layers + 1)] if self.connect_way == 'concat' else [config[
                                                                                                              'embedding_size']]
        if self.connect_way == 'concat':
            en_input_size = [self.embedding_size * (self.n_layers + 1)]
        elif self.connect_way == 'mean':
            en_input_size = [config['embedding_size']]

        if self.preference_disentangle:
            if self.feature_mapping_way == 'projection':
                mlp_hidden_size = en_input_size
                # for users
                self.source_en_common_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                         activation='none')
                self.source_en_specific_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                           activation='none')
                self.source_de_layers = MLPLayers(mlp_hidden_size + en_input_size, self.drop_rate,
                                                  activation='none')
                self.target_en_common_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                         activation='none')
                self.target_en_specific_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                           activation='none')
                self.target_de_layers = MLPLayers(mlp_hidden_size + en_input_size, self.drop_rate,
                                                  activation='none')
                # for items
                self.source_en_item_common_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                              activation='none')
                self.source_en_item_specific_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                                activation='none')
                self.target_en_item_common_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                              activation='none')
                self.target_en_item_specific_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                                activation='none')
            elif self.feature_mapping_way == 'mlp':
                # for users
                self.source_en_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                  activation=activation_func)
                self.source_de_layers = MLPLayers(mlp_hidden_size + en_input_size, self.drop_rate,
                                                  activation=activation_func)
                self.target_en_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                  activation=activation_func)
                self.target_de_layers = MLPLayers(mlp_hidden_size + en_input_size, self.drop_rate,
                                                  activation=activation_func)
                # for items
                self.source_en_item_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                       activation=activation_func)
                self.target_en_item_layers = MLPLayers(en_input_size + mlp_hidden_size, self.drop_rate,
                                                       activation=activation_func)
            if self.fuse_mode == 'concat':
                if self.concat_mode == 'all':
                    mapping_input = [en_input_size[0] + 2 * int(mlp_hidden_size[0])]
                elif self.concat_mode == 'part':
                    mapping_input = [2 * int(mlp_hidden_size[0])]
                self.mapping = MLPLayers(mapping_input + mlp_hidden_size, self.drop_rate, activation=activation_func)
            if (not self.item_disentangle) & self.item_mapping:
                self.source_item_mapping_layer = MLPLayers(en_input_size + en_input_size, self.drop_rate,
                                                           activation=activation_func)
                self.target_item_mapping_layer = MLPLayers(en_input_size + en_input_size, self.drop_rate,
                                                           activation=activation_func)

            self.sim_loss = torch.nn.CosineEmbeddingLoss(margin=0.5)

            # [P2] 跨域重叠用户 common 特征融合门控
            if self.use_cross_domain_fusion:
                self.cross_domain_gate = nn.Linear(en_input_size[0] * 2, en_input_size[0])

        self.rec_loss = None
        if self.loss_type == 'BPR':
            self.rec_loss = BPRLoss()
        elif self.loss_type == 'BCE' or self.loss_type == 'CE':
            self.rec_loss = nn.BCELoss()
            self.sigmoid = nn.Sigmoid()
        self.reg_loss = EmbLoss()
        self.dropout = nn.Dropout(p=self.drop_rate)

        # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(
            np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(
            np.float32)
        self.source_norm_adj_matrix = self.get_norm_adj_mat(self.source_interaction_matrix, self.total_num_users,
                                                            self.total_num_items).to(self.device)
        self.target_norm_adj_matrix = self.get_norm_adj_mat(self.target_interaction_matrix, self.total_num_users,
                                                            self.total_num_items).to(self.device)

        # storage variables for full sort evaluation acceleration
        self.target_restore_user_e = None
        self.target_restore_item_e = None
        # ===== [联邦改造] 新增：源域评估缓存（支持双向推荐）=====
        self.source_restore_user_e = None
        self.source_restore_item_e = None

        if init_way == 'kaiming':
            self.apply(self._kaiming_init_weights)
        elif init_way == 'xavier':
            self.apply(xavier_normal_initialization)

        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    def _kaiming_init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Embedding):
            kaiming_uniform_(module.weight.data)
        elif isinstance(module, nn.Linear):
            kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None):
        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
            norm_adj_matrix = self.source_norm_adj_matrix
        else:
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
            norm_adj_matrix = self.target_norm_adj_matrix
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix

    def graph_layer(self, adj_matrix, all_embeddings):
        side_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
        new_embeddings = side_embeddings + torch.mul(all_embeddings, side_embeddings)
        new_embeddings = all_embeddings + new_embeddings
        new_embeddings = self.dropout(new_embeddings)
        return new_embeddings

    def fuse_and_update(self, common_preference, specific_preference, user_embeddings):
        if self.fuse_mode == 'concat':
            if self.concat_mode == 'all':
                user_all_embeddings = self.mapping(
                    torch.cat((user_embeddings, common_preference, specific_preference), -1))
            elif self.concat_mode == 'part':
                user_all_embeddings = self.mapping(torch.cat((common_preference, specific_preference), -1))

        elif self.fuse_mode == 'attention':
            a_1 = torch.sum(torch.mul(user_embeddings, common_preference), dim=1)
            a_2 = torch.sum(torch.mul(user_embeddings, specific_preference), dim=1)
            b_1 = a_1.unsqueeze(1)
            b_2 = a_2.unsqueeze(1)
            att = torch.cat((b_1, b_2), dim=1)
            softed_att = F.softmax(att, dim=1)
            c_1 = softed_att[:, 0].unsqueeze(1).repeat(1, common_preference.shape[1])
            c_2 = softed_att[:, 1].unsqueeze(1).repeat(1, specific_preference.shape[1])
            e_c = c_1 * common_preference
            e_s = c_2 * specific_preference
            if self.attention_mode == 'all':
                user_all_embeddings = user_embeddings + e_c + e_s
            elif self.attention_mode == 'part':
                user_all_embeddings = e_c + e_s
        return user_all_embeddings

    def disentangle_layer(self, source_embeddings, target_embeddings, is_user=True):
        if is_user:
            if self.overlapped_num_users > 1:
                source_independent_user_embeddings = source_embeddings[self.overlapped_num_users:]
                target_independent_user_embeddings = target_embeddings[self.overlapped_num_users:]
                source_overlap_users = source_embeddings[:self.overlapped_num_users]
                target_overlap_users = target_embeddings[:self.overlapped_num_users]

                if self.feature_mapping_way == 'projection':
                    source_common_preference = source_overlap_users * torch.sigmoid(
                        self.source_en_common_layers(source_overlap_users))
                    source_specific_preference = source_overlap_users * torch.sigmoid(
                        self.source_en_specific_layers(source_overlap_users))
                    target_common_preference = target_overlap_users * torch.sigmoid(
                        self.target_en_common_layers(target_overlap_users))
                    target_specific_preference = target_overlap_users * torch.sigmoid(
                        self.target_en_specific_layers(target_overlap_users))
                    source_decode_user = source_overlap_users * torch.sigmoid(
                        self.source_de_layers(source_overlap_users))
                    source_decode_common = source_common_preference * torch.sigmoid(
                        self.source_de_layers(source_common_preference))
                    source_decode_specific = source_specific_preference * torch.sigmoid(
                        self.source_de_layers(source_specific_preference))
                    target_decode_user = target_overlap_users * torch.sigmoid(
                        self.target_de_layers(target_overlap_users))
                    target_decode_common = target_common_preference * torch.sigmoid(
                        self.target_de_layers(target_common_preference))
                    target_decode_specific = target_specific_preference * torch.sigmoid(
                        self.target_de_layers(target_specific_preference))
                elif self.feature_mapping_way == 'mlp':
                    source_common_preference = self.source_en_layers(source_overlap_users)
                    source_specific_preference = source_overlap_users - source_common_preference
                    target_common_preference = self.target_en_layers(target_overlap_users)
                    target_specific_preference = target_overlap_users - target_common_preference
                    source_decode_user = self.source_de_layers(source_overlap_users)
                    source_decode_common = self.source_de_layers(source_common_preference)
                    source_decode_specific = self.source_de_layers(source_specific_preference)
                    target_decode_user = self.target_de_layers(target_overlap_users)
                    target_decode_common = self.target_de_layers(target_common_preference)
                    target_decode_specific = self.target_de_layers(target_specific_preference)

                # [P2] 跨域 common 特征融合门控：在 fuse_and_update 之前，
                # 让重叠用户的 common 特征在两个域之间软融合，增强跨域知识迁移
                if self.use_cross_domain_fusion:
                    gate = torch.sigmoid(self.cross_domain_gate(
                        torch.cat([source_common_preference, target_common_preference], dim=-1)
                    ))
                    src_fused = gate * source_common_preference + (1 - gate) * target_common_preference
                    tgt_fused = (1 - gate) * source_common_preference + gate * target_common_preference
                    source_common_preference = src_fused
                    target_common_preference = tgt_fused

                source_update_overlap_user_embeddings = self.fuse_and_update(source_common_preference,
                                                                             source_specific_preference,
                                                                             source_overlap_users)
                target_update_overlap_user_embeddings = self.fuse_and_update(target_common_preference,
                                                                             target_specific_preference,
                                                                             target_overlap_users)

                source_all_user_embeddings = torch.cat(
                    [source_update_overlap_user_embeddings, source_independent_user_embeddings], dim=0)
                target_all_user_embeddings = torch.cat(
                    [target_update_overlap_user_embeddings, target_independent_user_embeddings], dim=0)

                return source_all_user_embeddings, target_all_user_embeddings, [source_common_preference,
                                                                                target_common_preference,
                                                                                source_specific_preference,
                                                                                target_specific_preference,
                                                                                source_overlap_users,
                                                                                target_overlap_users,
                                                                                source_decode_user,
                                                                                source_decode_common,
                                                                                source_decode_specific,
                                                                                target_decode_user,
                                                                                target_decode_common,
                                                                                target_decode_specific]
        else:
            if self.feature_mapping_way == 'projection':
                source_common_feature = source_embeddings * torch.sigmoid(
                    self.source_en_item_common_layers(source_embeddings))
                source_specific_feature = source_embeddings * torch.sigmoid(
                    self.source_en_item_specific_layers(source_embeddings))
                target_common_feature = target_embeddings * torch.sigmoid(
                    self.target_en_item_common_layers(target_embeddings))
                target_specific_feature = target_embeddings * torch.sigmoid(
                    self.target_en_item_specific_layers(target_embeddings))
            elif self.feature_mapping_way == 'mlp':
                source_common_feature = self.source_en_item_layers(source_embeddings)
                source_specific_feature = source_embeddings - source_common_feature
                target_common_feature = self.target_en_item_layers(target_embeddings)
                target_specific_feature = target_embeddings - target_common_feature

            source_update_item_embeddings = self.fuse_and_update(source_common_feature,
                                                                 source_specific_feature,
                                                                 source_embeddings)
            target_update_item_embeddings = self.fuse_and_update(target_common_feature,
                                                                 target_specific_feature,
                                                                 target_embeddings)
            return source_update_item_embeddings, target_update_item_embeddings

    def forward(self):
        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(domain='source')
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(domain='target')

        source_embeddings_list = [source_all_embeddings]
        target_embeddings_list = [target_all_embeddings]
        for layer_idx in range(self.n_layers):
            source_all_embeddings = self.graph_layer(source_norm_adj_matrix, source_all_embeddings)
            target_all_embeddings = self.graph_layer(target_norm_adj_matrix, target_all_embeddings)
            source_embeddings_list.append(source_all_embeddings)
            target_embeddings_list.append(target_all_embeddings)

        if self.connect_way == 'concat':
            source_lightgcn_all_embeddings = torch.cat(source_embeddings_list, 1)
            target_lightgcn_all_embeddings = torch.cat(target_embeddings_list, 1)
        elif self.connect_way == 'mean':
            source_lightgcn_all_embeddings = torch.stack(source_embeddings_list, dim=1)
            source_lightgcn_all_embeddings = torch.mean(source_lightgcn_all_embeddings, dim=1)
            target_lightgcn_all_embeddings = torch.stack(target_embeddings_list, dim=1)
            target_lightgcn_all_embeddings = torch.mean(target_lightgcn_all_embeddings, dim=1)

        source_user_embeddings, source_item_embeddings = torch.split(source_lightgcn_all_embeddings,
                                                                     [self.total_num_users,
                                                                      self.total_num_items])
        target_user_embeddings, target_item_embeddings = torch.split(target_lightgcn_all_embeddings,
                                                                     [self.total_num_users,
                                                                      self.total_num_items])
        if self.preference_disentangle:
            source_user_embeddings, target_user_embeddings, user_disentangled_list = self.disentangle_layer(
                source_user_embeddings, target_user_embeddings)
            if self.item_disentangle:
                source_item_embeddings, target_item_embeddings = self.disentangle_layer(source_item_embeddings,
                                                                                        target_item_embeddings, False)
            elif self.item_mapping:
                source_item_embeddings = source_item_embeddings * torch.sigmoid(
                    self.source_item_mapping_layer(source_item_embeddings))
                target_item_embeddings = target_item_embeddings * torch.sigmoid(
                    self.target_item_mapping_layer(target_item_embeddings))
        else:
            user_disentangled_list = []

        return user_disentangled_list, source_user_embeddings, source_item_embeddings, target_user_embeddings, target_item_embeddings

    # ===== [联邦改造] 新增：提取解纠缠特征（供服务器进行跨域对齐）=====
    def extract_disentangled_features(self):
        """
        提取当前模型的重叠用户解纠缠特征。
        返回共享特征和特定特征，用于联邦服务器计算跨客户端对齐损失。

        Returns:
            dict: {
                'source_common': Tensor[overlapped_num_users, feat_dim],
                'source_specific': Tensor[overlapped_num_users, feat_dim],
                'target_common': Tensor[overlapped_num_users, feat_dim],
                'target_specific': Tensor[overlapped_num_users, feat_dim],
            }
        """
        self.eval()
        with torch.no_grad():
            src_ego, src_adj = self.get_ego_embeddings(domain='source')
            tgt_ego, tgt_adj = self.get_ego_embeddings(domain='target')

            src_list = [src_ego]
            tgt_list = [tgt_ego]
            for _ in range(self.n_layers):
                src_ego = self.graph_layer(src_adj, src_ego)
                tgt_ego = self.graph_layer(tgt_adj, tgt_ego)
                src_list.append(src_ego)
                tgt_list.append(tgt_ego)

            if self.connect_way == 'concat':
                src_emb = torch.cat(src_list, 1)
                tgt_emb = torch.cat(tgt_list, 1)
            else:
                src_emb = torch.mean(torch.stack(src_list, dim=1), dim=1)
                tgt_emb = torch.mean(torch.stack(tgt_list, dim=1), dim=1)

            src_user_emb, _ = torch.split(src_emb, [self.total_num_users, self.total_num_items])
            tgt_user_emb, _ = torch.split(tgt_emb, [self.total_num_users, self.total_num_items])

            overlap_src = src_user_emb[:self.overlapped_num_users]
            overlap_tgt = tgt_user_emb[:self.overlapped_num_users]

            if self.feature_mapping_way == 'projection':
                src_common = overlap_src * torch.sigmoid(self.source_en_common_layers(overlap_src))
                src_specific = overlap_src * torch.sigmoid(self.source_en_specific_layers(overlap_src))
                tgt_common = overlap_tgt * torch.sigmoid(self.target_en_common_layers(overlap_tgt))
                tgt_specific = overlap_tgt * torch.sigmoid(self.target_en_specific_layers(overlap_tgt))
            elif self.feature_mapping_way == 'mlp':
                src_common = self.source_en_layers(overlap_src)
                src_specific = overlap_src - src_common
                tgt_common = self.target_en_layers(overlap_tgt)
                tgt_specific = overlap_tgt - tgt_common
            else:
                return {}

        self.train()
        return {
            'source_common': src_common.detach().cpu(),
            'source_specific': src_specific.detach().cpu(),
            'target_common': tgt_common.detach().cpu(),
            'target_specific': tgt_specific.detach().cpu(),
        }

    # ===== [联邦改造] 新增：获取本地模型参数（上传给联邦服务器）=====
    def get_local_model_state(self):
        """
        获取适合联邦聚合的模型参数（只返回可学习参数，不含图结构）。

        Returns:
            dict: 模型参数的深拷贝（CPU张量）
        """
        state = {}
        for k, v in self.state_dict().items():
            # 排除图交互矩阵（不参与联邦聚合，属于本地数据衍生）
            if 'norm_adj_matrix' not in k and 'interaction_matrix' not in k:
                state[k] = v.detach().cpu().clone()
        return state

    # ===== [联邦改造] 新增：接收全局聚合参数（从联邦服务器下载）=====
    def set_global_model_state(self, global_state: dict, strict=False):
        """
        只将联邦聚合的 MLP 参数注入本地模型。
        embedding 层保持本地独立，不被全局参数覆盖。
        """
        from recbole_cdr.federated.server import _is_aggregatable

        current_state = self.state_dict()
        updated_count = 0

        for k, v in global_state.items():
            # 只更新应聚合的 MLP 层，embedding 等本地层不动
            if k in current_state and _is_aggregatable(k):
                if v.shape == current_state[k].shape:
                    current_state[k] = v.to(self.device)
                    updated_count += 1

        self.load_state_dict(current_state, strict=False)
        self.init_restore_e()

    # ===== [P1] FedProx：近端约束辅助方法 =====

    def set_fedprox_ref(self, global_state: dict):
        """
        将当前轮次的全局聚合参数存为 FedProx 参考点。
        由 FederatedDGCDRTrainer 在每轮广播完成后调用。
        """
        from recbole_cdr.federated.server import _is_aggregatable
        self._fedprox_global_params = {
            k: v.to(self.device)
            for k, v in global_state.items()
            if _is_aggregatable(k)
        }

    def _compute_fedprox_loss(self):
        """FedProx 近端正则项：mu/2 * ||w_local - w_global||^2"""
        if self._fedprox_global_params is None or self.fedprox_mu == 0.0:
            return 0.0
        prox = torch.tensor(0.0, device=self.device)
        for name, param in self.named_parameters():
            if name in self._fedprox_global_params:
                global_p = self._fedprox_global_params[name]
                prox = prox + torch.norm(param - global_p) ** 2
        return (self.fedprox_mu / 2) * prox

    # ===== [P2] SGL：自监督图对比学习辅助方法 =====

    def _graph_aug_dropout(self, adj_matrix, drop_ratio: float):
        """随机 Edge Dropout 图增强：以 drop_ratio 概率丢弃边，生成增强图。"""
        indices = adj_matrix._indices()
        values = adj_matrix._values()
        mask = torch.rand(values.size(0), device=values.device) >= drop_ratio
        new_indices = indices[:, mask]
        new_values = values[mask]
        aug = torch.sparse_coo_tensor(new_indices, new_values, adj_matrix.shape,
                                      device=values.device)
        return aug.coalesce()

    def _gnn_propagate(self, ego_emb, adj_matrix):
        """运行 n_layers 层 GNN 并按 connect_way 聚合所有层输出。"""
        emb_list = [ego_emb]
        curr = ego_emb
        for _ in range(self.n_layers):
            curr = self.graph_layer(adj_matrix, curr)
            emb_list.append(curr)
        if self.connect_way == 'concat':
            return torch.cat(emb_list, dim=1)
        return torch.mean(torch.stack(emb_list, dim=1), dim=1)

    def _ssl_loss(self, emb1, emb2):
        """
        InfoNCE 对比损失（双向对称）。
        emb1, emb2: [N, D]，N 为 batch 内唯一用户数。
        """
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)
        sim = torch.mm(emb1, emb2.T) / self.ssl_temp
        labels = torch.arange(sim.size(0), device=sim.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
        return loss

    def calculate_loss(self, interaction):
        self.init_restore_e()
        user_disentangled_list, source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()

        losses = []
        if self.loss_type == 'BCE':
            source_user = interaction[self.SOURCE_USER_ID]
            source_item = interaction[self.SOURCE_ITEM_ID]
            source_label = interaction[self.SOURCE_LABEL]

            target_user = interaction[self.TARGET_USER_ID]
            target_item = interaction[self.TARGET_ITEM_ID]
            target_label = interaction[self.TARGET_LABEL]

            source_u_embeddings = source_user_all_embeddings[source_user]
            source_i_embeddings = source_item_all_embeddings[source_item]
            target_u_embeddings = target_user_all_embeddings[target_user]
            target_i_embeddings = target_item_all_embeddings[target_item]

            source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
            source_bce_loss = self.rec_loss(source_output, source_label)
            u_ego_embeddings = self.source_user_embedding(source_user)
            i_ego_embeddings = self.source_item_embedding(source_item)
            source_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)
            source_loss = source_bce_loss + self.reg_weight * source_reg_loss
            losses.append(source_loss)

            target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
            target_bce_loss = self.rec_loss(target_output, target_label)
            u_ego_embeddings = self.target_user_embedding(target_user)
            i_ego_embeddings = self.target_item_embedding(target_item)
            target_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)
            target_loss = target_bce_loss + self.reg_weight * target_reg_loss
            losses.append(target_loss)

        elif self.loss_type == 'BPR':
            source_user = interaction[self.SOURCE_USER_ID]
            source_item = interaction[self.SOURCE_ITEM_ID]
            source_neg_item = interaction[self.SOURCE_NEG_ITEM_ID]

            source_user_e = source_user_all_embeddings[source_user]
            source_item_e = source_item_all_embeddings[source_item]
            source_neg_item_e = source_item_all_embeddings[source_neg_item]

            source_pos_output = torch.mul(source_user_e, source_item_e).sum(dim=1)
            source_neg_output = torch.mul(source_user_e, source_neg_item_e).sum(dim=1)
            source_loss = self.rec_loss(source_pos_output, source_neg_output) + \
                          self.reg_weight * self.reg_loss(self.source_user_embedding(source_user),
                                                          self.source_item_embedding(source_item),
                                                          self.source_item_embedding(source_neg_item))
            losses.append(source_loss)

            target_user = interaction[self.TARGET_USER_ID]
            target_item = interaction[self.TARGET_ITEM_ID]
            target_neg_item = interaction[self.TARGET_NEG_ITEM_ID]

            target_user_e = target_user_all_embeddings[target_user]
            target_item_e = target_item_all_embeddings[target_item]
            target_neg_item_e = target_item_all_embeddings[target_neg_item]
            target_pos_output = torch.mul(target_user_e, target_item_e).sum(dim=1)
            target_neg_output = torch.mul(target_user_e, target_neg_item_e).sum(dim=1)
            target_loss = self.rec_loss(target_pos_output, target_neg_output) + \
                          self.reg_weight * self.reg_loss(self.target_user_embedding(target_user),
                                                          self.target_item_embedding(target_item),
                                                          self.target_item_embedding(target_neg_item))
            losses.append(target_loss)

        if self.preference_disentangle:
            sr_c, tg_c, sr_s, tg_s, sr_user, tg_user, sr_de_user, sr_de_c, sr_de_s, tg_de_user, tg_de_c, tg_de_s = user_disentangled_list

            is_sr_common_user = (source_user >= 0) & (source_user < self.overlapped_num_users)
            sr_common_user = source_user[is_sr_common_user]
            sr_common_s = sr_s[sr_common_user]
            sr_tg_common_s = tg_s[sr_common_user]
            sr_common_c = sr_c[sr_common_user]
            sr_tg_common_c = tg_c[sr_common_user]
            sr_common_user_e = sr_user[sr_common_user]

            is_tg_common_user = (target_user >= 0) & (target_user < self.overlapped_num_users)
            tg_common_user = target_user[is_tg_common_user]
            tg_common_s = tg_s[tg_common_user]
            tg_sr_common_s = sr_s[tg_common_user]
            tg_common_c = tg_c[tg_common_user]
            tg_sr_common_c = sr_c[tg_common_user]
            tg_common_user_e = tg_user[tg_common_user]

            sr_L_sim = self.sim_loss(sr_common_c, sr_tg_common_c, torch.ones(sr_common_c.size(0)).to(self.device))
            tg_L_sim = self.sim_loss(tg_common_c, tg_sr_common_c, torch.ones(tg_common_c.size(0)).to(self.device))

            L_sim = sr_L_sim + tg_L_sim
            losses.append(self.cl_sim_weight * L_sim)

            if self.cl_org_weight != 0:
                sr_L_ort_cs = torch.mean(torch.sum(torch.mul(sr_common_c, sr_common_s), dim=1) ** 2, dim=0)
                tg_L_ort_cs = torch.mean(torch.sum(torch.mul(tg_common_c, tg_common_s), dim=1) ** 2, dim=0)
                losses.extend(self.cl_org_weight * loss for loss in [sr_L_ort_cs, tg_L_ort_cs])

            if self.cl_decoder_weight != 0:
                sr_common_de_user = sr_de_user[tg_common_user]
                sr_common_de_c = sr_de_c[tg_common_user]
                sr_common_de_s = sr_de_s[tg_common_user]
                tg_common_de_user = tg_de_user[sr_common_user]
                tg_common_de_c = tg_de_c[sr_common_user]
                tg_common_de_s = tg_de_s[sr_common_user]

                decoder_loss_T2S = self.decoder_loss_function(sr_common_user_e, tg_common_de_user, tg_common_de_c,
                                                              tg_common_de_s, self.tem)
                decoder_loss_S2T = self.decoder_loss_function(tg_common_user_e, sr_common_de_user, sr_common_de_c,
                                                              sr_common_de_s, self.tem)
                losses.extend(self.cl_decoder_weight * loss for loss in [decoder_loss_T2S, decoder_loss_S2T])

            if self.item_negative:
                sr_tg_user_specific_e = tg_s[sr_common_user]
                sr_common_user_items = source_item[is_sr_common_user]
                sr_item_e = source_item_all_embeddings[sr_common_user_items]
                sr_item_loss = self.item_disentangle_loss(sr_common_s, sr_tg_user_specific_e, sr_item_e, self.tem)

                tg_sr_user_specific_e = sr_s[tg_common_user]
                tg_common_user_items = target_item[is_tg_common_user]
                tg_item_e = target_item_all_embeddings[tg_common_user_items]
                tg_item_loss = self.item_disentangle_loss(tg_common_s, tg_sr_user_specific_e, tg_item_e, self.tem)
                losses.extend(self.item_cl_weight * loss for loss in [sr_item_loss, tg_item_loss])

        # ── [P2] SGL：自监督图对比损失 ─────────────────────────────
        if self.ssl_weight > 0:
            batch_src_users = torch.unique(interaction[self.SOURCE_USER_ID])
            batch_tgt_users = torch.unique(interaction[self.TARGET_USER_ID])

            # 源域：两个 edge-dropout 增强视图
            src_ego, src_adj = self.get_ego_embeddings(domain='source')
            src_aug1 = self._graph_aug_dropout(src_adj, self.ssl_aug_ratio)
            src_aug2 = self._graph_aug_dropout(src_adj, self.ssl_aug_ratio)
            src_emb1 = self._gnn_propagate(src_ego, src_aug1)
            src_emb2 = self._gnn_propagate(src_ego, src_aug2)
            src_u1, _ = torch.split(src_emb1, [self.total_num_users, self.total_num_items])
            src_u2, _ = torch.split(src_emb2, [self.total_num_users, self.total_num_items])
            ssl_src = self._ssl_loss(src_u1[batch_src_users], src_u2[batch_src_users])

            # 目标域：两个 edge-dropout 增强视图
            tgt_ego, tgt_adj = self.get_ego_embeddings(domain='target')
            tgt_aug1 = self._graph_aug_dropout(tgt_adj, self.ssl_aug_ratio)
            tgt_aug2 = self._graph_aug_dropout(tgt_adj, self.ssl_aug_ratio)
            tgt_emb1 = self._gnn_propagate(tgt_ego, tgt_aug1)
            tgt_emb2 = self._gnn_propagate(tgt_ego, tgt_aug2)
            tgt_u1, _ = torch.split(tgt_emb1, [self.total_num_users, self.total_num_items])
            tgt_u2, _ = torch.split(tgt_emb2, [self.total_num_users, self.total_num_items])
            ssl_tgt = self._ssl_loss(tgt_u1[batch_tgt_users], tgt_u2[batch_tgt_users])

            losses.append(self.ssl_weight * (ssl_src + ssl_tgt))

        # ── [P1] FedProx：近端约束损失 ─────────────────────────────
        prox_loss = self._compute_fedprox_loss()
        if isinstance(prox_loss, torch.Tensor):
            losses.append(prox_loss)

        return tuple(losses)

    def decoder_loss_function(self, sr_user, tg_de_user, tg_de_c, tg_de_s, t):
        sr = F.normalize(sr_user, dim=1)
        tg = F.normalize(tg_de_user, dim=1)
        tg_c = F.normalize(tg_de_c, dim=1)
        tg_s = F.normalize(tg_de_s, dim=1)
        pos_1 = torch.sum(sr * tg_c, dim=1)
        pos_2 = torch.sum(sr * tg, dim=1)
        neg_1 = torch.sum(sr * tg_s, dim=1)
        pos_1_h = torch.exp(pos_1 / t)
        pos_2_h = torch.exp(pos_2 / t)
        neg_1_h = torch.exp(neg_1 / t)
        loss_1 = -torch.mean(torch.log(pos_1_h / (pos_1_h + pos_2_h) + 1e-24))
        loss_2 = -torch.mean(torch.log(pos_2_h / (pos_2_h + neg_1_h) + 1e-24))
        return loss_1 + loss_2

    def item_disentangle_loss(self, sr_dis_s, tg_dis_s, sr_positive_item, t):
        sr_s = F.normalize(sr_dis_s, dim=1)
        tg_s = F.normalize(tg_dis_s, dim=1)
        sr_item = F.normalize(sr_positive_item, dim=1)
        pos = torch.sum(sr_s * sr_item, dim=1)
        neg = torch.sum(tg_s * sr_item, dim=1)
        pos_h = torch.exp(pos / t)
        neg_h = torch.exp(neg / t)
        sr_loss = -torch.mean(torch.log(pos_h / (pos_h + neg_h) + 1e-24))
        return sr_loss

    def predict(self, interaction):
        """
        [联邦改造] 自动检测 interaction 中实际存在的字段，
        决定使用源域还是目标域嵌入，兼容所有评估场景。
        """
        user_disentangled_list, source_user_embeddings, source_item_embeddings, \
            target_user_embeddings, target_item_embeddings = self.forward()

        # 优先检测 interaction 中实际存在的字段（而非依赖 eval_direction）
        if self.TARGET_USER_ID in interaction.interaction:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            u_embeddings = target_user_embeddings[user]
            i_embeddings = target_item_embeddings[item]
        else:
            # 反向评估：interaction 中只有源域字段
            user = interaction[self.SOURCE_USER_ID]
            item = interaction[self.SOURCE_ITEM_ID]
            u_embeddings = source_user_embeddings[user]
            i_embeddings = source_item_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        """
        [联邦改造] 自动检测 interaction 字段，支持双向 full-sort 评估。
        """
        if self.TARGET_USER_ID in interaction.interaction:
            # 正向评估：目标域
            user = interaction[self.TARGET_USER_ID]
            restore_user_e, restore_item_e = self.get_restore_e()
            u_embeddings = restore_user_e[user]
            i_embeddings = restore_item_e[:self.target_num_items]
        else:
            # 反向评估：源域
            user = interaction[self.SOURCE_USER_ID]
            restore_user_e, restore_item_e = self.get_source_restore_e()
            u_embeddings = restore_user_e[user]
            i_embeddings = restore_item_e[:self.source_num_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None
        if self.source_restore_user_e is not None or self.source_restore_item_e is not None:
            self.source_restore_user_e, self.source_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, _, self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e

    def get_source_restore_e(self):
        """获取源域嵌入缓存（用于反向推荐评估）"""
        if self.source_restore_user_e is None or self.source_restore_item_e is None:
            _, self.source_restore_user_e, self.source_restore_item_e, _, _ = self.forward()
        return self.source_restore_user_e, self.source_restore_item_e