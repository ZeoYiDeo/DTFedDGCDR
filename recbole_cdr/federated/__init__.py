"""
recbole_cdr.federated
####################################
联邦学习模块，包含中央服务器和客户端实现。
"""
from recbole_cdr.federated.server import FederatedCentralServer
from recbole_cdr.federated.client import FederatedClient

__all__ = ['FederatedCentralServer', 'FederatedClient']