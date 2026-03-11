"""
联邦学习 DGCDR 训练入口脚本

客户端A: 源域=DoubanMovie → 目标域=DoubanBook
客户端B: 源域=DoubanBook  → 目标域=DoubanMovie

使用方式:
    python run_federated_cdr.py
    python run_federated_cdr.py --client_a_config xxx.yaml --client_b_config yyy.yaml
"""
import argparse
import logging

from recbole.utils import init_logger, init_seed, set_color
from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset, data_preparation
from recbole_cdr.utils import get_model, get_trainer
from recbole_cdr.federated import FederatedCentralServer, FederatedClient
from recbole_cdr.trainer.trainer import FederatedDGCDRTrainer


def _build_client(client_id, model_name, overall_config_file,
                  data_config_file, model_config_file, device, logger):
    """为单个客户端独立构建 config → dataset → dataloader → model → trainer。"""
    logger.info(f"\n[{client_id}] Loading config from: {data_config_file}")

    config = CDRConfig(
        model=model_name,
        config_file_list=[overall_config_file, data_config_file, model_config_file],
    )
    config['device'] = device

    init_seed(config['seed'], config['reproducibility'])

    dataset = create_dataset(config)
    logger.info(f"[{client_id}] Dataset: {dataset}")

    train_data, valid_data, test_data = data_preparation(config, dataset)
    logger.info(
        f"[{client_id}] source_domain={config['source_domain']['dataset']}  "
        f"target_domain={config['target_domain']['dataset']}"
    )

    init_seed(config['seed'], config['reproducibility'])
    model = get_model(config['model'])(config, train_data.dataset).to(config['device'])
    logger.info(f"[{client_id}] Model initialized: {model.__class__.__name__}")

    TrainerClass = get_trainer(config['MODEL_TYPE'], config['model'])
    trainer = TrainerClass(config, model)

    client = FederatedClient(
        client_id=client_id,
        model=model,
        train_data=train_data,
        valid_data=valid_data,
        config=config,
    )
    logger.info(f"[{client_id}] Data size: {client.data_size}")

    return client, trainer, config, test_data


def run_federated_dgcdr(
    overall_config_file=None,
    client_a_data_config=None,
    client_b_data_config=None,
    model_config_file=None,
):
    """联邦 DGCDR 训练主函数。"""

    # ------------------------------------------------------------------
    # 用客户端A的配置初始化 RecBole 日志系统（写入文件）
    # ------------------------------------------------------------------
    _tmp_cfg = CDRConfig(
        model='DGCDR',
        config_file_list=[overall_config_file, client_a_data_config, model_config_file],
    )
    init_seed(_tmp_cfg['seed'], _tmp_cfg['reproducibility'])
    init_logger(_tmp_cfg)   # ← 初始化 RecBole 文件日志（只调用一次）

    # 获取 RecBole 根 logger，所有 INFO:root: 日志都写入同一文件
    root_logger = logging.getLogger()
    # 用同一套 handler 给 __main__ logger，避免双套输出
    main_logger = logging.getLogger(__name__)
    main_logger.handlers = root_logger.handlers
    main_logger.setLevel(root_logger.level)
    main_logger.propagate = False

    device = _tmp_cfg['device']

    main_logger.info(set_color('=' * 70, 'cyan'))
    main_logger.info(set_color('Federated DGCDR Training  (Bidirectional CDR)', 'cyan'))
    main_logger.info(set_color('=' * 70, 'cyan'))
    main_logger.info(f"  Client A data config : {client_a_data_config}")
    main_logger.info(f"  Client B data config : {client_b_data_config}")
    main_logger.info(f"  Device               : {device}")

    # ------------------------------------------------------------------
    # 构建客户端A
    # ------------------------------------------------------------------
    main_logger.info(set_color('\n[Setup] Building Client A...', 'green'))
    client_A, trainer_A, config_A, test_data_A = _build_client(
        client_id='client_A',
        model_name='DGCDR',
        overall_config_file=overall_config_file,
        data_config_file=client_a_data_config,
        model_config_file=model_config_file,
        device=device,
        logger=main_logger,
    )

    # ------------------------------------------------------------------
    # 构建客户端B
    # ------------------------------------------------------------------
    main_logger.info(set_color('\n[Setup] Building Client B...', 'green'))
    client_B, trainer_B, config_B, test_data_B = _build_client(
        client_id='client_B',
        model_name='DGCDR',
        overall_config_file=overall_config_file,
        data_config_file=client_b_data_config,
        model_config_file=model_config_file,
        device=device,
        logger=main_logger,
    )

    # ------------------------------------------------------------------
    # 联邦服务器
    # ------------------------------------------------------------------
    server = FederatedCentralServer(config_A)
    main_logger.info(set_color('\n[Setup] Federated infrastructure ready.', 'green'))
    main_logger.info(
        f"  num_rounds={(config_A['num_federated_rounds'] if 'num_federated_rounds' in config_A else 10)}  "
        f"local_epochs={(config_A['local_epochs'] if 'local_epochs' in config_A else 1)}  "
        f"fed_stopping_step={(config_A['fed_stopping_step'] if 'fed_stopping_step' in config_A else 5)}"
    )

    # ------------------------------------------------------------------
    # 联邦训练
    # ------------------------------------------------------------------
    main_logger.info(set_color('\n[Train] Starting federated training...', 'green'))
    fed_trainer = FederatedDGCDRTrainer(
        config=config_A,
        clients=[client_A, client_B],
        server=server,
        local_trainers=[trainer_A, trainer_B],
        # 传入客户端标识供训练器标注日志
        client_labels={
            'client_A': f"A [{config_A['source_domain']['dataset']} → {config_A['target_domain']['dataset']}]",
            'client_B': f"B [{config_B['source_domain']['dataset']} → {config_B['target_domain']['dataset']}]",
        }
    )
    results = fed_trainer.run(test_data_list=[test_data_A, test_data_B])

    # ------------------------------------------------------------------
    # 最终评估
    # ------------------------------------------------------------------
    main_logger.info(set_color('\n[Eval] Final evaluation...', 'green'))

    test_result_A = trainer_A.evaluate(
        test_data_A, load_best_model=False, show_progress=False
    )
    test_result_B = trainer_B.evaluate(
        test_data_B, load_best_model=False, show_progress=False
    )

    main_logger.info(set_color('\n' + '=' * 70, 'yellow'))
    main_logger.info(set_color('Federated Training Complete!', 'yellow'))
    main_logger.info(set_color(
        f'[Client A] {config_A["source_domain"]["dataset"]} → '
        f'{config_A["target_domain"]["dataset"]}:\n  {test_result_A}', 'yellow'))
    main_logger.info(set_color(
        f'[Client B] {config_B["source_domain"]["dataset"]} → '
        f'{config_B["target_domain"]["dataset"]}:\n  {test_result_B}', 'yellow'))
    main_logger.info(set_color(
        f'Total communication rounds: {server.communication_rounds}', 'yellow'))
    main_logger.info(set_color('=' * 70, 'yellow'))

    return {
        'test_result_A': test_result_A,
        'test_result_B': test_result_B,
        'round_stats':   results['round_stats'],
        'server_stats':  results['server_stats'],
    }


# ======================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Federated DGCDR Training')
    parser.add_argument('--overall_config', type=str,
                        default='recbole_cdr/properties/overall.yaml')
    parser.add_argument('--client_a_config', type=str,
                        default='recbole_cdr/properties/dataset/'
                                'AmazonSport_AmazonCloth_commonUser_5-core.yaml')
    parser.add_argument('--client_b_config', type=str,
                        default='recbole_cdr/properties/dataset/'
                                'AmazonCloth_AmazonSport_commonUser_5-core.yaml')
    parser.add_argument('--model_config', type=str,
                        default='recbole_cdr/properties/model/DGCDR.yaml')
    args = parser.parse_args()

    run_federated_dgcdr(
        overall_config_file=args.overall_config,
        client_a_data_config=args.client_a_config,
        client_b_data_config=args.client_b_config,
        model_config_file=args.model_config,
    )