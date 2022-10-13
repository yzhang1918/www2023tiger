import argparse
import json
import os
import math
import pathlib
import pickle
import shutil
import time
import traceback

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from CHANGELOG import MODEL_VERSION
from tiger.data.data_loader import ChunkSampler, GraphCollator, load_jodie_data
from tiger.data.graph import Graph
from tiger.eval_utils import eval_edge_prediction, warmup
from tiger.model.feature_getter import NumericalFeature
from tiger.model.tiger import TIGER
from tiger.utils import BackgroundThreadGenerator
from tiger.model.restarters import SeqRestarter, StaticRestarter, WalkRestarter
from tiger.eval_utils import eval_edge_prediction, warmup

from init_utils import init_data, init_model, init_parser
from train_utils import EarlyStopMonitor, hash_args, seed_all, get_logger, dist_setup, dist_cleanup, DummyLogger


def worker(rank, world_size, args):
    dist_setup(rank, world_size)
 
    run(rank, world_size, prefix=args.prefix, root=args.root, data=args.data,
        dim=args.dim, feature_as_buffer=not args.no_feat_buffer,
        gpu=args.gpu, seed=args.seed, num_workers=args.num_workers, subset=args.subset,
        hit_type=args.hit_type, restarter_type=args.restarter_type,
        hist_len=args.hist_len,
        n_neighbors=args.n_neighbors,
        n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout,
        strategy=args.strategy, msg_src=args.msg_src, upd_src=args.upd_src,
        mem_update_type=args.upd_fn, msg_tsfm_type=args.tsfm_fn,
        lr=args.lr, n_epochs=args.n_epochs, bs=args.bs, 
        mutual_coef=args.mutual_coef, patience=args.patience,
        restart_prob=args.restart_prob,
        recover_from=args.recover_from, recover_step=args.recover_step,
        force=args.force, warmup_steps=args.warmup
        )
   
    dist_cleanup()


def run(rank, world_size, *, prefix,
        root, data, dim, feature_as_buffer,
        gpu, seed, num_workers, subset,
        hit_type, restarter_type, hist_len,
        n_neighbors, n_layers, n_heads, dropout,
        strategy, msg_src, upd_src,
        mem_update_type, msg_tsfm_type,
        lr, n_epochs, bs, mutual_coef, patience,
        restart_prob, 
        recover_from, recover_step, force, warmup_steps,
        ):
    # Get hash
    args = {k: v for k, v in locals().items() 
            if not k in {'gpu', 'force', 'rank', 'recover_from', 'recover_step'}}
    HASH = hash_args(**args, MODEL_VERSION=MODEL_VERSION)
    prefix = HASH if prefix == '' else f'{prefix}.{HASH}'
    device = torch.device(f'cuda:{rank}')
    
    restart_mode = restart_prob > 0

    # Sanity check
    if (not restart_mode) and (warmup_steps > 0):
        raise ValueError('Warmup is not needed without restart.')

    # Path
    MODEL_SAVE_PATH = f'./saved_models/{prefix}.pth'
    RESULT_SAVE_PATH = f"results/{prefix}.json"
    PICKLE_SAVE_PATH = "results/{}.pkl".format(prefix)

    if rank == 0:  # only the first process logs and saves
        pathlib.Path("./saved_models/").mkdir(parents=True, exist_ok=True)
        ckpts_dir = pathlib.Path(f"./saved_checkpoints/{prefix}")
        ckpts_dir.mkdir(parents=True, exist_ok=True)
        get_checkpoint_path = lambda epoch: ckpts_dir / f'{epoch}.pth'
        pathlib.Path("results/").mkdir(parents=True, exist_ok=True)

    # init logger
    logger = get_logger(HASH) if rank == 0 else DummyLogger()
    logger.info(f'[START {HASH}]')
    logger.info(f'Model version: {MODEL_VERSION}')
    logger.info(", ".join([f"{k}={v}" for k, v in args.items()]))

    if pathlib.Path(RESULT_SAVE_PATH).exists() and not force:
        logger.info('Duplicate task! Abort!')
        return False

    if world_size > 1:
        dist.barrier()  # for single process should also work

    try:
        # Init
        seed_all(seed)
        # ============= Load Data ===========
        basic_data, graphs, dls = init_data(
            data, root, seed,
            rank=rank, world_size=world_size,
            num_workers=num_workers, bs=bs, warmup_steps=warmup_steps,
            subset=subset, strategy=strategy,
            n_layers=n_layers, n_neighbors=n_neighbors,
            restarter_type=restarter_type, hist_len=hist_len
        )
        (
            nfeats, efeats, full_data, train_data, val_data, test_data, 
            inductive_val_data, inductive_test_data
        ) = basic_data
        train_graph, full_graph = graphs
        (
            train_dl, offline_dl, val_dl, ind_val_dl, 
            test_dl, ind_test_dl, val_warmup_dl, test_warmup_dl
        ) = dls

        # ============= Init Model ===========
        model = init_model(
            nfeats, efeats, train_graph, full_graph, full_data, device,
            feature_as_buffer=feature_as_buffer, dim=dim,
            n_layers=n_layers, n_heads=n_heads, n_neighbors=n_neighbors,
            hit_type=hit_type, dropout=dropout,
            restarter_type=restarter_type, hist_len=hist_len,
            msg_src=msg_src, upd_src=upd_src,
            msg_tsfm_type=msg_tsfm_type, mem_update_type=mem_update_type
        )

        # recover training
        if rank == 0 and recover_from != '':
            model.load_state_dict(torch.load(recover_from, map_location=device))
            epoch_start = recover_step
        else:
            epoch_start = 0

        ddp_model = DDP(model, device_ids=[device], broadcast_buffers=False, find_unused_parameters=True)
        optimizer = optim.Adam(ddp_model.parameters(), lr=lr * np.sqrt(world_size))  # NB: larger lr

        val_aps = []
        ind_val_aps = []
        val_aucs = []
        ind_val_aucs = []
        epoch_times = []
        total_epoch_times = []
        train_losses = []

        if rank == 0:
            early_stopper = EarlyStopMonitor(max_round=patience, epoch_start=epoch_start)
        signal = torch.tensor([1]).to(device)  # as a flag for multiprocessing
        done = False
        for epoch in range(epoch_start, n_epochs):
            dist.all_reduce(signal, op=dist.ReduceOp.MIN)
            if signal.item() == 0:  # fininsh training
                break
            # Training
            start_epoch_t0 = time.time()
            logger.info('Start {} epoch'.format(epoch))

            m_loss = []
            m_contrast_loss = []
            m_mutual_loss = []
            it = BackgroundThreadGenerator(train_dl)
            if rank == 0:
                it = tqdm.tqdm(it, total=len(train_dl), ncols=50)

            ddp_model.train()
            model.reset()
            model.graph = train_graph
            uptodate_nodes = set()
            for i_batch, (src_ids, dst_ids, neg_dst_ids, ts, eids, _, comp_graph) in enumerate(it):
                src_ids = src_ids.long().to(device)
                dst_ids = dst_ids.long().to(device)
                neg_dst_ids = neg_dst_ids.long().to(device)
                ts = ts.float().to(device)
                eids = eids.long().to(device)
                comp_graph.to(device)
                optimizer.zero_grad()

                # Restart
                if np.random.rand() < restart_prob and i_batch:
                    uptodate_nodes = set()
                    model.msg_store.clear()
                
                # different from single process mode
                # in DDP, we are also in the restarting mode
                involved_nodes = comp_graph.np_computation_graph_nodes
                restart_nodes = set(involved_nodes) - set(uptodate_nodes)
                r_nids = torch.tensor(list(restart_nodes)).long().to(device)
                model.restart(r_nids, torch.full((len(r_nids),), ts.min().item()).to(device))
                uptodate_nodes.update(restart_nodes)

                contrast_loss, mutual_loss = ddp_model(
                    src_ids, dst_ids, neg_dst_ids, ts, eids, comp_graph,
                    contrast_only=(restart_prob == 0)
                )
                loss = contrast_loss + mutual_coef * mutual_loss
                loss.backward()
                optimizer.step()

                dist.all_reduce(loss)
                dist.all_reduce(contrast_loss)
                dist.all_reduce(mutual_loss)
                m_contrast_loss.append(contrast_loss.item() / world_size)
                m_mutual_loss.append(mutual_loss.item() / world_size)
                m_loss.append(loss.item() / world_size)

            epoch_time = time.time() - start_epoch_t0
            epoch_times.append(epoch_time)

            if rank == 0:
                # EVAL
                model.eval()
                model.flush_msg()
                model.graph = full_graph

                model.msg_store.clear()
                if warmup_steps:
                    uptodate_nodes = warmup(model, val_warmup_dl, device)
                else:
                    uptodate_nodes = set()

                memory_state_train_end = model.save_memory_state()  # save states at t_train_end
                val_ap, val_auc = eval_edge_prediction(
                    model, val_dl, device, restart_mode, uptodate_nodes=uptodate_nodes.copy()
                )  # memory modified
                memory_state_valid_end = model.save_memory_state()  # save states at t_valid_end
                model.load_memory_state(memory_state_train_end)  # load states at t_train_end
                ind_val_ap, ind_val_auc = eval_edge_prediction(
                    model, ind_val_dl, device, restart_mode, uptodate_nodes=uptodate_nodes.copy()
                )
                model.load_memory_state(memory_state_valid_end)

                total_epoch_time = time.time() - start_epoch_t0
                total_epoch_times.append(total_epoch_time)

                # save
                model.flush_msg()
                torch.save(model.state_dict(), get_checkpoint_path(epoch))

                logger.info('Epoch {:4d} total    took  {:.2f}s'.format(epoch, total_epoch_time))
                logger.info('Epoch {:4d} training took  {:.2f}s'.format(epoch, epoch_time))
                logger.info(f'Epoch mean    total loss: {np.mean(m_loss):.4f}')
                logger.info(f'Epoch mean contrast loss: {np.mean(m_contrast_loss):.4f}')
                logger.info(f'Epoch mean   mutual loss: {np.mean(m_mutual_loss):.4f}')
                logger.info(f'Val     ap: {val_ap:.4f}, Val     auc: {val_auc:.4f}')
                logger.info(f'Val ind ap: {ind_val_ap:.4f}, Val ind auc: {ind_val_auc:.4f}')

                val_aps.append(val_ap)
                ind_val_aps.append(ind_val_ap)
                val_aucs.append(val_auc)
                ind_val_aucs.append(ind_val_auc)
                train_losses.append(np.mean(m_loss))

                if early_stopper.early_stop_check(val_ap):
                    logger.info('No improvement over {} epochs, stop training'.format(
                        early_stopper.max_round))
                    done = True

            dist.barrier()  # wait
            if done:
                if rank == 0:  # send signal to all processes
                    dist.all_reduce(torch.tensor([0]).to(device), op=dist.ReduceOp.MIN)
                break

        if rank > 0:
            # only main process evals test data
            # other processes can safely exit now
            return True

        if rank == 0:
            # Testing
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_epoch = early_stopper.best_epoch - epoch_start
            best_val_ap = val_aps[best_epoch]
            best_val_auc = val_aucs[best_epoch]
            best_ind_val_ap = ind_val_aps[best_epoch]
            best_ind_val_auc = ind_val_aucs[best_epoch]
            logger.info(f'[ Val] Best     ap: {    best_val_ap:.4f} Best     auc: {    best_val_auc:.4f}')
            logger.info(f'[ Val] Best ind ap: {best_ind_val_ap:.4f} Best ind auc: {best_ind_val_auc:.4f}')

            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            model_state = torch.load(best_model_path)
            model.load_state_dict(model_state)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)  # save to the model save folder

            model.eval()
            model.graph = full_graph
            if restart_mode:
                model.msg_store.clear()
                if warmup_steps:
                    uptodate_nodes = warmup(model, test_warmup_dl, device)
                else:
                    uptodate_nodes = set()

            memory_state_val_end = model.save_memory_state()  # save states at t_valid_end
            test_ap, test_auc = eval_edge_prediction(
                model, test_dl, device, restart_mode,
                uptodate_nodes=uptodate_nodes.copy()
            )  # memory modified
            model.load_memory_state(memory_state_val_end)  # load states at t_valid_end
            ind_test_ap, ind_test_auc = eval_edge_prediction(
                model, ind_test_dl, device, restart_mode,
                uptodate_nodes=uptodate_nodes.copy()
            )
            logger.info(f'[Test]     ap: {    test_ap:.4f}     auc: {    test_auc:.4f}')
            logger.info(f'[Test] ind ap: {ind_test_ap:.4f} ind auc: {ind_test_auc:.4f}')

            # Save results for this run
            pickle.dump({
            "val_aps": val_aps,
            "val_aucs": val_aucs,
            "ind_val_aps": ind_val_aps,
            "ind_val_aucs": ind_val_aucs,
            "test_ap": test_ap,
            "ind_test_ap": ind_test_ap,
            "test_auc": test_auc,
            "ind_test_auc": ind_test_auc,
            "epoch_times": epoch_times,
            "train_losses": train_losses,
            "total_epoch_times": total_epoch_times
            }, open(PICKLE_SAVE_PATH, "wb"))

            results = args.copy()
            results.update(
                HASH=HASH,
                VERSION=MODEL_VERSION,
                val_ap=best_val_ap, ind_val_ap=best_ind_val_ap, 
                val_auc=best_val_auc, ind_val_auc=best_ind_val_auc,
                test_ap=test_ap, test_auc=test_auc,
                ind_test_ap=ind_test_ap, ind_test_auc=ind_test_auc
            )
            json.dump(results, open(RESULT_SAVE_PATH, 'w'))

            # remove all ckpts
            shutil.rmtree(ckpts_dir)
            return True
    
    except Exception as e:
        if rank == 0:
            logger.error(traceback.format_exc())
            logger.error(e)
        raise


def get_args():
    parser = init_parser()
    # Exp Setting
    parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # DDP
    parser.add_argument('--gpu', type=str, default='0', help='Cuda index')
    parser.add_argument('--port', type=str, default='29500', help='port for DDP')
    # Data
    parser.add_argument('--subset', type=float, default=1.0, help='Only use a subset of training data')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help='Number of workers in train dataloader')
    # Training
    parser.add_argument('--n_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--bs', type=int, default=200, help='Batch size')
    # MISC
    parser.add_argument('--force', action='store_true', help='Overwirte the existing task')
    parser.add_argument('--recover_from', type=str, default='', help='ckpt path')
    parser.add_argument('--recover_step', type=int, default=0, help='recover step')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_args()
    gpus = args.gpu.split(',')

    # Sanity Check
    if len(gpus) <= 1 or '-1' in gpus:
        raise ValueError('Please use the single-gpu train file instead!')
    if args.restart_prob <= 0.:
        raise ValueError('Restart prob should be > 0 in the DDP mode.')

    WORLD_SIZE = len(gpus)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port
    mp.spawn(worker, nprocs=WORLD_SIZE, args=(WORLD_SIZE, args), join=True, daemon=True)
