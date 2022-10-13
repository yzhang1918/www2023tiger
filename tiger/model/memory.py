import copy
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
from torch import nn, Tensor

from .time_encoding import TimeEncode
from .feature_getter import NumericalFeature
from .utils import select_latest_nids


class Memory(nn.Module):
    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.dim = dim
        self.register_buffer('vals', torch.zeros(n, dim), persistent=True)
        self.register_buffer('update_ts', torch.zeros(n), persistent=True)
        self.register_buffer('active_mask', torch.zeros(n).bool(), persistent=True)
    
    def clone(self):
        copy_mem = Memory(self.n, self.dim)
        copy_mem.vals.data = self.vals.data.clone()
        copy_mem.update_ts.data = self.update_ts.data.clone()
        return copy_mem
    
    @property
    def device(self):
        return self.vals.device
    
    def clear(self):
        self.vals[...] = 0.
        self.update_ts[...] = 0.
        self.active_mask[...] = False
    
    def get(self, ids: Tensor) -> Tuple[Tensor, Tensor]:
        v = self.vals[ids]
        ts = self.update_ts[ids]
        return v, ts
    
    def set(self, ids: Tensor, vals: Tensor, ts: Tensor, skip_check=False):
        if not skip_check:
            # sanity check
            last_ts = self.update_ts[ids]
            if (last_ts > ts).any().item():
                raise ValueError('You are not allowed to modify past memory.')
            if len(ids) != len(ids.unique()):
                raise ValueError('Duplicate node ids are not allowed.')
        
        self.update_ts.data[ids] = ts
        self.vals.data[ids] = vals
        self.active_mask[ids] = True
    

class MessageStoreNoGradLastOnly(nn.Module):
    node_messages: Tuple[Tensor, Tensor]

    def __init__(self, n, dim):
        super().__init__()
        self.n = n
        self.dim = dim
        self.register_buffer('node_msg_vals',
                             torch.zeros((n, dim)).float(),
                             persistent=False)
        self.register_buffer('node_msg_ts',
                             torch.zeros(n).float(),
                             persistent=False)
        self.nodes_with_messages = set()
    
    @property
    def node_messages(self):
        return (self.node_msg_vals, self.node_msg_ts)

    def clone(self):
        return copy.deepcopy(self)

    def store_events(self, src_ids: Tensor, dst_ids: Tensor,
                     src_prev_ts: Tensor, dst_prev_ts: Tensor,
                     src_vals: Tensor, dst_vals: Tensor,
                     eids: Tensor, ts: Tensor, 
                     emb_getter: NumericalFeature, time_encoder: TimeEncode):
        # sanity check
        # for these nodes, their messages should be used and thus cleared
        # TODO: To remove this check
        for n in torch.concat([src_ids, dst_ids]).cpu().numpy():
            if n in self.nodes_with_messages:
                raise ValueError(f'Node #{n} has unused messages.')
        
        src_vals += emb_getter.get_node_embeddings(src_ids)  # add static features
        dst_vals += emb_getter.get_node_embeddings(dst_ids)
        edge_vals = emb_getter.get_edge_embeddings(eids)  # [n, d]

        src_full_vals = torch.cat(
            [src_vals, dst_vals, edge_vals, time_encoder(ts - src_prev_ts)], 1)  # [n, 4d]
        dst_full_vals = torch.cat(
            [dst_vals, src_vals, edge_vals, time_encoder(ts - dst_prev_ts)], 1)  # [n, 4d]

        full_nids, index = select_latest_nids(torch.cat([src_ids, dst_ids]), ts.repeat(2))

        full_vals = torch.cat([src_full_vals, dst_full_vals], dim=0)[index]
        full_ts = ts.repeat(2)[index]
        
        self.nodes_with_messages.update(full_nids.cpu().numpy())

        self.node_msg_vals[full_nids] = full_vals
        self.node_msg_ts[full_nids] = full_ts

    def get_outdated_node_ids(self, node_ids: Union[Tensor, np.ndarray, None]
                             ) -> Tensor:
        """
        Return ids of nodes that store unused messages.
        -----
        node_ids: only return a subset of this tensor
        -----
        Return: outdated_node_ids
        outdated_node_ids: a LongTensor (on CPU)
        """
        if node_ids is None:
            outdated_node_ids = self.nodes_with_messages
        else:
            if isinstance(node_ids, np.ndarray):
                outdated_node_ids = self.nodes_with_messages & set(node_ids)
            else:
                outdated_node_ids = self.nodes_with_messages & set(node_ids.cpu().numpy())
        outdated_node_ids = torch.tensor(list(outdated_node_ids))
        return outdated_node_ids
    
    def clear(self, nids: Optional[Tensor]=None):
        """
        Clear pending messages.
        """
        if nids is None:
            nids = torch.tensor(list(self.nodes_with_messages),
                                device=self.node_msg_vals.device).long()
        
        self.nodes_with_messages.difference_update(nids.cpu().numpy())
        self.node_msg_vals[nids].fill_(0)
        self.node_msg_ts[nids].fill_(0)


class MessageStore:
    node_messages: Dict[int, List[Tuple[Tensor, int, float, bool]]]

    def __init__(self, n):
        self.n = n
        self.node_messages = {i: [] for i in range(n)}
        self.nodes_with_messages = set()
    
    def clone(self):
        return copy.deepcopy(self)
    
    def store_events(self, src_ids: Tensor, dst_ids: Tensor,
                     src_prev_ts: Tensor, dst_prev_ts: Tensor,
                     src_vals: Tensor, dst_vals: Tensor,
                     eids: Tensor, ts: Tensor, 
                     emb_getter: NumericalFeature, time_encoder: TimeEncode):
        src_ids, dst_ids, ts, eids = (x.cpu().numpy() for x in (src_ids, dst_ids, ts, eids))
        # sanity check
        # for these nodes, their messages should be used and thus cleared
        # TODO: To remove this check
        for n in np.concatenate([src_ids, dst_ids]):
            if len(self.node_messages[n]):
                raise ValueError(f'Node #{n} has {len(self.node_messages[n])} unused messages.')
        
        vals = torch.stack([src_vals, dst_vals], 1)  # [n, 2, d]
        for src, dst, val, eid, t in zip(src_ids, dst_ids, vals, eids, ts):
            self.node_messages[src].append((val.flatten(), eid, t, True))
            self.node_messages[dst].append((val.roll(shifts=1, dims=0).flatten(), eid, t, False))
            self.nodes_with_messages.add(src)
            self.nodes_with_messages.add(dst)
    
    def get_outdated_node_ids(self, node_ids: Union[Tensor, np.ndarray, None]
                             ) -> Tensor:
        """
        Return ids of nodes that store unused messages.
        -----
        node_ids: only return a subset of this tensor
        -----
        Return: node_ids
        outdated_node_ids: a LongTensor (on CPU)
        """
        if node_ids is None:
            outdated_node_ids = self.nodes_with_messages
        else:
            if isinstance(node_ids, np.ndarray):
                outdated_node_ids = self.nodes_with_messages & set(node_ids)
            else:
                outdated_node_ids = self.nodes_with_messages & set(node_ids.cpu().numpy())
        outdated_node_ids = torch.tensor(list(outdated_node_ids))
        return outdated_node_ids
    
    def clear(self, nids: Optional[Tensor]=None):
        """
        Clear stashed messages.
        """
        if nids is not None:
            for n in nids:
                n = n.item()
                self.node_messages[n].clear()
                self.nodes_with_messages.remove(n)
        else:
            for n in self.nodes_with_messages:
                self.node_messages[n].clear()
            self.nodes_with_messages.clear()


class MessageStoreNoGrad(MessageStore):
    node_messages: Dict[int, List[Tuple[Tensor, float]]]

    def store_events(self, src_ids: Tensor, dst_ids: Tensor,
                     src_prev_ts: Tensor, dst_prev_ts: Tensor,
                     src_vals: Tensor, dst_vals: Tensor,
                     eids: Tensor, ts: Tensor, 
                     emb_getter: NumericalFeature, time_encoder: TimeEncode):
        # sanity check
        # for these nodes, their messages should be used and thus cleared
        # TODO: To remove this check
        for n in torch.concat([src_ids, dst_ids]).cpu().numpy():
            if len(self.node_messages[n]):
                raise ValueError(f'Node #{n} has {len(self.node_messages[n])} unused messages.')
        
        src_vals += emb_getter.get_node_embeddings(src_ids)  # add static features
        dst_vals += emb_getter.get_node_embeddings(dst_ids)

        edge_vals = emb_getter.get_edge_embeddings(eids)  # [n, d]

        src_full_vals = torch.cat([src_vals, dst_vals, edge_vals, time_encoder(ts - src_prev_ts)], 1)  # [n, 4d]
        dst_full_vals = torch.cat([dst_vals, src_vals, edge_vals, time_encoder(ts - dst_prev_ts)], 1)  # [n, 4d]

        src_ids = src_ids.cpu().numpy()
        dst_ids = dst_ids.cpu().numpy()
        ts = ts.cpu().numpy()
        for src, dst, sval, dval, t in zip(src_ids, dst_ids, src_full_vals, dst_full_vals, ts):
            self.node_messages[src].append((sval, t))
            self.node_messages[dst].append((dval, t))
            self.nodes_with_messages.add(src)
            self.nodes_with_messages.add(dst)

