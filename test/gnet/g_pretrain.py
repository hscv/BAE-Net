import time
import sys
import yaml

import numpy as np
import torch as t
import torch.nn.functional as F

sys.path.insert(0, './gnet')
from g_init import set_optimizer_g

opts = yaml.safe_load(open('./tracking/options.yaml','r'))

def g_pretrain(model, model_g, criterion_g, pos_dataArr):
    # Evaluate mask
    n = pos_dataArr[0].size(0)
    if n % opts['batch_gnet'] == 0:
        nBatches = n / opts['batch_gnet']
    else:
        nBatches = n // opts['batch_gnet'] + 1

    print(nBatches)

    pos_dataArr = [pos_dataArr[i].view(n, 512, 3, 3) for i in range(5)]
    rowArr = []
    colArr = []
    for pos_data in pos_dataArr:
        prob = t.zeros(n)
        prob_k = t.zeros(9)
        for k in range(9):
            row = k // 3
            col = k % 3

            model.eval()
            for i in range(nBatches):
                batch = pos_data[opts['batch_gnet'] * i:min(n, opts['batch_gnet'] * (i + 1)), :, :, :]
                batch[:, :, col, row] = 0
                batch = batch.view(batch.size(0), -1)

                if opts['use_gpu']:
                    batch = batch.cuda()

                score = model(batch, in_layer='fc4', out_layer='fc6_softmax')[:, 1]
                prob[opts['batch_gnet'] * i:min(n, opts['batch_gnet'] * (i + 1))] = score
            model.train()

            prob_k[k] = prob.sum()

        _, idx = t.min(prob_k, 0)
        idx = idx.item()
        row = idx // 3
        col = idx % 3
        rowArr.append(row)
        colArr.append(col)

    batch_pos = opts['batch_pos']
    maxiter_g = opts['maxiter_g']
    pos_idx = np.random.permutation(pos_dataArr[0].size(0))
    while len(pos_idx) < batch_pos * maxiter_g:
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_dataArr[0].size(0))])

    pos_pointer = 0

    objective = t.zeros(maxiter_g)
    optimizer = set_optimizer_g(model_g)

    for iter in range(maxiter_g):
        for kk in range(5):
            row = rowArr[kk]
            col = colArr[kk]
            pos_data = pos_dataArr[kk]
            start = time.time()
            # select pos idx
            pos_next = pos_pointer + batch_pos
            pos_cur_idx = pos_idx[pos_pointer:pos_next]
            pos_cur_idx = pos_data.new(pos_cur_idx).long()
            if kk == 5:
                pos_pointer = pos_next

            labels = t.ones(batch_pos, 1, 3, 3)
            labels[:, :, col, row] = 0

            batch_pos_data = pos_data.index_select(0, pos_cur_idx)
            batch_pos_data = batch_pos_data.view(batch_pos_data.size(0), -1) # [32, 4608]
            # print ('batch_pos_data.size() = ',batch_pos_data.size())
            res = model_g(batch_pos_data)
            labels = labels.view(batch_pos, -1)
            # print ('res.size() = ',res.size()) # [32,9]
            # print ('labels.size() = ',labels.size()) # [32,9]
            loss_g = criterion_g(res.float(), labels.cuda().float())
            if kk == 0:
                loss_g_sum = loss_g
            else:
                loss_g_sum += loss_g
        model_g.zero_grad()
        loss_g_sum.backward()
        optimizer.step()

        objective[iter] = loss_g_sum
        end = time.time()
        print('asdn objective %.3f, %.2f s' % (objective.mean(), end - start))
