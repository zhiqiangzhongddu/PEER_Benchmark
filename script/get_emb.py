#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from torchdrug.utils import comm, pretty
from torchdrug import data, core, utils
from torch.utils import data as torch_data

from IPython import get_ipython
def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    sys.path.append('/home/zhiqiang/PEER_Benchmark')
else:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from peer import protbert, util, flip
from script.run_single import *


# In[1]:





# In[2]:


# train the model, same as PEER code
args = parse_args()

args.config = '/home/zhiqiang/PEER_Benchmark/config/single_task/ESM/gb1_ESM_fix.yaml' if is_notebook() else os.path.realpath(args.config)
cfg = util.load_config(args.config)

cfg.dataset["split"] = args.split if is_notebook() else "two_vs_rest"


# In[2]:





# In[ ]:


set_seed(args.seed)
output_dir = util.create_working_directory(cfg)
logger = util.get_root_logger()
if comm.get_rank() == 0:
    logger.warning("Config file: %s" % args.config)
    logger.warning(pprint.pformat(cfg))
    logger.warning("Output dir: %s" % output_dir)
    shutil.copyfile(args.config, os.path.basename(args.config))
os.chdir(output_dir)

solver = build_solver(cfg, logger)
solver, best_epoch = train_and_validate(cfg, solver)
if comm.get_rank() == 0:
    logger.warning("Best epoch on valid: %d" % best_epoch)
test(cfg, solver)


# In[7]:





# In[4]:


# code: https://torchdrug.ai/docs/_modules/torchdrug/core/engine.html#Engine.evaluate
def get_embedding(solver, split):
    # split = "train"

    if comm.get_rank() == 0:
        logger.warning(pretty.separator)
        logger.warning("Extract on %s" % split)
    test_set = getattr(solver, "%s_set" % split)
    sampler = torch_data.DistributedSampler(test_set, solver.world_size, solver.rank)
    dataloader = data.DataLoader(test_set, solver.batch_size, sampler=sampler, num_workers=solver.num_worker)
    model = solver.model.model

    model.eval()
    preds = []
    targets = []
    graph_features = []
    for batch in dataloader:
        if solver.device.type == "cuda":
            batch = utils.cuda(batch, device=solver.device)

        all_loss = torch.tensor(0, dtype=torch.float32, device=solver.device)
        metric = {}

        graph = batch["graph"]
        if solver.model.graph_construction_model:
            graph = solver.model.graph_construction_model(graph)
        output = solver.model.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        graph_feature = output["graph_feature"].cpu()

        pred = solver.model.mlp(output["graph_feature"])
        if solver.model.normalization:
            pred = pred * solver.model.std + solver.model.mean
        target = solver.model.target(batch)
        preds.append(pred)
        targets.append(target)
        if solver.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)

        graph_features.append(graph_feature)

    pred = utils.cat(preds)
    target = utils.cat(targets)
    graph_features = utils.cat(graph_features)

    metric = solver.model.evaluate(pred, target)

    return graph_features, metric


# In[4]:





# In[5]:


graph_train, metric_train = get_embedding(solver, 'train')
graph_valid, metric_valid = get_embedding(solver, 'valid')
graph_test, metric_test = get_embedding(solver, 'test')

graphs = utils.cat([graph_train, graph_valid, graph_test])


# In[5]:





# In[7]:


if cfg.get("fix_encoder"):
    save_file = "/home/zhiqiang/PEER_Benchmark/script/extracted_embeddings/{}_{}.pt".format(
        cfg.dataset["class"],
        cfg.task["model"]["model"]
    )
else:
    save_file = "/home/zhiqiang/PEER_Benchmark/script/extracted_embeddings/" \
                "{}_{}_{}_Spearman_train_{:.3f}_valid_{:.3f}_test_{:.3f}.pt".format(
        cfg.dataset["class"],
        cfg.task["model"]["model"],
        cfg.dataset["split"],
        metric_train["spearmanr [target]"].cpu().numpy().tolist(),
        metric_valid["spearmanr [target]"].cpu().numpy().tolist(),
        metric_test["spearmanr [target]"].cpu().numpy().tolist()
    )
print(save_file)
torch.save(graphs, save_file)


# In[7]:


# python script/run_single.py -c config/single_task/$model/$yaml_config --seed 0

