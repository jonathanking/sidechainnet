""" A script to hold some utility functions for model logging.

    Note that MSE is always recorded as MSE, and reported as RMSE.
"""
import sys
import time
import os

import numpy as np
import torch
import wandb

from protein_transformer.protein.Sequence import VOCAB
from .dataset import  VALID_SPLITS, paired_collate_fn
from .protein.PDB_Creator import PDB_Creator
from .losses import angles_to_coords, inverse_trig_transform

def print_train_batch_status(args, items):
    """
    Print the status line during training after a single batch update. Uses tqdm progress bar
    by default, unless the script is run in a high performance computing (cluster) env.
    """
    # Extract relevant metrics
    pbar, metrics, src_seq = items
    cur_lr = metrics["history-lr"][-1]
    train_drmsd_loss = metrics["train"]["batch-drmsd-full"]
    train_mse_loss = metrics["train"]["batch-mse-full"]
    train_comb_loss = metrics["train"]["batch-combined-full"]
    if args.loss  == "combined":
        loss = train_comb_loss
    else:
        loss = metrics["train"]["batch-lndrmsd-full"]
    lr_string = f", LR = {cur_lr:.7f}" if args.lr_scheduling == "noam" else ""
    speed_avg = np.mean(metrics["train"]["speeds"])


    pbar.set_description('\r  - (Train) drmsd={drmsd:.2f}, lndrmsd={lnd:0.7f}, rmse={rmse:.4f},'
                         ' c={comb:.2f}{lr}, res/s={speed:.0f}'.format(
        drmsd=float(train_drmsd_loss),
        lr=lr_string,
        rmse=np.sqrt(float(train_mse_loss)),
        comb=float(train_comb_loss),
        lnd=metrics["train"]["batch-lndrmsd-full"],
        speed=speed_avg))


def print_eval_batch_status(args, items):
    """
    Print the status line during evaluation after a single batch update.
    Will only be seen if using a progress bar. Otherwise, there is no information logged.
    """
    pbar, d_loss, mode, m_loss, c_loss = items
    pbar.set_description('\r  - (Eval-{1}) drmsd = {0:.6f}, rmse = {2:.6f}, comb = {3:.6f}'.format(
        float(d_loss),
        mode,
        np.sqrt(float(m_loss)),
        float(c_loss)))


def print_end_of_epoch_status(mode, items):
    """
    Prints the training status at the end of an epoch and updates wandb summary stats.
    """
    missing_str = "      "
    start, metrics = items
    cur_lr = metrics["history-lr"][-1]
    drmsd_loss_str = "{:6.3f}".format(metrics[mode]["epoch-drmsd-full"]) if metrics[mode]["epoch-drmsd-full"] != 0 else missing_str
    mse_loss = metrics[mode]["epoch-mse-full"]
    rmsd_loss_str = "{:6.3f}".format(metrics[mode]["epoch-rmsd-full"]) if metrics[mode]["epoch-rmsd-full"] != 0 else missing_str
    comb_loss_str = "{:6.3f}".format(metrics[mode]["epoch-combined-full"]) if metrics[mode]["epoch-combined-full"] != 0 else missing_str
    avg_speed = np.mean(metrics[mode]["speed-history"])
    print('\r  - ({mode})   drmsd: {d}, rmse: {m: 6.3f}, rmsd: {rmsd}, comb: {comb}, '
          'elapse: {elapse:3.3f} min, lr: {lr: {lr_precision}}, res/sec = {speed:.0f}'.format(
        mode=mode.capitalize(),
        d=drmsd_loss_str,
        m=np.sqrt(mse_loss),
        elapse=(time.time() - start) / 60,
        lr=cur_lr,
        rmsd=rmsd_loss_str,
        comb=comb_loss_str,
        lr_precision="5.2e" if (cur_lr < .001 and cur_lr != 0) else "5.3f",
        speed=avg_speed))

    # Log end of epoch stats with wandb
    wandb.run.summary[f"final_epoch_{mode}_drmsd"] = metrics[mode]["epoch-drmsd-full"]
    wandb.run.summary[f"final_epoch_{mode}_mse"] = metrics[mode]["epoch-mse-full"]
    wandb.run.summary[f"final_epoch_{mode}_rmsd"] = metrics[mode]["epoch-rmsd-full"]
    wandb.run.summary[f"final_epoch_{mode}_comb"] = metrics[mode]["epoch-combined-full"]
    wandb.run.summary[f"final_epoch_{mode}_speed"] = avg_speed


def update_loss_trackers(args, epoch_i, metrics):
    """
    Updates the current loss to compare according to an early stopping policy.
    """

    loss_to_compare = metrics[args.es_mode][f"epoch-{args.es_metric}-full"]
    losses_to_compare = metrics[args.es_mode][f"epoch-history-{args.es_metric}"]

    if metrics["best_valid_loss_so_far"] - loss_to_compare > args.early_stopping_threshold:
        metrics["best_valid_loss_so_far"] = loss_to_compare
        metrics["epoch_last_improved"] = epoch_i
    elif args.early_stopping and epoch_i - metrics["epoch_last_improved"] > args.early_stopping:
        # Model hasn't improved in X epochs
        print("No improvement for {} epochs. Stopping model training early.".format(args.early_stopping))
        wandb.run.summary["stopped_training_early"] = True
        raise EarlyStoppingCondition

    metrics["loss_to_compare"] = loss_to_compare
    metrics["losses_to_compare"] = losses_to_compare

    return metrics


def log_batch(log_writer, metrics, start_time,  mode="valid", end_of_epoch=False, t=None):
    """
    Logs training info to an already instantiated CSV-writer log.
    """
    if not t:
        t = time.time()
    m = metrics[mode]
    if end_of_epoch:
        be = "epoch"
    else:
        be = "batch"
    if "speed" not in m.keys():
        m["speed"] =0
    log_writer.writerow([m[f"{be}-drmsd-full"], m[f"{be}-lndrmsd-full"], np.sqrt(m[f"{be}-mse-full"]),
                         m[f"{be}-rmsd-full"], m[f"{be}-combined-full"], metrics["history-lr"][-1],
                         mode, "epoch", round(t - start_time, 4), m["speed"]])


def do_train_batch_logging(metrics, losses, src_seq, optimizer, args, log_writer, pbar, start_time, pred_angs,
                           tgt_coords, step, validation_datasets, model, device):
    """
    Performs all necessary logging at the end of a batch in the training epoch.
    Updates custom metrics dictionary and wandb logs. Prints status of training.
    Also checks for NaN losses.
    # TODO log structure using a subprocess for speed

        1. Updates metrics.
        2. Logs training batch performance with wandb.
        3. Logs training batch performance with local csv (`log_batch`).
        4. Updates the training progress bar (`print_train_batch_status`).
        5. Logs structures.

    Parameters
    ----------
    losses

    """
    d_loss, ln_d_loss, m_loss_full, c_loss, loss = losses["drmsd-full"], losses["lndrmsd-full"], losses["mse-full"], \
                                              losses["combined-full"], losses["loss"]
    d_bb_loss, d_bb_ln_loss = losses["drmsd-bb"], losses["lndrmsd-bb"]

    metrics = update_metrics(metrics, losses, "train", src_seq, tracking_loss=loss, batch_level=True)

    do_log_str = not step or step % args.log_structure_step == 0
    do_log_lr  = args.lr_scheduling == "noam" and (not step or args.log_wandb_step % step == 0)

    if not step or step % args.log_wandb_step == 0:
        wandb.log({"Train Batch RMSE": np.sqrt(m_loss_full.item()),
                   "Train Batch DRMSD": d_loss,
                   "Train Batch ln-DRMSD": ln_d_loss,
                   "Train Batch Combined Loss": c_loss,
                   "Train Batch Speed": metrics["train"]["speed"],
                   "Batch size": pred_angs.shape[0],

                   "Train Batch DRMSD Backbone": losses["drmsd-bb"],
                   "Train Batch ln-DRMSD Backbone": losses["lndrmsd-bb"],
                   "Train Batch RMSE Backbone": np.sqrt(losses["mse-bb"].item()),
                   "Train Batch RMSE Sidechain": np.sqrt(losses["mse-sc"].item())}, commit=not do_log_lr and not do_log_str)
    if args.lr_scheduling == "noam":
        metrics["history-lr"].append(optimizer.cur_lr)
        if not step or step % args.log_wandb_step  == 0:
            wandb.log({"Learning Rate": optimizer.cur_lr}, commit=not do_log_str)

    log_batch(log_writer, metrics, start_time, mode="train", end_of_epoch=False)
    if pbar:
        print_train_batch_status(args, (pbar, metrics, src_seq))

    # Check for NaNs
    if np.isnan(loss.item()):
        print("A nan loss has occurred. Exiting training.")
        sys.exit(1)

    # Log the 16th structure of each validation set
    if args.log_val_struct_step != 0 and step % args.log_val_struct_step == 0:
        with torch.no_grad():
            for split, validation_dataset in validation_datasets.items():
                val_idx = len(validation_dataset.dataset) // 2
                val_src_seq, val_tgt_ang, val_tgt_crds = validation_dataset.dataset[val_idx : val_idx + 1]
                val_src_seq, val_tgt_ang, val_tgt_crds = map(lambda x: x.to(device),
                                                             paired_collate_fn(zip(val_src_seq, val_tgt_ang, val_tgt_crds)))
                val_pred_angs = model(val_src_seq, val_tgt_ang)
                pred_coords = angles_to_coords(inverse_trig_transform(val_pred_angs)[0].cpu(), val_src_seq[0].cpu(),
                    remove_batch_padding=True)
                log_structure_and_angs(args, val_pred_angs[0], pred_coords, val_tgt_crds[0], val_src_seq[0],
                                       commit=False, log_angs=False, struct_name=f"V{split}")

    if do_log_str:
        with torch.no_grad():
            pred_coords = angles_to_coords(inverse_trig_transform(pred_angs)[-1].cpu(), src_seq[-1].cpu(),
                                           remove_batch_padding=True)
        tgt_coords_unpadded = tgt_coords[-1:, :pred_coords.shape[0]]
        log_structure_and_angs(args, pred_angs[-1], pred_coords, tgt_coords_unpadded[-1], src_seq[-1], commit=True)
    return metrics


def log_angle_distributions(args, pred_ang, src_seq):
    """ Logs a histogram of predicted angles to wandb. """
    # Remove batch-level masking
    batch_mask = src_seq.ne(VOCAB.pad_id)
    pred_ang = pred_ang[batch_mask]
    inv_ang = inverse_trig_transform(pred_ang.view(1, pred_ang.shape[0], -1)).cpu().detach().numpy()
    pred_ang = pred_ang.cpu().detach().numpy()

    wandb.log({"Predicted Angles (sin cos)": wandb.Histogram(np_histogram=np.histogram(pred_ang)),
               "Predicted Angles (radians)": wandb.Histogram(np_histogram=np.histogram(inv_ang))}, commit=False)

    for sincos_idx in range(pred_ang.shape[-1]):
        wandb.log({f"Predicted Angles (sin cos) - {sincos_idx:02}":
                       wandb.Histogram(np_histogram=np.histogram(pred_ang[:,sincos_idx]))}, commit=False)

    for rad_idx in range(inv_ang.shape[-1]):
        wandb.log({f"Predicted Angles (radians) - {rad_idx:02}":
                       wandb.Histogram(np_histogram=np.histogram(inv_ang[0,:,rad_idx]))}, commit=False)


def do_eval_batch_logging(metrics, losses, src_seq, args, pbar, pred_angs, tgt_coords, mode, log_structures=False):
    """
       Performs all necessary logging at the end of a batch in an eval epoch.
       Updates custom metrics dictionary and wandb logs. Prints status of
       training.
       Also checks for NaN losses.

        1. Updates metrics.
        2. Logs training batch performance with wandb.
        3. Logs training batch performance with local csv (`log_batch`).
        4. Updates the training progress bar (`print_train_batch_status`).
        5. Logs structures.

    """
    
    metrics = update_metrics(metrics, losses, mode, src_seq, batch_level=True)
    print_eval_batch_status(args, (pbar, losses["drmsd-full"], mode, losses["mse-full"], losses["combined-full"]))

    if log_structures:
        with torch.no_grad():
            pred_coords = angles_to_coords(
                inverse_trig_transform(pred_angs)[-1].cpu(), src_seq[-1].cpu(),
                remove_batch_padding=True)
        log_structure_and_angs(args, pred_angs[-1], pred_coords, tgt_coords, src_seq[-1], commit=False)
    return metrics

def log_avg_validation_performance(metrics, validation_datasets):
    """
    After evaluating performance on all validation sets, this method records
    the average validation performance to wandb.
    """
    rmse_full, rmse_bb, rmse_sc, rmsd_full, drmsd_full, drmsd_bb, lndrmsd_full, lndrmsd_bb, combined_full = 0, 0, 0, 0, 0, 0, 0, 0, 0
    n = 0.
    for split, validation_data in validation_datasets.items():
        mode = f"valid-{split}"
        rmse_full += np.sqrt(metrics[mode]["epoch-mse-full"])
        rmsd_full += metrics[mode]["epoch-rmsd-full"]
        drmsd_full += metrics[mode]["epoch-drmsd-full"]
        lndrmsd_full += metrics[mode]["epoch-lndrmsd-full"]
        combined_full += metrics[mode]["epoch-combined-full"]

        rmse_bb += np.sqrt(metrics[mode]["epoch-mse-bb"])
        rmse_sc += np.sqrt(metrics[mode]["epoch-mse-sc"])
        drmsd_bb += metrics[mode]["epoch-drmsd-bb"]
        lndrmsd_bb += metrics[mode]["epoch-lndrmsd-bb"]

        n += 1

    wandb.log({"Valid-Avg Epoch RMSE": rmse_full/n,
               "Valid-Avg Epoch RMSD": rmsd_full/n,
               "Valid-Avg Epoch DRMSD": drmsd_full/n,
               "Valid-Avg Epoch ln-DRMSD": lndrmsd_full/n,
               "Valid-Avg Epoch Combined Loss": combined_full/n,

               "Valid-Avg Epoch RMSE Backbone": rmse_bb / n,
               "Valid-Avg Epoch RMSE Sidechain": rmse_sc / n,
               "Valid-Avg Epoch DRMSD Backbone": drmsd_bb / n,
               "Valid-Avg Epoch ln-DRMSD Backbone": lndrmsd_bb / n,
               }, commit=False)


def do_eval_epoch_logging(metrics, mode):
    """
    Performs all necessary logging at the end of an evaluation batch.
    Updates custom metrics dictionary and wandb logs. Prints status of training.
    """
    metrics = update_metrics_end_of_epoch(metrics, mode)

    wandb.log({f"{mode.title()} Epoch RMSE": np.sqrt(metrics[mode]["epoch-mse-full"]),
               f"{mode.title()} Epoch RMSD": metrics[mode]["epoch-rmsd-full"],
               f"{mode.title()} Epoch DRMSD": metrics[mode]["epoch-drmsd-full"],
               f"{mode.title()} Epoch ln-DRMSD": metrics[mode]["epoch-lndrmsd-full"],
               f"{mode.title()} Epoch Combined Loss": metrics[mode]["epoch-combined-full"],

               f"{mode.title()} Epoch ln-DRMSD Backbone": metrics[mode]["epoch-lndrmsd-bb"],
               f"{mode.title()} Epoch DRMSD Backbone": metrics[mode]["epoch-drmsd-bb"],
               f"{mode.title()} Epoch RMSE Backbone": np.sqrt(metrics[mode]["epoch-mse-bb"]),
               f"{mode.title()} Epoch RMSE Sidechain": np.sqrt(metrics[mode]["epoch-mse-sc"]),}, commit=False)


def log_structure_and_angs(args, pred_ang, pred_coords, true_coords, src_seq, commit, log_angs=True, struct_name="train"):
    """
    Logs a 3D structure prediction to wandb.
    """
    if log_angs:
        log_angle_distributions(args, pred_ang, src_seq)

    src_seq_cpu = src_seq.cpu().detach().numpy()

    # Make dir if needed
    cur_struct_path = os.path.join(args.structure_dir, struct_name)
    os.makedirs(cur_struct_path, exist_ok=True)

    # Remove coordinate level padding (each residue has about 13 atoms,
    # even if some are missing)
    gold_item_non_batch_pad = (true_coords != VOCAB.pad_id).any(dim=-1)
    true_coords = true_coords[gold_item_non_batch_pad]
    true_coords[torch.isnan(true_coords)] = 0

    creator = PDB_Creator(pred_coords.detach().numpy(),
                          seq=VOCAB.ints2str(src_seq_cpu))
    creator.save_pdb(f"{cur_struct_path}/{wandb.run.step:05}_pred.pdb",
                     title="pred")

    t_creator = PDB_Creator(true_coords.cpu().detach().numpy(),
                            seq=VOCAB.ints2str(src_seq_cpu))
    if not os.path.isfile(f"{cur_struct_path}/true.pdb") or struct_name == "train":
        t_creator.save_pdb(f"{cur_struct_path}/true.pdb", title="true")
        wandb.log({f"{struct_name}_mol_true" : wandb.Molecule(f"{cur_struct_path}/true.pdb")}, commit=False)

    gltf_out_path = os.path.join(args.gltf_dir, f"{wandb.run.step:05}_{struct_name}.gltf")
    t_creator.save_gltfs(f"{cur_struct_path}/true.pdb",
                         f"{cur_struct_path}/{wandb.run.step:05}_pred.pdb",
                         gltf_out_path=gltf_out_path,
                         make_pse=True,
                         make_png=args.save_pngs,
                         pse_out_path=f"{cur_struct_path}/{wandb.run.step:05}_both.pse")
    log_items = {struct_name: wandb.Object3D(gltf_out_path),
                 f"{struct_name}_mol": wandb.Molecule(f"{cur_struct_path}/{wandb.run.step:05}_pred.pdb"),
                 f"{struct_name}_mol_comb": wandb.Molecule(f"{cur_struct_path}/{wandb.run.step:05}_both.pdb")}
    if args.save_pngs:
        try:
            log_items[struct_name + "_img"]  = wandb.Image(gltf_out_path.replace("gltf", "png"))
        except FileNotFoundError:
            # Account for the possibility that a PyMol session may have failed to create successfully
            pass
    wandb.log(log_items, commit=commit)


def init_metrics(args):
    """
    Returns an empty metric dictionary for recording model performance.
    """
    metrics = {"train": {"epoch-history-drmsd": [],
                         "epoch-history-combined": [],
                         "epoch-history-lndrmsd": [],
                         "epoch-history-mse": []},
               "test":  {"epoch-history-drmsd": [],
                         "epoch-history-combined": [],
                         "epoch-history-lndrmsd": [],
                         "epoch-history-mse": []},
               "history-lr": [],
               "epoch_last_improved": -1,
               "best_valid_loss_so_far": np.inf,
               "last_chkpt_time": time.time(),
               "n_batches": 0
               }
    v_metrics = {}
    for split in VALID_SPLITS:
        v_metrics[f"valid-{split}"] = {"epoch-history-drmsd": [],
                                       "epoch-history-combined": [],
                                       "epoch-history-lndrmsd": [],
                                       "epoch-history-mse": []}
    metrics.update(v_metrics)
    if args.lr_scheduling != "noam":
        metrics["history-lr"] = [0]
    return metrics


def update_metrics(metrics, losses, mode, src_seq, tracking_loss=None, batch_level=True):
    """
    Records relevant metrics in the metrics data structure while training.
    If batch_level is true, this means the loss for the current batch is
    recorded in addition to the running epoch loss.

    Parameters
    ----------
    losses
    """
    drmsd, ln_drmsd, mse, combined, rmsd = losses["drmsd-full"], losses["lndrmsd-full"], losses["mse-full"], losses["combined-full"], losses["rmsd-full"]
    # Update loss values
    if batch_level:
        metrics["n_batches"] += 1
        metrics[mode]["batch-drmsd-full"] = drmsd.item()
        metrics[mode]["batch-lndrmsd-full"] = ln_drmsd.item()
        metrics[mode]["batch-mse-full"] = mse.item()
        metrics[mode]["batch-combined-full"] = combined.item()
        if rmsd: metrics[mode]["batch-rmsd-full"] = rmsd.item()
        metrics[mode]["batch-drmsd-bb"] = losses["drmsd-bb"].item()
        metrics[mode]["batch-mse-bb"] = losses["mse-bb"].item()
        metrics[mode]["batch-mse-sc"] = losses["mse-sc"].item()
        metrics[mode]["batch-lndrmsd-bb"] = losses["lndrmsd-bb"].item()
    metrics[mode]["epoch-drmsd-full"] += drmsd.item()
    metrics[mode]["epoch-lndrmsd-full"] += ln_drmsd.item()
    metrics[mode]["epoch-mse-full"] += mse.item()
    metrics[mode]["epoch-combined-full"] += combined.item()
    if rmsd: metrics[mode]["epoch-rmsd-full"] += rmsd.item()
    metrics[mode]["epoch-drmsd-bb"] = losses["drmsd-bb"].item()
    metrics[mode]["epoch-mse-bb"] = losses["mse-bb"].item()
    metrics[mode]["epoch-mse-sc"] = losses["mse-sc"].item()
    metrics[mode]["epoch-lndrmsd-bb"] = losses["lndrmsd-bb"].item()

    # Compute and update speed
    num_res = (src_seq != VOCAB.pad_id).sum().item()
    metrics[mode]["speed"] = num_res / (time.time() - metrics[mode]["batch-time"])
    if "speeds" not in metrics[mode].keys():
        metrics[mode]["speeds"] = []
    metrics[mode]["speeds"].append(metrics[mode]["speed"])

    metrics[mode]["batch-time"] = time.time()
    metrics[mode]["speed-history"].append(metrics[mode]["speed"])

    if tracking_loss:
        metrics[mode]["batch-history"].append(float(tracking_loss))
    return metrics


def reset_metrics_for_epoch(metrics, mode):
    """
    Resets the running and batch-specific metrics for a new epoch.
    """
    metrics[mode]["epoch-drmsd-full"] = metrics[mode]["batch-drmsd-full"] = 0
    metrics[mode]["epoch-lndrmsd-full"] = metrics[mode]["batch-lndrmsd-full"] = 0
    metrics[mode]["epoch-mse-full"] = metrics[mode]["batch-mse-full"] = 0
    metrics[mode]["epoch-combined-full"] = metrics[mode]["batch-combined-full"] = 0
    metrics[mode]["epoch-rmsd-full"] = metrics[mode]["batch-rmsd-full"] = 0

    metrics[mode]["epoch-drmsd-bb"] = metrics[mode]["batch-drmsd-bb"] = 0
    metrics[mode]["epoch-lndrmsd-bb"] = metrics[mode]["batch-lndrmsd-bb"] = 0
    metrics[mode]["epoch-mse-sc"] = metrics[mode]["batch-mse-sc"] = 0
    metrics[mode]["epoch-mse-bb"] = metrics[mode]["batch-mse-bb"] = 0

    metrics[mode]["batch-history"] = []
    metrics[mode]["batch-time"] = time.time()
    metrics[mode]["speed-history"] = []
    metrics["n_batches"] = 0
    return metrics


def update_metrics_end_of_epoch(metrics, mode):
    """
    Averages the running metrics over an epoch
    """
    n_batches = metrics["n_batches"]
    metrics[mode]["epoch-drmsd-full"] /= n_batches
    metrics[mode]["epoch-lndrmsd-full"] /= n_batches
    metrics[mode]["epoch-mse-full"] /= n_batches

    metrics[mode]["epoch-drmsd-bb"] /= n_batches
    metrics[mode]["epoch-lndrmsd-bb"] /= n_batches
    metrics[mode]["epoch-mse-bb"] /= n_batches
    metrics[mode]["epoch-mse-sc"] /= n_batches

    if metrics[mode]["epoch-drmsd-full"] == 0:
        metrics[mode]["epoch-combined-full"] = 0
    else:
        metrics[mode]["epoch-combined-full"] /= n_batches

    metrics[mode]["epoch-rmsd-full"] /= n_batches
    metrics[mode]["epoch-history-combined"].append(metrics[mode]["epoch-combined-full"])
    metrics[mode]["epoch-history-drmsd"].append(metrics[mode]["epoch-drmsd-full"])
    metrics[mode]["epoch-history-mse"].append(metrics[mode]["epoch-mse-full"])
    metrics[mode]["epoch-history-lndrmsd"].append(metrics[mode]["epoch-lndrmsd-full"])


    return metrics


def prepare_log_header(args):
    """
    Returns the column ordering for the logfile.
    """
    if args.loss == "combined":
        return 'drmsd,ln_drmsd,rmse,rmsd,combined,lr,mode,granularity,time,speed\n'
    else:
        return 'drmsd,ln_drmsd,rmse,rmsd,lr,mode,granularity,time,speed\n'


class EarlyStoppingCondition(Exception):
    """
    An exception to raise when Early Stopping conditions are met.
    """
    def __init__(self, *args):
        super().__init__(*args)
