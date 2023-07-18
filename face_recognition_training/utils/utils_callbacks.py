import logging
import os
import time
from typing import List
import torch

from utils import verification
from utils.utils_logging import AverageMeter


class CallBackVerification(object):
    def __init__(
        self,
        frequent,
        rank,
        val_targets,
        rec_prefix,
        img_size,
        gen_im_path=None,
        is_vae=False,
    ):
        self.frequent: int = frequent
        self.rank: int = rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.gen_im_path = gen_im_path
        self.is_vae = is_vae
        if self.rank == 0:
            self.ver_list, self.ver_name_list = self.init_dataset(
                val_targets=val_targets, data_dir=rec_prefix, image_size=img_size
            )

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                self.ver_list[i],
                backbone,
                batch_size=10,
                nfolds=10,
                gen_im_path=self.gen_im_path,
                is_vae=self.is_vae,
            )
            logging.info(
                "[%s][%d]XNorm: %f" % (self.ver_name_list[i], global_step, xnorm)
            )
            logging.info(
                "[%s][%d]Accuracy-Flip: %1.5f+-%1.5f"
                % (self.ver_name_list[i], global_step, acc2, std2)
            )
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                "[%s][%d]Accuracy-Highest: %1.5f"
                % (self.ver_name_list[i], global_step, self.highest_acc_list[i])
            )
            results.append(acc2)
        return results

    def ver_test_da(self, backbone, global_step, ranking, curr_beta, modified_neurons, means):
        results = []
        for i in range(len(self.ver_list)):
            acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test_da(
                self.ver_list[i],
                backbone,
                batch_size=10,
                nfolds=10,
                ranking=ranking,
                curr_beta=curr_beta,
                modified_neurons=modified_neurons,
                means=means
            )
            logging.info(
                "[%s][%d]XNorm: %f" % (self.ver_name_list[i], global_step, xnorm)
            )
            logging.info(
                "[%s][%d]Accuracy-Flip: %1.5f+-%1.5f"
                % (self.ver_name_list[i], global_step, acc2, std2)
            )
            if acc2 > self.highest_acc_list[i]:
                self.highest_acc_list[i] = acc2
            logging.info(
                "[%s][%d]Accuracy-Highest: %1.5f"
                % (self.ver_name_list[i], global_step, self.highest_acc_list[i])
            )
            results.append(acc2)
        return results

    def init_dataset(self, val_targets, data_dir, image_size):
        ver_list = []
        ver_name_list = []
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size)
                ver_list.append(data_set)
                ver_name_list.append(name)
        return ver_list, ver_name_list

    def __call__(self, num_update, backbone: torch.nn.Module, do_da=False, ranking=[], curr_beta=0, modified_neurons=[], means=[]):
        results = []
        if self.rank == 0 and num_update % self.frequent == 0:
            backbone.eval()
            if do_da:
                results = self.ver_test_da(backbone, num_update, ranking, curr_beta, modified_neurons, means)
            else:
                results = self.ver_test(backbone, num_update)
            backbone.train()
        return results


class CallBackLogging(object):
    def __init__(self, frequent, rank, total_step, batch_size, world_size):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size

        self.init = False
        self.tic = 0

    def __call__(
        self,
        global_step,
        loss: AverageMeter,
        acc1: AverageMeter,
        acc5: AverageMeter,
        epoch: int,
        loss2: AverageMeter = None,
    ):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = (
                        self.frequent * self.batch_size / (time.time() - self.tic)
                    )
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float("inf")

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step + 1) / self.total_step)
                time_for_end = time_total - time_now

                msg = "Epoch: {:>2}  Speed {:.2f} samples/sec   Loss {:.4f}   Acc1 {:.2f}   Acc5 {:.2f}   Step: {:>4}/{}   Required: {:.1f} hours".format(
                    epoch,
                    speed_total,
                    loss.avg,
                    acc1.avg,
                    acc5.avg,
                    global_step,
                    self.total_step,
                    time_for_end,
                )
                if not loss2 is None:
                    msg = "Epoch: {:>2}  Speed {:.2f} samples/sec   Loss {:.4f}   Loss2 {:.4f}   Acc1 {:.2f}   Acc5 {:.2f}   Step: {:>4}/{}   Required: {:.1f} hours".format(
                        epoch,
                        speed_total,
                        loss.avg,
                        loss2.avg,
                        acc1.avg,
                        acc5.avg,
                        global_step,
                        self.total_step,
                        time_for_end,
                    )
                    loss2.reset()
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(self, epoch, backbone: torch.nn.Module):
        if self.rank == 0:
            torch.save(
                backbone.module.state_dict(),
                os.path.join(self.output, "checkpoint_{:03d}.pth".format(epoch)),
            )


class CallBackModelCheckpointOld(object):
    def __init__(self, rank, output="./"):
        self.rank: int = rank
        self.output: str = output

    def __call__(
        self,
        global_step,
        backbone: torch.nn.Module,
        header: torch.nn.Module = None,
        quantiza: bool = False,
    ):
        if quantiza:
            if global_step > 100 and self.rank == 0:
                torch.save(
                    backbone.module,
                    os.path.join(self.output, str(global_step) + "backbone.pt"),
                )
        else:
            if global_step > 100 and self.rank == 0:
                torch.save(
                    backbone.module.state_dict(),
                    os.path.join(self.output, str(global_step) + "backbone.pth"),
                )
            if global_step > 100 and self.rank == 0 and header is not None:
                torch.save(
                    header.module.state_dict(),
                    os.path.join(self.output, str(global_step) + "header.pth"),
                )
