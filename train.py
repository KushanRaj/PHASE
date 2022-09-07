import argparse
import utils
import random
import numpy as np
import torch
import dataset
from torch.utils.data import DataLoader, DistributedSampler
import os
from torch.nn.parallel import DistributedDataParallel
import wandb
import json
import network
from torch.autograd import grad
from utils import plot_surface

def W(x):

    return x**2 - 2*abs(x) + 1

def compute_grad(inputs, outputs):

    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs, requires_grad=False, device=outputs.device),
        )
        
    return points_grad[0]

class Trainer:
    def __init__(self, args):
        self.args = args
        self.out_dir = args.out_dir

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        self.device, local_rank = utils.setup_device(args.dist)
        self.main_thread = True if local_rank == 0 else False
        if self.main_thread:
            print(f"\nsetting up device, distributed = {args.dist}")
        print(f" | {self.device}")

        train_dset = dataset.PCLoader(root = args.root,
                                      name = args.name,
                                      points_batch=args.points_batch,
                                      with_normals=args.with_normals)
        if self.main_thread:
            print(f"setting up dataset, train: {len(train_dset)}")
        if args.dist:
            train_sampler = DistributedSampler(train_dset)
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                sampler=train_sampler,
                num_workers=args.n_workers,
            )
        else:
            self.train_loader = DataLoader(
                train_dset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.n_workers,
            )

        model = network.ImplicitNet(d_in = args.d_in,
                                    dims = [int(i) for i in args.dims.split(',')],
                                    skip = [int(i) for i in args.dims.skip(',')],
                                    geometric_init=args.geometric_init,
                                    radius_init=args.radius_int,
                                    beta = args.beta)
                    
        if args.dist:
            torch.set_num_threads(1)
            self.model = DistributedDataParallel(
                model.to(self.device),
                device_ids=[local_rank],
                output_device=local_rank,
            )
        else:
            self.model = model.to(self.device)
        if self.main_thread:
            print(f"# of model parameters: {sum(p.numel() for p in self.model.parameters())/1e6}M")

        if self.args.lr_step_mode == "epoch":
            total_steps = args.epochs - args.warmup
        else:
            total_steps = int(args.epochs * len(self.train_loader) - args.warmup)
        if args.warmup > 0:
            for group in self.optim.param_groups:
                group["lr"] = 1e-12 * group["lr"]
        if args.lr_sched == "cosine":
            self.lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, total_steps)
        elif args.lr_sched == 'step':
            self.lr_sched = torch.optim.lr_scheduler.StepLR(self.optim, args.step_size, args.lr_decay)
        elif args.lr_sched == "multi_step":
            milestones = [
                int(milestone) - total_steps for milestone in args.lr_decay_steps.split(",")
            ]
            self.lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=milestones, gamma=args.lr_decay
            )
        else:
            raise ValueError(f"args.lr_sched_type = {args.lr_sched} not implemented")

        if os.path.exists(os.path.join(self.out_dir, "last.ckpt")):
            if args.resume == False and self.main_thread:
                raise ValueError(
                    f"directory {self.out_dir} already exists, change output directory or use --resume argument"
                )
            ckpt = torch.load(os.path.join(self.out_dir, "last.ckpt"), map_location=self.device)
            model_dict = ckpt["model"]
            if "module" in list(model_dict.keys())[0] and args.dist == False:
                model_dict = {
                    key.replace("module.", ""): value for key, value in model_dict.items()
                }
            self.model.load_state_dict(model_dict)
            self.optim.load_state_dict(ckpt["optim"])
            self.lr_sched.load_state_dict(ckpt["lr_sched"])
            self.start_epoch = ckpt["epoch"] + 1
            if self.main_thread:
                print(
                    f"loaded checkpoint, resuming training expt from {self.start_epoch} to {args.epochs} epochs."
                )
        else:
            if args.resume == True and self.main_thread:
                raise ValueError(
                    f"resume training args are true but no checkpoint found in {self.out_dir}"
                )
            os.makedirs(self.out_dir, exist_ok=True)
            with open(os.path.join(self.out_dir, "args.txt"), "w") as f:
                json.dump(args.__dict__, f, indent=4)
            self.start_epoch = 0
            if self.main_thread:
                print(f"starting fresh training expt for {args.epochs} epochs.")
        self.train_steps = self.start_epoch * len(self.train_loader)

        self.log_wandb = False
        self.metric_meter = utils.AvgMeter()
        self.time_meter = utils.TimeMeter()
        if self.main_thread:
            self.log_f = open(os.path.join(self.out_dir, "logs.txt"), "w")
            print(f"start file logging @ {os.path.join(self.out_dir, 'logs.txt')}")
            if args.wandb:
                self.log_wandb = True
                run = wandb.init()
                print(f"start wandb logging @ {run.get_url()}")
                self.log_f.write(f"\nwandb url @ {run.get_url()}\n")

        # grid_size = 1.5
        # num_x = 100
        # num_y = 100
        # num_z = 100
        # grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
        # grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)
        # grid_z = torch.linspace(-grid_size, grid_size, steps=num_z)

        # x, y, z = torch.meshgrid(grid_x, grid_y, grid_z)
        # self.grid = torch.stack([x, y, z]).permute(1, 2, 3, 0).numpy().reshape(-1, 3)

    def train_epoch(self):
        self.metric_meter.reset()
        self.time_meter.reset()
        self.model.train()
        for indx, (points, normals) in enumerate(self.train_loader):
            
            points = points.to(self.device)

            B, N, _ = points.shape

            if self.with_normals:
                normals = normals.cuda()

            reconstruction_points = utils.sample_local_points(points)
            sampled_points = utils.sample_global_points()

            sampled_points.requires_grad = True
            reconstruction_points.requires_grad = True

            points_density = self.model(points).view(B, N)
            reconstruction_density = self.model(reconstruction_points).view(B, N)
            sampled_density = self.model(sampled_points).view(B, N)

            perimeter_loss = compute_grad(sampled_points, sampled_density).norm(2, -1).sum(-1).mean()*(1.5**3)/N
            sdf_loss = W(sampled_density).sum(-1).mean()*(1.5**3)/N

            if self.with_normals:
                point_normal_loss = (normals - compute_grad(points, points_density)).norm(2, -1).mean()
            else:
                point_normal_loss = (1 - compute_grad(points, points_density)).norm(2, -1).mean()

            reconstruction_loss = reconstruction_density.sum(-1).abs().mean()*(1.5**3)/N

            loss = reconstruction_loss*self.args.eta + self.args.lbda*perimeter_loss + self.args.mu*point_normal_loss + sdf_loss

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            metrics = {"train_loss": loss.item(), "recon" : reconstruction_loss.item(), "lbda" : perimeter_loss.item(), "normal" : point_normal_loss.item(), "sdf" : sdf_loss.item()}
            self.metric_meter.add(metrics)

            if self.main_thread and indx % self.args.log_every == 0:
                if self.log_wandb:
                    wandb.log({"train step": self.train_steps, "train loss": loss.item()})
                utils.pbar(
                    indx / len(self.train_loader),
                    msg=self.metric_meter.msg() + self.time_meter.msg(),
                )

            if self.args.lr_step_mode == "step":
                if self.train_steps < self.args.warmup and self.args.warmup > 0:
                    self.optim.param_groups[0]["lr"] = (
                        self.train_steps / (self.args.warmup) * self.args.lr
                    )
                else:
                    self.lr_sched.step()

            self.train_steps += 1
        if self.main_thread:
            utils.pbar(1, msg=self.metric_meter.msg() + self.time_meter.msg())

    @torch.no_grad()
    def plot(self, points, epoch):
        self.model.eval()
        plot_surface(with_points=True,
                    points=points,
                         decoder=self.model,
                         path=self.args.out_dir,
                         epoch=epoch,
                         shapename=self.args.name,
                         mc_value=0, save_html=self.args.save_html, resolution=self.args.resolution)

    def train(self):
        best_train, best_val = float('inf'), float('inf')
        for epoch in range(self.start_epoch, self.args.epochs):
            if self.main_thread:
                print(
                    f"epoch: {epoch}, best train: {round(best_train, 5)}, best val: {round(best_val, 5)}, lr: {round(self.optim.param_groups[0]['lr'], 5)}"
                )
                print("---------------")

            self.train_epoch()
            if self.main_thread:
                train_metrics = self.metric_meter.get()
                if train_metrics["train_loss"] < best_train:
                    print(
                        "\x1b[34m"
                        + f"train loss improved from {round(best_train, 5)} to {round(train_metrics['train_loss'], 5)}"
                        + "\033[0m"
                    )
                    best_train = train_metrics["train_loss"]

                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.args.out_dir, f"best.ckpt"),
                    )
                msg = f"epoch: {epoch}, last train: {round(train_metrics['train_loss'], 5)}, best train: {round(best_train, 5)}"

                self.log_f.write(msg + f", lr: {round(self.optim.param_groups[0]['lr'], 5)}\n")
                self.log_f.flush()

                if self.log_wandb:

                    if epoch % self.args.plot_every == 0:

                        self.plot(self.grid)
                    

                    train_metrics = {"epoch " + key: value for key, value in train_metrics.items()}
                    wandb.log(
                        {
                            "epoch": epoch,
                            **train_metrics,
                            "lr": self.optim.param_groups[0]["lr"],
                        }
                    )

                torch.save(
                    {
                        "model": self.model.state_dict(),
                        "optim": self.optim.state_dict(),
                        "lr_sched": self.lr_sched.state_dict(),
                        "epoch": epoch,
                    },
                    os.path.join(self.args.out_dir, "last.ckpt"),
                )

            if self.args.lr_step_mode == "epoch":
                if epoch <= self.args.warmup and self.args.warmup > 0:
                    self.optim.param_groups[0]["lr"] = epoch / self.args.warmup * self.args.lr
                else:
                    self.lr_sched.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = utils.add_args(parser)
    args = parser.parse_args()
    utils.print_args(args)

    trainer = Trainer(args)
    trainer.train()

    if args.dist:
        torch.distributed.destroy_process_group()