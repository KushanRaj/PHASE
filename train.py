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
import trimesh

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        assert len(scene_or_mesh.geometry) > 0
        mesh = trimesh.util.concatenate(
            tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                for g in scene_or_mesh.geometry.values()))
    else:
        assert isinstance(scene_or_mesh, trimesh.Trimesh)
        mesh = scene_or_mesh
    return mesh

def sample_mesh(m, n):
    vpos, _ = trimesh.sample.sample_surface(m, n)
    return torch.tensor(vpos, dtype=torch.float32)

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def W(x):

    return x**2 - 2*abs(x) + 1

def compute_grad(inputs, outputs):

    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs, requires_grad=False, device=outputs.device),
        retain_graph=True
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
                                      with_normals=args.with_normals,
                                      batch_size = args.batch_size,)
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

        embed_kwargs = {
                'include_input' : True,
                'input_dims' : args.d_in,
                'max_freq_log2' : args.multires-1,
                'num_freqs' : args.multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
        }

        self.embed = Embedder(**embed_kwargs)

        model = network.ImplicitNet(d_in = self.embed.out_dim,
                                    dims = [int(i) for i in args.dims.split(',')],
                                    skip_in = [int(i) for i in args.skip.split(',')],
                                    geometric_init=args.geometric_init,
                                    radius_init=args.radius_init,
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

        self.optim = torch.optim.Adam(self.model.parameters(), lr = self.args.lr, weight_decay = 0)

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
            self.lr_sched = torch.optim.lr_scheduler.StepLR(self.optim, args.lr_step_size, args.lr_decay)
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

    def train_epoch(self):
        self.metric_meter.reset()
        self.time_meter.reset()
        self.model.train()
        for indx, (points, normals, sampled_points) in enumerate(self.train_loader):
            
            points = points.to(self.device)
            sampled_points = sampled_points.to(self.device)

            B, N, _ = points.shape

            if self.args.with_normals:
                normals = normals.to(self.device)

            reconstruction_points = utils.sample_local_points(points, local_sigma = self.args.local_sigma, sample_size = self.args.sample_N).to(self.device)
            #sampled_points = utils.sample_global_points(points.shape).to(self.device)

            points = self.embed(points)
            reconstruction_points = self.embed(reconstruction_points)

            sampled_points.requires_grad = True
            reconstruction_points.requires_grad = True
            points.requires_grad = True

            sampled_points_embed = self.embed(sampled_points)

            points_density = self.model(points).view(B, N)
            reconstruction_density = self.model(reconstruction_points).view(B, N*self.args.sample_N)
            sampled_density = self.model(sampled_points_embed).view(B, N)

            perimeter_loss = compute_grad(sampled_points, sampled_density).norm(2, -1).mean(-1).mean()
            sdf_loss = W(sampled_density).mean(-1).mean()

            if self.args.use_normal:
                point_normal_loss = (normals - (self.args.eta**0.5)*points_density.unsqueeze(-1)).abs().mean(-1).mean()
            else:
                point_normal_loss = (1 - ((self.args.eta**0.5)*(points_density.unsqueeze(-1).norm(2, -1)))).pow(2).mean(-1).mean()

            reconstruction_loss = reconstruction_density.view(B, N, self.args.sample_N).mean(-1).abs().mean(-1).mean()

            del reconstruction_points, points, reconstruction_density, points_density

            #my failed loss
            # sampled_reconstruction_points = utils.sample_local_points(sampled_points, local_sigma = self.args.local_sigma, sample_size = self.args.sample_N).to(self.device)
            # sampled_reconstruction_loss = (self.model(sampled_reconstruction_points).view(B, N, self.args.sample_N) - sampled_density.unsqueeze(-1)).norm(2, -1).mean(-1).mean(-1).mean()

            loss = (reconstruction_loss)*self.args.lbda + self.args.eta*perimeter_loss + self.args.mu*point_normal_loss + sdf_loss

            self.optim.zero_grad()

            loss.backward()

            self.optim.step()

            metrics = {"loss": loss.item(), "rec" : reconstruction_loss.item(), "peri" : perimeter_loss.item(), "sdf" : sdf_loss.item(), "norm" : point_normal_loss.item()}#, 'sample' : sampled_reconstruction_loss.item()}
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
    def plot(self, epoch):
        self.model.eval()
        plot_surface(
                    decoder=self.model,
                    path=self.args.out_dir,
                    epoch=epoch,
                    shapename=self.args.name,
                    mc_value=0, 
                    save_html=self.args.save_html, 
                    resolution=self.args.resolution, 
                    verbose = self.args.plot_verbose, 
                    save_ply = self.args.save_ply,
                    embed = self.embed,
                    )

    def train(self):
        best_train, best_val = float('inf'), float('inf')
        for epoch in range(self.start_epoch, self.args.epochs):
            if self.main_thread:
                print(
                    f"epoch: {epoch}, steps : {self.train_steps}, best train: {round(best_train, 5)}, lr: {round(self.optim.param_groups[0]['lr'], 5)}"
                )
                print("---------------")

            self.train_epoch()
            if self.main_thread:
                train_metrics = self.metric_meter.get()
                if train_metrics["loss"] < best_train:
                    print(
                        "\x1b[34m"
                        + f"train loss improved from {round(best_train, 5)} to {round(train_metrics['loss'], 5)}"
                        + "\033[0m"
                    )
                    best_train = train_metrics["loss"]

                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.args.out_dir, f"best.ckpt"),
                    )
                msg = f"epoch: {epoch}, last train: {round(train_metrics['loss'], 5)}, best train: {round(best_train, 5)}"

                self.log_f.write(msg + f", lr: {round(self.optim.param_groups[0]['lr'], 5)}\n")
                self.log_f.flush()

                if self.log_wandb:

                    train_metrics = {"epoch " + key: value for key, value in train_metrics.items()}
                    wandb.log(
                        {
                            "epoch": epoch,
                            **train_metrics,
                            "lr": self.optim.param_groups[0]["lr"],
                        }
                    )

                if epoch % self.args.plot_every == 0:

                    self.plot(epoch)

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