import plotly.graph_objs as go
import plotly.offline as offline
import torch
import numpy as np
from skimage import measure
from datetime import datetime
import time
import os

def sample_local_points(pc_input, local_sigma=1e-3):

    batch_size, sample_size, dim = pc_input.shape

    sample_local = pc_input + (torch.randn_like(pc_input) * local_sigma)
    
    return sample_local

def sample_global_points(shape):

    return (torch.rand(shape) - 0.5) * 3

def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj

def get_threed_scatter_trace(points,caption = None,colorscale = None,color = None):

    if (type(points) == list):
        trace = [go.Scatter3d(
            x=p[0][:, 0],
            y=p[0][:, 1],
            z=p[0][:, 2],
            mode='markers',
            name=p[1],
            marker=dict(
                size=3,
                line=dict(
                    width=2,
                ),
                opacity=0.9,
                colorscale=colorscale,
                showscale=True,
                color=color,
            ), text=caption) for p in points]

    else:

        trace = [go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            name='projection',
            marker=dict(
                size=3,
                line=dict(
                    width=2,
                ),
                opacity=0.9,
                colorscale=colorscale,
                showscale=True,
                color=color,
            ), text=caption)]

    return trace


def plot_threed_scatter(points,path,epoch,in_epoch):
    trace = get_threed_scatter_trace(points)
    layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                                           yaxis=dict(range=[-2, 2], autorange=False),
                                                           zaxis=dict(range=[-2, 2], autorange=False),
                                                           aspectratio=dict(x=1, y=1, z=1)))

    fig1 = go.Figure(data=trace, layout=layout)

    filename = '{0}/scatter_iteration_{1}_{2}.html'.format(path, epoch, in_epoch)
    offline.plot(fig1, filename=filename, auto_open=False)


def plot_surface(decoder,path,epoch, shapename,resolution,mc_value,verbose,save_html,save_ply,overwrite = True, points=None, with_points=False, latent=None, connected=False, is_uniform_grid = True):

    filename = f'{path}/{epoch}_{shapename}'

    if (not os.path.exists(filename) or overwrite):

        if with_points:
            pnts_val = decoder(points)
            pnts_val = pnts_val.cpu()
            points = points.cpu()
            caption = ["decoder : {0}".format(val.item()) for val in pnts_val.squeeze()]
            trace_pnts = get_threed_scatter_trace(points[:,-3:],caption=caption)

        surface = get_surface_trace(points,decoder,latent,resolution,mc_value,is_uniform_grid,verbose,save_ply, connected)
        trace_surface = surface["mesh_trace"]

        layout = go.Layout(title= go.layout.Title(text=shapename), width=1200, height=1200, scene=dict(xaxis=dict(range=[-2, 2], autorange=False),
                                                               yaxis=dict(range=[-2, 2], autorange=False),
                                                               zaxis=dict(range=[-2, 2], autorange=False),
                                                               aspectratio=dict(x=1, y=1, z=1)))
        if (with_points):
            fig1 = go.Figure(data=trace_pnts + trace_surface, layout=layout)
        else:
            fig1 = go.Figure(data=trace_surface, layout=layout)


        if (save_html):
            offline.plot(fig1, filename=filename + '.html', auto_open=False)
        if (not surface['mesh_export'] is None):
            surface['mesh_export'].export(filename + '.ply', 'ply')
        return surface['mesh_export']


def get_surface_trace(points,decoder,latent,resolution,mc_value,is_uniform,verbose,save_ply, connected=False):

    trace = []
    meshexport = None

    if (is_uniform):
        grid = get_grid_uniform(resolution)
    else:
        if not points is None:
            grid = get_grid(points[:,-3:],resolution)
        else:
            grid = get_grid(None, resolution)

    z = []
    for i,pnts in enumerate(torch.split(grid['grid_points'],100000,dim=0)):
        if (verbose):
            print ('{0}'.format(i/(grid['grid_points'].shape[0] // 100000) * 100))

        if (not latent is None):
            pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
        z.append(decoder(pnts).detach().cpu().numpy())
    z = np.concatenate(z,axis=0)

    if (not (np.min(z) > mc_value or np.max(z) < mc_value)):

        import trimesh
        z  = z.astype(np.float64)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=mc_value,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0],grid['xyz'][1][0],grid['xyz'][2][0]])
        if (save_ply):
            meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
            if connected:
                connected_comp = meshexport.split(only_watertight=False)
                max_area = 0
                max_comp = None
                for comp in connected_comp:
                    if comp.area > max_area:
                        max_area = comp.area
                        max_comp = comp
                meshexport = max_comp

        def tri_indices(simplices):
            return ([triplet[c] for triplet in simplices] for c in range(3))

        I, J, K = tri_indices(faces)

        trace.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                          i=I, j=J, k=K, name='',
                          color='orange', opacity=0.5))

    return {"mesh_trace":trace,
            "mesh_export":meshexport}


def plot_cuts_axis(points,decoder,latent,path,epoch,near_zero,axis,file_name_sep='/'):
    onedim_cut = np.linspace(-1.0, 1.0, 200)
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_axis = points[:,axis].min(dim=0)[0].item()
    max_axis = points[:,axis].max(dim=0)[0].item()
    mask = np.zeros(3)
    mask[axis] = 1.0
    if (axis == 0):
        position_cut = np.vstack(([np.zeros(xx.shape[0]), xx, yy]))
    elif (axis == 1):
        position_cut = np.vstack(([xx,np.zeros(xx.shape[0]), yy]))
    elif (axis == 2):
        position_cut = np.vstack(([xx, yy, np.zeros(xx.shape[0])]))
    position_cut = [position_cut + i*mask.reshape(-1, 1) for i in np.linspace(min_axis - 0.1, max_axis + 0.1, 50)]
    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)

        field_input = to_cuda(torch.tensor(pos.T, dtype=torch.float))
        z = []
        for i, pnts in enumerate(torch.split(field_input, 10000, dim=0)):
            if (not latent is None):
                pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
            z.append(decoder(pnts).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (near_zero):
            if (np.min(z) < -1.0e-5):
                start = -0.1
            else:
                start = 0.0
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='axis {0} = {1}'.format(axis,pos[axis, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=start,
                                     end=0.1,
                                     size=0.01
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )
        else:
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='axis {0} = {1}'.format(axis,pos[axis, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=True,
                                ncontours=70
                                # contours=dict(
                                #      start=-0.001,
                                #      end=0.001,
                                #      size=0.00001
                                #      )
                                # ),colorbar = {'dtick': 0.05}
                                )

        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                               yaxis=dict(range=[-1, 1], autorange=False),
                                                               aspectratio=dict(x=1, y=1)),
                           title=dict(text='axis {0} = {1}'.format(axis,pos[axis, 0])))
        # fig['layout']['xaxis2'].update(range=[-1, 1])
        # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)

        filename = '{0}{1}cutsaxis_{2}_{3}_{4}.html'.format(path,file_name_sep,axis, epoch, index)
        fig1 = go.Figure(data=[trace1], layout=layout)
        offline.plot(fig1, filename=filename, auto_open=False)


def plot_cuts(points,decoder,path,epoch,near_zero,latent=None):
    onedim_cut = np.linspace(-1, 1, 200)
    xx, yy = np.meshgrid(onedim_cut, onedim_cut)
    xx = xx.ravel()
    yy = yy.ravel()
    min_y = points[:,-2].min(dim=0)[0].item()
    max_y = points[:,-2].max(dim=0)[0].item()
    position_cut = np.vstack(([xx, np.zeros(xx.shape[0]), yy]))
    position_cut = [position_cut + np.array([0., i, 0.]).reshape(-1, 1) for i in np.linspace(min_y - 0.1, max_y + 0.1, 10)]
    for index, pos in enumerate(position_cut):
        #fig = tools.make_subplots(rows=1, cols=1)

        field_input = torch.tensor(pos.T, dtype=torch.float).cuda()
        z = []
        for i, pnts in enumerate(torch.split(field_input, 1000, dim=-1)):
            input_=pnts
            if (not latent is None):
                input_ = torch.cat([latent.expand(pnts.shape[0],-1) ,pnts],dim=1)
            z.append(decoder(input_).detach().cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (near_zero):
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=False,
                                contours=dict(
                                     start=-0.001,
                                     end=0.001,
                                     size=0.00001
                                     )
                                # ),colorbar = {'dtick': 0.05}
                                )
        else:
            trace1 = go.Contour(x=onedim_cut,
                                y=onedim_cut,
                                z=z.reshape(onedim_cut.shape[0], onedim_cut.shape[0]),
                                name='y = {0}'.format(pos[1, 0]),  # colorbar=dict(len=0.4, y=0.8),
                                autocontour=True,
                                # contours=dict(
                                #      start=-0.001,
                                #      end=0.001,
                                #      size=0.00001
                                #      )
                                # ),colorbar = {'dtick': 0.05}
                                )

        layout = go.Layout(width=1200, height=1200, scene=dict(xaxis=dict(range=[-1, 1], autorange=False),
                                                               yaxis=dict(range=[-1, 1], autorange=False),
                                                               aspectratio=dict(x=1, y=1)),
                           title=dict(text='y = {0}'.format(pos[1, 0])))
        # fig['layout']['xaxis2'].update(range=[-1, 1])
        # fig['layout']['yaxis2'].update(range=[-1, 1], scaleanchor="x2", scaleratio=1)

        filename = '{0}/cuts{1}_{2}.html'.format(path, epoch, index)
        fig1 = go.Figure(data=[trace1], layout=layout)
        offline.plot(fig1, filename=filename, auto_open=False)


def get_grid(points,resolution):
    eps = 0.1
    input_min = torch.min(points, dim=0)[0].squeeze().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().cpu().numpy()
    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if (shortest_axis == 0):
        x = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(x) - np.min(x)
        y = np.arange(input_min[1] - eps, input_max[1] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (x.shape[0] - 1) + eps, length / (x.shape[0] - 1))
    elif (shortest_axis == 1):
        y = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(y) - np.min(y)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
        z = np.arange(input_min[2] - eps, input_max[2] + length / (y.shape[0] - 1) + eps, length / (y.shape[0] - 1))
    elif (shortest_axis == 2):
        z = np.linspace(input_min[shortest_axis] - eps,
                        input_max[shortest_axis] + eps, resolution)
        length = np.max(z) - np.min(z)
        x = np.arange(input_min[0] - eps, input_max[0] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))
        y = np.arange(input_min[1] - eps, input_max[1] + length / (z.shape[0] - 1) + eps, length / (z.shape[0] - 1))

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()
    return {"grid_points":grid_points,
            "shortest_axis_length":length,
            "xyz":[x,y,z],
            "shortest_axis_index":shortest_axis}


def get_grid_uniform(resolution):
    x = np.linspace(-1.2,1.2, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = to_cuda(torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float))

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.4,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}


def print_args(args):
    print("\n---- experiment configuration ----")
    args_ = vars(args)
    for arg, value in args_.items():
        print(f" * {arg} => {value}")
    print("----------------------------------")


def add_args(parser):
    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [default: output/year-month-date_hour-minute]",
    )
    parser.add_argument("--seed", type=int, default=42, help="set experiment seed")
    parser.add_argument("--n_cores", type=int, default=0, help="set tpu core count")
    parser.add_argument("--dist", action="store_true", help="start distributed training")
    parser.add_argument("--dset", type=str, default="cifar10", help="dataset name")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of workers for dataloading"
    )
    parser.add_argument(
        "--lr", type=float, default=0.005,
    )
    parser.add_argument(
        "--lr_step_mode",
        type=str,
        default="step",
        help="choose lr step mode, choose one of [epoch, step]",
    )
    parser.add_argument(
        "--warmup", type=int, default=0, help="lr warmup in epochs/steps based on epoch step mode"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--lr_sched", type=str, default="step", help="lr scheduler name")
    parser.add_argument(
        "--lr_decay_steps",
        type=str,
        default="100,150",
        help="multi step lr scheduler milestones",
    )
    parser.add_argument(
        "--lr_step_size",
        type=int,
        default=2_000,
        help="multi step lr scheduler milestones",
    )
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="multi step lr scheduler decay gamma"
    )
    parser.add_argument("--resume", action="store_true", help="resume training from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="start wandb logging")
    parser.add_argument("--plot_every", type=int, default=1, help="eval frequency")
    parser.add_argument("--log_every", type=int, default=1, help="logging frequency")

    parser.add_argument("--d_in", type=int, default=3, help="number of epochs")
    parser.add_argument("--dims", type=str, default="512,512,512,512,512,512,512,512", help="number of epochs")
    parser.add_argument("--skip", type = str, default = "4")
    parser.add_argument("--geometric_init", type = bool, default = True)
    parser.add_argument("--radius_init", type = float, default = 1.0)
    parser.add_argument("--beta", type = int, default = 100)

    parser.add_argument("--root", type = str, required = True)
    parser.add_argument("--name", type = str, required = True)
    parser.add_argument("--points_batch", type = int, default = 16_000)
    parser.add_argument("--with_normals", action="store_true")
    parser.add_argument("--save_html", action = "store_true")
    parser.add_argument("--save_ply", action = "store_true")
    parser.add_argument("--plot_verbose", action = "store_true")
    parser.add_argument("--resolution", type = int, default = 512)
    parser.add_argument("--eta", type = float, default = 0.01)
    parser.add_argument("--mu", type = float, default = 10)
    parser.add_argument('--lbda', type = float, default = 10)

    return parser


def setup_device(dist):
    if dist:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda:0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return device, local_rank


def pbar(p=0, msg="", bar_len=20):
    msg = msg.ljust(50)
    block = int(round(bar_len * p))
    text = "\rProgress: [{}] {}% {}".format(
        "\x1b[32m" + "=" * (block - 1) + ">" + "\033[0m" + "-" * (bar_len - block),
        round(p * 100, 2),
        msg,
    )
    print(text, end="")
    if p == 1:
        print()


class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, batch_metrics):
        if self.metrics == {}:
            for key, value in batch_metrics.items():
                self.metrics[key] = [value]
        else:
            for key, value in batch_metrics.items():
                self.metrics[key].append(value)

    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        avg_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])


class RunningMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        attr_keys = list(self.__dict__.keys())
        for key in attr_keys:
            delattr(self, key)
        self.cntr = 0

    def add(self, **kwargs):
        for key, value in kwargs.items():
            attr = getattr(self, key, None)
            if attr is None and self.cntr == 0:
                setattr(self, key, value)
            elif attr is None:
                raise ValueError(f"invalid key: {key}")
            else:
                attr = attr + value
                setattr(self, key, attr)
        self.cntr += 1

    def get(self):
        return {
            key: value.item() / self.cntr for key, value in self.__dict__.items() if key != "cntr"
        }

    def msg(self):
        avg_metrics = {
            key: value.item() / self.cntr for key, value in self.__dict__.items() if key != "cntr"
        }
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])


class TimeMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()
        self.sample_cnt = 0

    def add(self, n_samples=1):
        self.sample_cnt += n_samples

    def get(self):
        return int(self.sample_cnt / (time.time() - self.start))

    def msg(self):
        return f"samples/sec: {self.get()}"
            