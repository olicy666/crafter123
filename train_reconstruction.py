"""Optimize a 3DGS scene from an exported GeoWitness window.

Requires a local graphdeco-inria/gaussian-splatting checkout and its CUDA
extensions. This entry point never downloads dependencies or model weights.
"""
import argparse
import inspect
import json
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

# Import local losses under their package name before loading upstream utils.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crafter123.utils.reconstruction import camera_matrices, reconstruction_loss, seed_points


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--gaussian-repo', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=2000)
    parser.add_argument('--completion-weight', type=float, default=.1)
    parser.add_argument('--variant', choices=('real_only', 'uniform', 'evidence'), default='evidence')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--init-points', type=Path, help='Optional shared NPZ with points/colors [N,3], in bundle world coordinates')
    parser.add_argument('--evaluation', type=Path, help='Optional separate PT with views: [{rgb:[1,3,H,W], camera:{K,R,t}}]')
    return parser


def main():
    args = get_parser().parse_args()
    if args.iterations < 1 or not math.isfinite(args.completion_weight) or not 0 <= args.completion_weight <= 1:
        raise ValueError('iterations must be positive and completion weight must be in [0,1]')
    if not (args.gaussian_repo / 'gaussian_renderer' / '__init__.py').is_file():
        raise ValueError('--gaussian-repo must point to a graphdeco-inria/gaussian-splatting checkout')
    if not torch.cuda.is_available():
        raise RuntimeError('The official 3DGS rasterizer requires CUDA; CPU loss tests are available separately')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'point_cloud.ply').exists():
        raise FileExistsError('Choose an output directory without an existing point_cloud.ply')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    data = torch.load(args.bundle, map_location='cpu', weights_only=True)
    if data.get('schema_version') != 1 or data.get('camera_convention') != 'RDF_world_to_camera':
        raise ValueError('unsupported reconstruction bundle')
    if data.get('trajectory_from_evaluation_images', False):
        raise ValueError('single_view_eval uses evaluation RGB to estimate cameras; export sparse_view_interp with real training images instead')
    observations, targets = data['observations'], data['targets']
    if not observations or not targets:
        raise ValueError('bundle must contain real observations and generated targets')
    sys.path.insert(0, str(args.gaussian_repo.resolve()))
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud
    from utils.loss_utils import ssim
    from arguments import OptimizationParams

    def make_camera(record, name):
        h, w = record['rgb'].shape[-2:]
        view, projection, center = camera_matrices(record['camera'], (h, w), 'cuda')
        k = record['camera']['K'].reshape(3, 3)
        return SimpleNamespace(image_name=name, image_height=h, image_width=w,
            FoVx=2*math.atan(w/(2*float(k[0, 0]))), FoVy=2*math.atan(h/(2*float(k[1, 1]))),
            world_view_transform=view, full_proj_transform=projection, camera_center=center)

    real_cameras = [make_camera(r, f'real_{i}') for i, r in enumerate(observations)]
    target_cameras = [make_camera(r, f'generated_{i}') for i, r in enumerate(targets)]
    if args.init_points:
        with np.load(args.init_points, allow_pickle=False) as initial:
            points, colors = initial['points'], initial['colors']
    else:
        points, colors = seed_points(observations)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape != colors.shape or len(points) < 4:
        raise ValueError('initial points and colors must have matching [N,3] shapes with N >= 4')
    if not np.isfinite(points).all() or not np.isfinite(colors).all() or (colors < 0).any() or (colors > 1).any():
        raise ValueError('initial points must be finite and RGB colors must be in [0,1]')
    pcd = BasicPointCloud(points=points, colors=colors, normals=np.zeros_like(points))
    centers = torch.stack([c.camera_center for c in real_cameras + target_cameras])
    extent = max(float((centers - centers.mean(0)).norm(dim=1).max())*1.1, .01)
    gaussians = GaussianModel(3)
    parameters = inspect.signature(gaussians.create_from_pcd).parameters
    if 'cam_infos' in parameters:
        gaussians.create_from_pcd(pcd, real_cameras + target_cameras, extent)
    else:
        gaussians.create_from_pcd(pcd, extent)
    optimizer_parser = argparse.ArgumentParser()
    optimizer_group = OptimizationParams(optimizer_parser)
    optimizer_args = optimizer_group.extract(optimizer_parser.parse_args([]))
    optimizer_args.iterations = args.iterations
    optimizer_args.opacity_lr = .05  # Original 3DGS value, fixed across variants.
    gaussians.training_setup(optimizer_args)
    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    background = torch.zeros(3, device='cuda')

    def predict(camera):
        return render(camera, gaussians, pipeline, background)['render'].unsqueeze(0)

    def photometric(pred, truth):
        return .8*(pred-truth).abs().mean() + .2*(1-ssim(pred, truth))

    history = []
    for step in range(1, args.iterations+1):
        gaussians.update_learning_rate(step)
        if step % 1000 == 0:
            gaussians.oneupSHdegree()
        real_index, target_index = rng.randrange(len(observations)), rng.randrange(len(targets))
        real_pred = predict(real_cameras[real_index])
        real_rgb = observations[real_index]['rgb'].cuda()
        if args.variant == 'real_only':
            loss = photometric(real_pred, real_rgb)
            terms = {'real': loss.detach()}
        else:
            target = targets[target_index]
            target_pred, target_rgb = predict(target_cameras[target_index]), target['rgb'].cuda()
            if args.variant == 'uniform':
                loss = photometric(real_pred, real_rgb) + photometric(target_pred, target_rgb)
                terms = {'uniform': loss.detach()}
            else:
                loss, terms = reconstruction_loss(real_pred, real_rgb, target_pred, target_rgb,
                    target['references'], data['scales'], ssim, args.completion_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite reconstruction loss at step {step}')
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)
        # Fixed point count matches the short, controlled reconstruction protocol.
        if step == 1 or step % 100 == 0 or step == args.iterations:
            row = dict(step=step, loss=float(loss.detach()), **{k: float(v) for k,v in terms.items()})
            history.append(row)
            print(json.dumps(row), flush=True)
    gaussians.save_ply(str(args.output / 'point_cloud.ply'))
    (args.output / 'training.json').write_text(json.dumps(dict(
        config={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        densification=False, splitting=False, opacity_reset=False, history=history), indent=2))
    if args.evaluation:
        # Load held-out photographs only after fitting; never use them as evidence.
        from PIL import Image
        evaluation = torch.load(args.evaluation, map_location='cpu', weights_only=True)['views']
        metrics = []
        with torch.no_grad():
            for index, record in enumerate(evaluation):
                pred = predict(make_camera(record, f'eval_{index}')).clamp(0, 1)
                truth = record['rgb'].cuda()
                mse = (pred-truth).square().mean().clamp_min(1e-12)
                metrics.append(dict(view=index, psnr=float(-10*mse.log10()), ssim=float(ssim(pred, truth))))
                Image.fromarray((pred[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')).save(args.output/f'eval_{index:04d}.png')
        (args.output/'evaluation.json').write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
