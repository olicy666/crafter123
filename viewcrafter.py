import sys
sys.path.append('./extern/dust3r')
from dust3r.inference import inference, load_model
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
import trimesh
import torch
import numpy as np
import torchvision
import os
import copy
import cv2  
import glob
from PIL import Image
import pytorch3d
from pytorch3d.structures import Pointclouds
from torchvision.utils import save_image
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from utils.pvd_utils import *
from utils.auto_traj_planner import plan_traj_sequences, write_traj_txt
from utils.warp_guidance import WarpGuidanceEngine
from utils.frame_data import FrameData
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from utils.diffusion_utils import instantiate_from_config,load_model_checkpoint,image_guided_synthesis
from pathlib import Path
from torchvision.utils import save_image

class ViewCrafter:
    def __init__(self, opts, gradio = False):
        self.opts = opts
        self.device = opts.device
        self.setup_dust3r()
        self.setup_diffusion()
        self.setup_warp_guidance()
        # initialize ref images, pcd
        if not gradio:
            if os.path.isfile(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_images(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images)
            elif os.path.isdir(self.opts.image_dir):
                self.images, self.img_ori = self.load_initial_dir(image_dir=self.opts.image_dir)
                self.run_dust3r(input_images=self.images, clean_pc = True)    
            else:
                print(f"{self.opts.image_dir} doesn't exist")           
        
    def run_dust3r(self, input_images,clean_pc = False):
        pairs = make_pairs(input_images, scene_graph='complete', prefilter=None, symmetrize=True)
        output = inference(pairs, self.dust3r, self.device, batch_size=self.opts.batch_size)

        mode = GlobalAlignerMode.PointCloudOptimizer #if len(self.images) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=self.device, mode=mode)
        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(init='mst', niter=self.opts.niter, schedule=self.opts.schedule, lr=self.opts.lr)

        if clean_pc:
            self.scene = scene.clean_pointcloud()
        else:
            self.scene = scene

    def render_pcd(self,pts3d,imgs,masks,views,renderer,device,nbv=False):

        imgs = to_numpy(imgs)
        pts3d = to_numpy(pts3d)

        if masks == None:
            pts = torch.from_numpy(np.concatenate([p for p in pts3d])).view(-1, 3).to(device)
            col = torch.from_numpy(np.concatenate([p for p in imgs])).view(-1, 3).to(device)
        else:
            # masks = to_numpy(masks)
            pts = torch.from_numpy(np.concatenate([p[m] for p, m in zip(pts3d, masks)])).to(device)
            col = torch.from_numpy(np.concatenate([p[m] for p, m in zip(imgs, masks)])).to(device)

        point_cloud = Pointclouds(points=[pts], features=[col]).extend(views)
        images = renderer(point_cloud)

        # Generate depth map
        depths = None
        try:
            # Access the rasterizer directly to get depth information
            rasterizer = renderer.rasterizer
            raster_results = rasterizer(point_cloud)

            # Extract depth from rasterization results
            # The depth is stored in raster_results.zbuf
            zbuf = raster_results.zbuf  # Shape: (B, H, W, points_per_pixel)

            # Select the minimum depth for each pixel (closest point)
            depths, _ = torch.min(zbuf, dim=-1)  # Shape: (B, H, W)

            # Convert to (B, 1, H, W) format and mask invalid values
            depths = depths.unsqueeze(1)  # (B, 1, H, W)
            depths = torch.where(torch.isfinite(depths), depths, torch.zeros_like(depths))
        except Exception as e:
            print(f"Error generating depth map: {e}")
            print("Falling back to placeholder depth")
            # Fall back to placeholder if any error occurs
            B, _, H, W = images.shape
            depths = torch.zeros(B, 1, H, W, device=device)

        if nbv:
            color_mask = torch.ones(col.shape).to(device)
            point_cloud_mask = Pointclouds(points=[pts],features=[color_mask]).extend(views)
            view_masks = renderer(point_cloud_mask)
        else:
            view_masks = None

        return images, view_masks, depths
    
    def run_render(self, pcd, imgs,masks, H, W, camera_traj,num_views,nbv=False):
        render_setup = setup_renderer(camera_traj, image_size=(H,W))
        renderer = render_setup['renderer']
        render_results, viewmask, depths = self.render_pcd(pcd, imgs, masks, num_views,renderer,self.device,nbv=nbv)
        return render_results, viewmask, depths

    
    def run_diffusion(self, renderings, frame_list=None):

        prompts = [self.opts.prompt]
        videos = (renderings * 2. - 1.).permute(3,0,1,2).unsqueeze(0).to(self.device)
        condition_index = [0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            # [1,1,c,t,h,w]
            batch_samples = image_guided_synthesis(self.diffusion, prompts, videos, self.noise_shape, self.opts.n_samples, self.opts.ddim_steps, self.opts.ddim_eta, \
                               self.opts.unconditional_guidance_scale, self.opts.cfg_img, self.opts.frame_stride, self.opts.text_input, self.opts.multiple_cond_cfg, self.opts.timestep_spacing, self.opts.guidance_rescale, condition_index, warp_guidance=self.warp_guidance, frame_list=frame_list)

            # save_results_seperate(batch_samples[0], self.opts.save_dir, fps=8)
            # torch.Size([1, 3, 25, 576, 1024]) [-1,1]

        return torch.clamp(batch_samples[0][0].permute(1,2,3,0), -1., 1.) 

    def _prepare_frame_list(self, render_results, depths, camera_traj, viewmask=None, reference_overrides=None):
        if getattr(self, 'warp_guidance', None) is None:
            return None
        if render_results is None or depths is None or camera_traj is None:
            return None

        render_tensor = render_results
        if not torch.is_tensor(render_tensor):
            render_tensor = torch.from_numpy(np.asarray(render_tensor))
        render_tensor = render_tensor.to(self.device).float()
        if render_tensor.dim() != 4 or render_tensor.shape[-1] != 3:
            return None

        depth_tensor = depths
        if not torch.is_tensor(depth_tensor):
            depth_tensor = torch.from_numpy(np.asarray(depth_tensor))
        depth_tensor = depth_tensor.to(self.device).float()

        B, H, W, _ = render_tensor.shape
        if depth_tensor.shape[0] != B or depth_tensor.shape[-2:] != (H, W):
            depth_tensor = F.interpolate(depth_tensor, size=(H, W), mode='bilinear', align_corners=False)

        mask_tensor = None
        if viewmask is not None:
            mask_tensor = viewmask
            if not torch.is_tensor(mask_tensor):
                mask_tensor = torch.from_numpy(np.asarray(mask_tensor))
            if mask_tensor.dim() == 4 and mask_tensor.shape[-1] == 3:
                mask_tensor = mask_tensor.permute(0, 3, 1, 2)
            mask_tensor = mask_tensor.to(self.device).float()
            if mask_tensor.dim() == 3:
                mask_tensor = mask_tensor.unsqueeze(1)
            if mask_tensor.shape[1] != 1:
                mask_tensor = mask_tensor.mean(dim=1, keepdim=True)
            mask_tensor = (mask_tensor > 0.1).float()

        coarse_tensor = render_tensor.permute(0, 3, 1, 2)
        overrides = {}
        if reference_overrides:
            for idx, image in reference_overrides.items():
                overrides[idx] = self._format_reference_image(image, (H, W))

        if hasattr(camera_traj, 'R'):
            num_cams = camera_traj.R.shape[0]
        else:
            num_cams = B
        num_frames = min(B, num_cams)

        frame_list = []
        for idx in range(num_frames):
            coarse_rgb = coarse_tensor[idx:idx+1]
            rgb = overrides.get(idx, coarse_rgb)
            depth = depth_tensor[idx:idx+1]
            if mask_tensor is not None:
                mask = mask_tensor[idx:idx+1]
            else:
                mask = (depth > 0).float()
            camera = self._extract_camera_dict(camera_traj, idx)
            frame_list.append(FrameData(rgb, coarse_rgb, depth, mask, camera))

        return frame_list

    def _format_reference_image(self, image, spatial_size):
        H, W = spatial_size
        if torch.is_tensor(image):
            tensor = image.clone().to(self.device)
        else:
            arr = np.asarray(image)
            tensor = torch.from_numpy(arr).to(self.device)
        tensor = tensor.float()
        if tensor.dim() == 3 and tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = F.interpolate(tensor, size=(H, W), mode='bilinear', align_corners=False)
        tensor = tensor.clamp(0.0, 1.0)
        return tensor

    def _resize_reference_frame(self, image):
        tensor = self._format_reference_image(image, (self.opts.height, self.opts.width))
        return tensor.squeeze(0).permute(1, 2, 0)

    def _extract_camera_dict(self, camera_traj, idx):
        if hasattr(camera_traj, 'R'):
            R = camera_traj.R[idx:idx+1].to(self.device)
        else:
            R = torch.eye(3, device=self.device).unsqueeze(0)
        if hasattr(camera_traj, 'T'):
            t = camera_traj.T[idx:idx+1].to(self.device)
            if t.dim() == 2:
                t = t.unsqueeze(-1)
        else:
            t = torch.zeros(1, 3, 1, device=self.device)
        if hasattr(camera_traj, 'focal_length'):
            focal = camera_traj.focal_length[idx:idx+1].to(self.device)
        else:
            focal = torch.ones(1, 2, device=self.device)
        if hasattr(camera_traj, 'principal_point'):
            principal = camera_traj.principal_point[idx:idx+1].to(self.device)
        else:
            principal = torch.zeros(1, 2, device=self.device)

        K = torch.zeros(1, 3, 3, device=self.device)
        K[:, 0, 0] = focal[:, 0]
        K[:, 1, 1] = focal[:, 1]
        K[:, 0, 2] = principal[:, 0]
        K[:, 1, 2] = principal[:, 1]
        K[:, 2, 2] = 1.0

        return {'K': K, 'R': R, 't': t}

    def nvs_single_view(self, gradio=False):
        # 最后一个view为 0 pose
        c2ws = self.scene.get_im_poses().detach()[1:] 
        principal_points = self.scene.get_principal_points().detach()[1:] #cx cy
        focals = self.scene.get_focals().detach()[1:] 
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[-1][H//2,W//2] #以图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## change coordinate
        c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)

        imgs = np.array(self.scene.imgs)
        reference_image = None
        
        masks = None
        reset_tail_to_ref = False

        if self.opts.mode == 'single_view_nbv':
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 2
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0],num_candidates, self.device)
            _, candidate_viewmask, _ = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, candidate_poses,num_candidates,nbv=True)
            nbv_id = torch.argmin(candidate_viewmask.sum(dim=[1,2,3])).item()
            save_image( candidate_viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask0_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv
        elif self.opts.mode == 'single_view_target':
            camera_traj,num_views = generate_traj_specified(c2ws, H, W, focals, principal_points, self.opts.d_theta[0], self.opts.d_phi[0], self.opts.d_r[0],self.opts.d_x[0]*depth_avg/focals.item(),self.opts.d_y[0]*depth_avg/focals.item(),self.opts.video_length, self.device)
        elif self.opts.mode == 'single_view_autotraj':
            phi, theta, r = plan_traj_sequences(
                c2ws_anchor=c2ws[-1:],
                pcd=pcd[-1],
                imgs=imgs[-1],
                masks=masks,
                H=H,
                W=W,
                focals=focals[-1:],
                principal_points=principal_points[-1:],
                opts=self.opts,
                viewcrafter=self,
            )
            write_traj_txt(os.path.join(self.opts.save_dir, 'traj_auto.txt'), phi, theta, r)
            camera_traj, num_views = generate_traj_txt(
                c2ws[-1:], H, W, focals[-1:], principal_points[-1:], phi, theta, r,
                self.opts.video_length, self.device, viz_traj=True, save_dir=self.opts.save_dir
            )
            reset_tail_to_ref = phi[-1] == 0. and theta[-1] == 0. and r[-1] == 0.
        elif self.opts.mode == 'single_view_txt':
            if not gradio:
                with open(self.opts.traj_txt, 'r') as file:
                    lines = file.readlines()
                    phi = [float(i) for i in lines[0].split()]
                    theta = [float(i) for i in lines[1].split()]
                    r = [float(i) for i in lines[2].split()]
            else: 
                phi, theta, r = self.gradio_traj
            camera_traj,num_views = generate_traj_txt(c2ws, H, W, focals, principal_points, phi, theta, r,self.opts.video_length, self.device,viz_traj=True, save_dir = self.opts.save_dir)
            reset_tail_to_ref = phi[-1]==0. and theta[-1]==0. and r[-1]==0.
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        render_results, viewmask, depths = self.run_render([pcd[-1]], [imgs[-1]],masks, H, W, camera_traj,num_views)
        # Render results shape: (B, H, W, 3)
        # Depths shape: (B, 1, H, W)
        # Note: B = num_views for rendering

        target_size = (self.opts.height, self.opts.width)
        # Interpolate render results to desired size
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=target_size, mode='bilinear', align_corners=False).permute(0,2,3,1)

        # Resize depths to match render_results size
        depths = F.interpolate(depths, size=target_size, mode='bilinear', align_corners=False)
        render_results[0] = self._resize_reference_frame(self.img_ori)
        if reset_tail_to_ref:
            render_results[-1] = self._resize_reference_frame(self.img_ori)
                
        save_video(render_results, os.path.join(self.opts.save_dir, 'render0.mp4'))
        save_pointcloud_with_normals([imgs[-1]], [pcd[-1]], msk=None, save_path=os.path.join(self.opts.save_dir,'pcd0.ply') , mask_pc=False, reduce_pc=False)

        ref_image = self.img_ori[0] if isinstance(self.img_ori, list) else self.img_ori
        reference_overrides = {0: ref_image}
        frame_list = self._prepare_frame_list(render_results, depths, camera_traj, viewmask=viewmask, reference_overrides=reference_overrides)

        diffusion_results = self.run_diffusion(render_results, frame_list=frame_list)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, 'diffusion0.mp4'))

        return diffusion_results

    def nvs_sparse_view(self,iter):

        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]
        depth_avg = depth[0][H//2,W//2] #以ref图像中心处的depth(z)为球心旋转
        radius = depth_avg*self.opts.center_scale #缩放调整

        ## masks for cleaner point cloud
        self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
        masks = self.scene.get_masks()
        depth = self.scene.get_depthmaps()
        bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
        masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
        masks = to_numpy(masks_new)

        ## render, 从c2ws[0]即ref image对应的相机开始
        imgs = np.array(self.scene.imgs)

        if self.opts.mode == 'single_view_ref_iterative':
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=0, r=radius, elevation=self.opts.elevation, device=self.device)
            camera_traj,num_views = generate_traj_specified(c2ws[0:1], H, W, focals[0:1], principal_points[0:1], self.opts.d_theta[iter], self.opts.d_phi[iter], self.opts.d_r[iter],self.opts.video_length, self.device)
            render_results, viewmask, depths = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=target_size, mode='bilinear', align_corners=False).permute(0,2,3,1)
            render_results[0] = self._resize_reference_frame(self.img_ori)
            reference_image = self.img_ori
        elif self.opts.mode == 'single_view_1drc_iterative':
            self.opts.elevation -= self.opts.d_theta[iter-1]
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            camera_traj,num_views = generate_traj_specified(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], self.opts.d_theta[iter], self.opts.d_phi[iter], self.opts.d_r[iter],self.opts.video_length, self.device)
            render_results, viewmask, depths = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=target_size, mode='bilinear', align_corners=False).permute(0,2,3,1)
            ref_img = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.
            render_results[0] = self._resize_reference_frame(ref_img)
            reference_image = ref_img
        elif self.opts.mode == 'single_view_nbv':
            c2ws,pcd =  world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=-1, r=radius, elevation=self.opts.elevation, device=self.device)
            ## 输入candidate->渲染mask->最大mask对应的pose作为nbv
            ## nbv模式下self.opts.d_theta[0], self.opts.d_phi[0]代表search space中的网格theta, phi之间的间距; self.opts.d_phi[0]的符号代表方向,分为左右两个方向
            ## FIXME hard coded candidate view数量, 以left为例,第一次迭代从[左,左上]中选取, 从第二次开始可以从[左,左上,左下]中选取
            num_candidates = 3
            candidate_poses,thetas,phis = generate_candidate_poses(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], self.opts.d_theta[0], self.opts.d_phi[0], num_candidates, self.device)
            _, viewmask, _ = self.run_render(pcd, imgs,masks, H, W, candidate_poses,num_candidates,nbv=True)
            nbv_id = torch.argmin(viewmask.sum(dim=[1,2,3])).item()
            save_image(viewmask.permute(0,3,1,2), os.path.join(self.opts.save_dir,f"candidate_mask{iter}_nbv{nbv_id}.png"), normalize=True, value_range=(0, 1))
            theta_nbv = thetas[nbv_id]
            phi_nbv = phis[nbv_id]   
            # generate camera trajectory from T_curr to T_nbv
            camera_traj,num_views = generate_traj_specified(c2ws[-1:], H, W, focals[-1:], principal_points[-1:], theta_nbv, phi_nbv, self.opts.d_r[0],self.opts.video_length, self.device)
            # 重置elevation
            self.opts.elevation -= theta_nbv    
            render_results, viewmask, depths = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
            render_results = F.interpolate(render_results.permute(0,3,1,2), size=target_size, mode='bilinear', align_corners=False).permute(0,2,3,1)
            ref_img = (self.images[-1]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.
            render_results[0] = self._resize_reference_frame(ref_img)
            reference_image = ref_img
        else:
            raise KeyError(f"Invalid Mode: {self.opts.mode}")

        depths = F.interpolate(depths, size=target_size, mode='bilinear', align_corners=False)
        reference_overrides = {0: reference_image} if reference_image is not None else None
        frame_list = self._prepare_frame_list(render_results, depths, camera_traj, viewmask=viewmask, reference_overrides=reference_overrides)

        save_video(render_results, os.path.join(self.opts.save_dir, f'render{iter}.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd{iter}.ply') , mask_pc=True, reduce_pc=False)
        diffusion_results = self.run_diffusion(render_results, frame_list=frame_list)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion{iter}.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results
    
    def nvs_sparse_view_interp(self):

        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        depth = [i.detach() for i in self.scene.get_depthmaps()]

        if len(self.images) == 2:
            masks = None
            mask_pc = False
        else:
            ## masks for cleaner point cloud
            self.scene.min_conf_thr = float(self.scene.conf_trf(torch.tensor(self.opts.min_conf_thr)))
            masks = self.scene.get_masks()
            depth = self.scene.get_depthmaps()
            bgs_mask = [dpt > self.opts.bg_trd*(torch.max(dpt[40:-40,:])+torch.min(dpt[40:-40,:])) for dpt in depth]
            masks_new = [m+mb for m, mb in zip(masks,bgs_mask)] 
            masks = to_numpy(masks_new)
            mask_pc = True

        imgs = np.array(self.scene.imgs)

        camera_traj,num_views = generate_traj_interp(c2ws, H, W, focals, principal_points, self.opts.video_length, self.device)
        render_results, viewmask, depths = self.run_render(pcd, imgs,masks, H, W, camera_traj,num_views)
        target_size = (self.opts.height, self.opts.width)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=target_size, mode='bilinear', align_corners=False).permute(0,2,3,1)
        depths = F.interpolate(depths, size=target_size, mode='bilinear', align_corners=False)
        
        reference_overrides = {}
        for i in range(len(self.img_ori)):
            frame_idx = i*(self.opts.video_length - 1)
            render_results[frame_idx] = self._resize_reference_frame(self.img_ori[i])
            reference_overrides[frame_idx] = self.img_ori[i]
        frame_list_full = self._prepare_frame_list(render_results, depths, camera_traj, viewmask=viewmask, reference_overrides=reference_overrides)
        save_video(render_results, os.path.join(self.opts.save_dir, f'render.mp4'))
        save_pointcloud_with_normals(imgs, pcd, msk=masks, save_path=os.path.join(self.opts.save_dir, f'pcd.ply') , mask_pc=mask_pc, reduce_pc=False)

        diffusion_results = []
        print(f'Generating {len(self.img_ori)-1} clips\n')
        for i in range(len(self.img_ori)-1 ):
            print(f'Generating clip {i} ...\n')
            start = i*(self.opts.video_length - 1)
            end = self.opts.video_length + start
            clip_frames = frame_list_full[start:end] if frame_list_full is not None else None
            diffusion_results.append(self.run_diffusion(render_results[start:end], frame_list=clip_frames))
        print(f'Finish!\n')
        diffusion_results = torch.cat(diffusion_results)
        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_single_view_eval(self):

        # get camera trajectory of the input frames
        c2ws = self.scene.get_im_poses().detach()
        principal_points = self.scene.get_principal_points().detach()
        focals = self.scene.get_focals().detach()
        shape = self.images[0]['true_shape']
        H, W = int(shape[0][0]), int(shape[0][1])
        pcd = [i.detach() for i in self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)] # a list of points of size whc
        c2ws,pcd =  world_point_to_kth(poses=c2ws, points=torch.stack(pcd), k=0, device=self.device)
        camera_traj,num_views = generate_traj(c2ws, H, W, focals, principal_points, self.device)
        
        # estimate pcd again using only one ref image
        images_ref = [self.images[0], copy.deepcopy(self.images[0])]
        images_ref[1]['idx'] = 1
        self.run_dust3r(input_images=images_ref)
        pcd_ref = self.scene.get_pts3d(clip_thred=self.opts.dpt_trd)[0].detach()
        img_ref = np.array(self.scene.imgs)[0]
        masks = None

        render_results, viewmask, depths = self.run_render([pcd_ref], [img_ref],masks, H, W, camera_traj,num_views)
        target_size = (self.opts.height, self.opts.width)
        render_results = F.interpolate(render_results.permute(0,3,1,2), size=target_size, mode='bilinear', align_corners=False).permute(0,2,3,1)
        depths = F.interpolate(depths, size=target_size, mode='bilinear', align_corners=False)
        render_results[0] = self._resize_reference_frame(self.img_ori[0])
        reference_overrides = {0: self.img_ori[0]}
        frame_list = self._prepare_frame_list(render_results, depths, camera_traj, viewmask=viewmask, reference_overrides=reference_overrides)
        save_video(render_results, os.path.join(self.opts.save_dir, f'render_ref0.mp4'))
        diffusion_results = self.run_diffusion(render_results, frame_list=frame_list)

        save_video((diffusion_results + 1.0) / 2.0, os.path.join(self.opts.save_dir, f'diffusion_ref0.mp4'))
        # torch.Size([25, 576, 1024, 3])
        return diffusion_results

    def nvs_single_view_ref_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, len(self.opts.d_phi)):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def nvs_single_view_1drc_iterative(self):

        all_results = []
        sample_rate = 6
        idx = 1 #初始包含1张ref image
        for itr in range(0, len(self.opts.d_phi)):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def nvs_single_view_nbv(self):
        # lef and right
        # d_theta and a_phi 是搜索空间的顶点间隔
        all_results = []
        ## FIXME: hard coded
        sample_rate = 6
        max_itr = 3

        idx = 1 #初始包含1张ref image
        for itr in range(0, max_itr):
            if itr == 0:
                self.images = [self.images[0]] #去掉后一份copy
                diffusion_results_itr = self.nvs_single_view()
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
            else:
                for i in range(0+sample_rate, diffusion_results_itr.shape[0], sample_rate):
                    self.images.append(get_input_dict(diffusion_results_itr[i:i+1,...],idx,dtype = torch.float32))
                    idx += 1
                self.run_dust3r(input_images=self.images, clean_pc=True)
                diffusion_results_itr = self.nvs_sparse_view(itr)
                # diffusion_results_itr = torch.randn([25, 576, 1024, 3]).to(self.device)
                diffusion_results_itr = diffusion_results_itr.permute(0,3,1,2)
                all_results.append(diffusion_results_itr)
        return all_results

    def setup_diffusion(self):
        seed_everything(self.opts.seed)

        config = OmegaConf.load(self.opts.config)
        model_config = config.pop("model", OmegaConf.create())

        ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
        model_config['params']['unet_config']['params']['use_checkpoint'] = False
        model = instantiate_from_config(model_config)
        model = model.to(self.device)
        model.cond_stage_model.device = self.device
        model.perframe_ae = self.opts.perframe_ae
        assert os.path.exists(self.opts.ckpt_path), "Error: checkpoint Not Found!"
        model = load_model_checkpoint(model, self.opts.ckpt_path)
        model.eval()
        self.diffusion = model

    def setup_warp_guidance(self):
        # Initialize the warp guidance engine with WAVE-style frequency domain mixing
        self.warp_guidance = WarpGuidanceEngine(
            vae_encoder=self.diffusion.first_stage_model,
            device=self.device,
            # WAVE-style parameters for stable background
            use_freq_mix=getattr(self.opts, 'use_freq_mix', True),  # Enable FFT frequency mixing
            filter_type=getattr(self.opts, 'filter_type', 'gaussian'),  # 'gaussian' or 'ideal'
            freq_cutoff=getattr(self.opts, 'freq_cutoff', 0.25),  # Frequency cutoff (0.0-1.0)
            low_freq_norm=getattr(self.opts, 'low_freq_norm', True),  # Normalize low-freq (key!)
            noise_level=getattr(self.opts, 'noise_level', 999),  # Noise level for q_sample
        )

        h, w = self.opts.height // 8, self.opts.width // 8
        channels = self.diffusion.model.diffusion_model.out_channels
        n_frames = self.opts.video_length
        self.noise_shape = [self.opts.bs, channels, n_frames, h, w]

    def setup_dust3r(self):
        self.dust3r = load_model(self.opts.model_path, self.device)
    
    def load_initial_images(self, image_dir):
        ## load images
        ## dict_keys(['img', 'true_shape', 'idx', 'instance', 'img_ori']),张量形式
        images = load_images([image_dir], size=512,force_1024 = True)
        img_ori = (images[0]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2. # [576,1024,3] [0,1]

        if len(images) == 1:
            images = [images[0], copy.deepcopy(images[0])]
            images[1]['idx'] = 1

        return images, img_ori

    def load_initial_dir(self, image_dir):

        image_files = glob.glob(os.path.join(image_dir, "*"))

        if len(image_files) < 2:
            raise ValueError("Input views should not less than 2.")
        image_files = sorted(image_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        images = load_images(image_files, size=512,force_1024 = True)

        img_gts = []
        for i in range(len(image_files)):
            img_gts.append((images[i]['img_ori'].squeeze(0).permute(1,2,0)+1.)/2.) 

        return images, img_gts

    def run_gradio(self,i2v_input_image, i2v_elevation, i2v_center_scale, i2v_d_phi, i2v_d_theta, i2v_d_r, i2v_steps, i2v_seed):
        self.opts.elevation = float(i2v_elevation)
        self.opts.center_scale = float(i2v_center_scale)
        self.opts.ddim_steps = i2v_steps
        # Add input validation with default values
        phi_values = [float(i) for i in i2v_d_phi.split()] if i2v_d_phi.strip() else [0.0]
        theta_values = [float(i) for i in i2v_d_theta.split()] if i2v_d_theta.strip() else [0.0]
        r_values = [float(i) for i in i2v_d_r.split()] if i2v_d_r.strip() else [1.0]
        self.gradio_traj = phi_values, theta_values, r_values
        seed_everything(i2v_seed)
        torch.cuda.empty_cache()
        img_tensor = torch.from_numpy(i2v_input_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        img_tensor = (img_tensor / 255. - 0.5) * 2

        image_tensor_resized = center_crop_image(img_tensor) #1,3,h,w
        images = get_input_dict(image_tensor_resized,idx = 0,dtype = torch.float32)
        images = [images, copy.deepcopy(images)]
        images[1]['idx'] = 1
        self.images = images
        self.img_ori = (image_tensor_resized.squeeze(0).permute(1,2,0) + 1.)/2.

        # self.images: torch.Size([1, 3, 288, 512]), [-1,1]
        # self.img_ori:  torch.Size([576, 1024, 3]), [0,1]
        # self.images, self.img_ori = self.load_initial_images(image_dir=i2v_input_image)
        self.run_dust3r(input_images=self.images)
        self.nvs_single_view(gradio=True)

        traj_dir = os.path.join(self.opts.save_dir, "viz_traj.mp4")
        gen_dir = os.path.join(self.opts.save_dir, "diffusion0.mp4")
        
        return traj_dir, gen_dir
