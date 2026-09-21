import torch

class FrameData:
    """
    Data structure to hold frame information for geometry evidence routing.
    """
    def __init__(self, rgb, coarse_rgb, depth, mask, camera, depth_std=None, is_observation=False):
        """
        :param rgb: RGB image (B, 3, H, W)
        :param coarse_rgb: Coarse RGB image from point cloud rendering (B, 3, H, W)
        :param depth: Depth map (B, 1, H, W)
        :param mask: Mask indicating valid geometry (B, 1, H, W)
        :param camera: Camera parameters (dict with K, R, t)
        :param depth_std: Optional depth standard deviation, same shape/units as depth.
            This supplied perturbation model is not assumed to be calibrated.
        """
        self.rgb = rgb
        self.coarse_rgb = coarse_rgb
        self.depth = depth
        self.mask = mask
        self.camera = camera
        self.depth_std = depth_std
        # Only original input photographs are observations. Renderings and
        # previous generations must remain false, even if used as conditioning.
        self.is_observation = bool(is_observation)
