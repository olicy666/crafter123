import torch

class FrameData:
    """
    Data structure to hold frame information for warp guidance
    """
    def __init__(self, rgb, coarse_rgb, depth, mask, camera):
        """
        :param rgb: RGB image (B, 3, H, W)
        :param coarse_rgb: Coarse RGB image from point cloud rendering (B, 3, H, W)
        :param depth: Depth map (B, 1, H, W)
        :param mask: Mask indicating valid geometry (B, 1, H, W)
        :param camera: Camera parameters (dict with K, R, t)
        """
        self.rgb = rgb
        self.coarse_rgb = coarse_rgb
        self.depth = depth
        self.mask = mask
        self.camera = camera