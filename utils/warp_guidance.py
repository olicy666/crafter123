import torch
import torch.nn.functional as F
import torch.fft as fft
from utils.evidence_routing import compute_geometry_evidence
from utils.uncertainty_guidance import (
    validate_scale_config, projection_covariance, latent_sigma, scale_residual,
)
try:
    from ViewCrafter123.lvdm.distributions import DiagonalGaussianDistribution
except ModuleNotFoundError:
    from lvdm.distributions import DiagonalGaussianDistribution


# ============== Frequency-Preserving Latent Functions ==============

def nearest_valid_depth(zbuf):
    """Extract the nearest positive depth from a PyTorch3D z-buffer.

    PyTorch3D uses negative sentinel values for empty points-per-pixel slots.
    Taking a raw minimum would therefore select an empty slot whenever a
    pixel contains both a valid point and an empty slot.  Such a depth map
    would erase the support used by the evidence verifier.

    Args:
        zbuf: Tensor shaped ``[B, H, W, P]`` as returned by a point
            rasterizer.

    Returns:
        Tensor shaped ``[B, H, W]`` with zero at pixels that have no valid
        positive depth.
    """
    if not torch.is_tensor(zbuf) or zbuf.ndim != 4:
        raise ValueError(
            "zbuf must be a tensor with shape [B, H, W, points_per_pixel], "
            f"got {type(zbuf).__name__} with shape "
            f"{getattr(zbuf, 'shape', None)}"
        )
    valid = torch.isfinite(zbuf) & (zbuf > 0)
    positive_zbuf = torch.where(
        valid, zbuf, torch.full_like(zbuf, float("inf"))
    )
    nearest = positive_zbuf.amin(dim=-1)
    return torch.where(torch.isfinite(nearest), nearest, torch.zeros_like(nearest))

def get_freq_filter(shape, device, filter_type='gaussian', d_s=0.25):
    """
    Generate a low-pass filter for frequency domain operations.
    The filter is used only to preserve a coarse spatial component of the
    initial latent; it is not a view or geometry representation.
    
    Args:
        shape: Shape of the latent (B, C, H, W) or (B, C, T, H, W)
        device: Device to use
        filter_type: Type of filter ('gaussian', 'ideal', 'butterworth')
        d_s: Normalized stop frequency for spatial dimensions (0.0-1.0)
    
    Returns:
        Low-pass filter tensor
    """
    if filter_type == "gaussian":
        return gaussian_low_pass_filter(shape=shape, d_s=d_s).to(device)
    elif filter_type == "ideal":
        return ideal_low_pass_filter(shape=shape, d_s=d_s).to(device)
    else:
        return gaussian_low_pass_filter(shape=shape, d_s=d_s).to(device)


def gaussian_low_pass_filter(shape, d_s=0.25):
    """
    Create a Gaussian low-pass filter.
    
    Args:
        shape: (B, C, H, W) or (B, C, T, H, W)
        d_s: Normalized cutoff frequency (0.0-1.0), smaller = more blur
    """
    if len(shape) == 5:
        # Video: (B, C, T, H, W)
        T, H, W = shape[2], shape[3], shape[4]
    else:
        # Image: (B, C, H, W)
        T, H, W = 1, shape[2], shape[3]
    
    # Create frequency coordinates
    y_freq = torch.fft.fftfreq(H).reshape(-1, 1)
    x_freq = torch.fft.fftfreq(W).reshape(1, -1)
    
    # Compute distance from center in frequency domain
    freq_dist = torch.sqrt(y_freq ** 2 + x_freq ** 2)
    
    # Gaussian filter.  A zero cutoff would otherwise create NaNs at the
    # DC component and silently corrupt the initialized latent.
    sigma = max(float(d_s), 1e-6)
    LPF = torch.exp(-freq_dist ** 2 / (2 * sigma ** 2))
    
    # Shift to match fftshift
    LPF = torch.fft.fftshift(LPF)
    
    if len(shape) == 5:
        # Expand for video
        LPF = LPF.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)
        LPF = LPF.expand(shape[0], shape[1], T, -1, -1)
    else:
        LPF = LPF.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        LPF = LPF.expand(shape[0], shape[1], -1, -1)
    
    return LPF


def ideal_low_pass_filter(shape, d_s=0.25):
    """
    Create an ideal (hard cutoff) low-pass filter.
    """
    if len(shape) == 5:
        T, H, W = shape[2], shape[3], shape[4]
    else:
        T, H, W = 1, shape[2], shape[3]
    
    y_freq = torch.fft.fftfreq(H).reshape(-1, 1)
    x_freq = torch.fft.fftfreq(W).reshape(1, -1)
    freq_dist = torch.sqrt(y_freq ** 2 + x_freq ** 2)
    
    LPF = (freq_dist <= max(float(d_s), 0.0)).float()
    LPF = torch.fft.fftshift(LPF)
    
    if len(shape) == 5:
        LPF = LPF.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        LPF = LPF.expand(shape[0], shape[1], T, -1, -1)
    else:
        LPF = LPF.unsqueeze(0).unsqueeze(0)
        LPF = LPF.expand(shape[0], shape[1], -1, -1)
    
    return LPF


def freq_mix_2d(x, noise, LPF, low_freq_norm=True, norm_factor=1.0):
    """
    Frequency-domain mixing for a single spatial latent.
    
    Args:
        x: Diffused latent from warped image (B, C, H, W)
        noise: Randomly sampled Gaussian noise (B, C, H, W)
        LPF: Low-pass filter
        low_freq_norm: Whether to normalize low-frequency components
        norm_factor: Normalization factor for low-frequency
    
    Returns:
        Mixed latent with low-freq from x and high-freq from noise
    """
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    noise_freq = fft.fftn(noise, dim=(-2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-2, -1))
    
    # Frequency separation
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    
    # Normalize low-frequency components (key for stability!)
    if low_freq_norm:
        low_freq_magnitude = torch.abs(x_freq_low)
        mean_mag = low_freq_magnitude.mean()
        if mean_mag > 1e-8:
            x_freq_low = (x_freq_low / mean_mag) * norm_factor
    
    # Mix in frequency domain
    x_freq_mixed = x_freq_low + noise_freq_high
    
    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-2, -1)).real
    
    return x_mixed


def freq_mix_3d(x, noise, LPF, low_freq_norm=True, norm_factor=1.0):
    """
    Frequency-domain mixing for a video latent.
    
    Args:
        x: Diffused latent from warped frames (B, C, T, H, W)
        noise: Randomly sampled Gaussian noise (B, C, T, H, W)
        LPF: Low-pass filter
        low_freq_norm: Whether to normalize low-frequency components
        norm_factor: Normalization factor
    
    Returns:
        Mixed latent
    """
    # FFT on spatial dimensions only (not temporal)
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    noise_freq = fft.fftn(noise, dim=(-2, -1))
    noise_freq = fft.fftshift(noise_freq, dim=(-2, -1))
    
    # Frequency separation
    HPF = 1 - LPF
    x_freq_low = x_freq * LPF
    noise_freq_high = noise_freq * HPF
    
    # Normalize low-frequency
    if low_freq_norm:
        low_freq_magnitude = torch.abs(x_freq_low)
        mean_mag = low_freq_magnitude.mean()
        if mean_mag > 1e-8:
            x_freq_low = (x_freq_low / mean_mag) * norm_factor
    
    # Mix
    x_freq_mixed = x_freq_low + noise_freq_high
    
    # IFFT
    x_freq_mixed = fft.ifftshift(x_freq_mixed, dim=(-2, -1))
    x_mixed = fft.ifftn(x_freq_mixed, dim=(-2, -1)).real
    
    return x_mixed

class EvidenceRoutingEngine:
    """
    Inference-time evidence router for camera-indexed novel-view diffusion.

    The engine projects existing scaffold observations, verifies their local
    target-camera depth agreement, and exposes the resulting state to the
    latent initialization and temporal attention paths. It never rewrites the
    input scaffold.
    """

    def __init__(self, vae_encoder=None, device='cuda', num_recent=8, num_ref=3, 
                 overlap_threshold=0.1, large_negative=-1e4,
                 # Frequency-preserving latent parameters
                 use_freq_mix=True,          # Enable FFT frequency mixing
                 filter_type='gaussian',      # 'gaussian' or 'ideal'
                 freq_cutoff=0.25,           # Normalized frequency cutoff (d_s)
                 low_freq_norm=True,         # Normalize low-frequency components
                 noise_level=999,            # Noise level for q_sample
                 shared_noise_seed=None,      # Shared seed for consistent noise across frames
                 # Geometry-verified evidence routing parameters
                 use_evidence_routing=True,
                 depth_rel_tolerance=0.08,
                 depth_abs_tolerance=0.02,
                 conflict_threshold=0.08,
                 min_support_ratio=0.02,
                 admit_ratio_threshold=0.20,
                 max_conflict_ratio=0.15,
                 attenuated_weight=0.35,
                 abstain_gate=0.02,
                 geo_early_scale=1.0,
                 geo_late_scale=0.35,
                 use_fmi=True,
                 use_cga=True,
                 init_scale_mode='legacy',
                 init_depth_rel_std=0.05,
                 init_fixed_sigma=1.0,
                 init_scales=(0., 0.5, 1., 2., 4., 8.)):
        """
        Initialize the geometry evidence routing engine.
        :param vae_encoder: VAE encoder for latent space operations
        :param device: Device to use for computations
        :param use_freq_mix: Preserve coarse layout with frequency-domain mixing
        :param filter_type: Type of low-pass filter ('gaussian' or 'ideal')
        :param freq_cutoff: Cutoff frequency for the filter (0.0-1.0, smaller = more blur)
        :param low_freq_norm: Normalize low-frequency components (key for stability)
        :param noise_level: Noise level for the forward diffusion initialization
        :param shared_noise_seed: Seed for shared noise generation across frames
        """
        self.vae_encoder = vae_encoder
        self.device = device
        self.num_recent = num_recent
        self.num_ref = num_ref
        self.overlap_threshold = overlap_threshold
        self.large_negative = large_negative
        
        # Frequency-preserving latent settings
        self.use_freq_mix = use_freq_mix
        self.filter_type = filter_type
        self.freq_cutoff = freq_cutoff
        self.low_freq_norm = low_freq_norm
        self.noise_level = noise_level
        self.shared_noise_seed = shared_noise_seed

        # The scaffold remains unchanged.  These values only determine how
        # its projected evidence is admitted into the diffusion trajectory.
        self.use_evidence_routing = use_evidence_routing
        self.depth_rel_tolerance = depth_rel_tolerance
        self.depth_abs_tolerance = depth_abs_tolerance
        self.conflict_threshold = conflict_threshold
        self.min_support_ratio = min_support_ratio
        self.admit_ratio_threshold = admit_ratio_threshold
        self.max_conflict_ratio = max_conflict_ratio
        self.attenuated_weight = attenuated_weight
        self.abstain_gate = abstain_gate
        self.geo_early_scale = geo_early_scale
        self.geo_late_scale = geo_late_scale
        self.use_fmi = use_fmi
        self.use_cga = use_cga
        validate_scale_config(init_scale_mode, init_depth_rel_std, init_fixed_sigma, init_scales)
        self.init_scale_mode = init_scale_mode
        self.init_depth_rel_std = float(init_depth_rel_std)
        self.init_fixed_sigma = float(init_fixed_sigma)
        self.init_scales = tuple(float(s) for s in init_scales)
        
        # Cache for frequency filter (created on first use)
        self._freq_filter_cache = {}
        
        # Diffusion model reference (for q_sample)
        self.diffusion_model = None

    @property
    def vae(self):
        # Backward compatibility for old attribute name
        return self.vae_encoder

    @vae.setter
    def vae(self, value):
        self.vae_encoder = value
    
    def set_diffusion_model(self, diffusion_model):
        """Set reference to diffusion model for q_sample operations."""
        self.diffusion_model = diffusion_model
    
    def get_freq_filter(self, shape):
        """Get or create frequency filter for the given shape."""
        shape_key = tuple(shape)
        if shape_key not in self._freq_filter_cache:
            self._freq_filter_cache[shape_key] = get_freq_filter(
                shape, self.device, self.filter_type, self.freq_cutoff
            )
        return self._freq_filter_cache[shape_key]
    
    def get_shared_noise(self, shape, seed=None):
        """Generate shared noise with consistent seed across frames."""
        if seed is None:
            seed = self.shared_noise_seed if self.shared_noise_seed is not None else 42
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(shape, generator=generator, device=self.device)

    def _as_rgb(self, value, name='rgb'):
        """Normalize an RGB tensor to ``[B, 3, H, W]`` on the engine device."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=self.device, dtype=torch.float32)
        if value.ndim == 3:
            if value.shape[0] == 3:
                value = value.unsqueeze(0)
            elif value.shape[-1] == 3:
                value = value.permute(2, 0, 1).unsqueeze(0)
            else:
                raise ValueError(
                    f"{name} must have three channels in the first or last "
                    f"dimension, got {tuple(value.shape)}"
                )
        elif value.ndim == 4 and value.shape[-1] == 3 and value.shape[1] != 3:
            value = value.permute(0, 3, 1, 2)
        if value.ndim != 4 or value.shape[1] != 3:
            raise ValueError(
                f"{name} must have shape [B, 3, H, W] or [B, H, W, 3], "
                f"got {tuple(value.shape)}"
            )
        return value

    def _as_map(self, value, name='map'):
        """Normalize a single-channel map to ``[B, 1, H, W]``."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=self.device, dtype=torch.float32)
        if value.ndim == 2:
            value = value.unsqueeze(0).unsqueeze(0)
        elif value.ndim == 3:
            value = value.unsqueeze(1)
        elif value.ndim == 4 and value.shape[-1] == 1 and value.shape[1] != 1:
            value = value.permute(0, 3, 1, 2)
        if value.ndim != 4 or value.shape[1] != 1:
            raise ValueError(
                f"{name} must have shape [B, 1, H, W] or [B, H, W], "
                f"got {tuple(value.shape)}"
            )
        return value

    def _camera_tensor(self, value, batch_size, name, trailing_shape):
        """Normalize camera matrices/vectors to ``[B, ...]`` tensors."""
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        value = value.to(device=self.device, dtype=torch.float32)

        if value.ndim == 2 and tuple(value.shape) == tuple(trailing_shape):
            value = value.unsqueeze(0)
        elif trailing_shape == (3, 1) and value.ndim == 1 and value.shape[0] == 3:
            value = value.reshape(1, 3, 1)
        elif trailing_shape == (3, 1) and value.ndim == 2 and value.shape[-1] == 3:
            value = value.unsqueeze(-1)

        if value.ndim != 3 or tuple(value.shape[1:]) != tuple(trailing_shape):
            raise ValueError(
                f"Camera parameter {name} must have trailing shape "
                f"{tuple(trailing_shape)}, got {tuple(value.shape)}"
            )
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1, -1)
        if value.shape[0] != batch_size:
            raise ValueError(
                f"Camera parameter {name} has batch {value.shape[0]}, "
                f"expected {batch_size}"
            )
        return value

    @staticmethod
    def _pytorch3d_to_rdf(R, t):
        """Convert PyTorch3D camera extrinsics to RDF column-vector form.

        PyTorch3D stores a row-vector world-to-view rotation in the left/up/
        forward camera convention.  The warp implementation uses the usual
        column-vector right/down/forward pinhole convention.  Keeping this
        conversion at the camera boundary prevents the verifier from mixing
        coordinate conventions with its depth tests.
        """
        axis_flip = torch.diag(
            torch.tensor([-1.0, -1.0, 1.0], device=R.device, dtype=R.dtype)
        ).unsqueeze(0).expand(R.shape[0], -1, -1)
        return torch.bmm(axis_flip, R.transpose(1, 2)), torch.bmm(axis_flip, t)

    def _pytorch3d_intrinsics(self, camera, batch_size):
        """Read pixel intrinsics from a PyTorch3D camera object.

        ``PerspectiveCameras`` commonly stores ``K=None`` when it was
        constructed from ``focal_length`` and ``principal_point``.  Falling
        back to an identity matrix in that case silently destroys the
        projection geometry, so reconstruct the 3x3 matrix from the explicit
        pixel parameters instead.
        """
        K = getattr(camera, "K", None)
        if K is not None:
            K = torch.as_tensor(K, device=self.device, dtype=torch.float32)
            if K.ndim == 2:
                K = K.unsqueeze(0)
            if K.ndim == 3 and K.shape[-2:] == (4, 4):
                K = K[:, :3, :3]
            return self._camera_tensor(K, batch_size, "PyTorch3D K", (3, 3))

        def _vector(value, width, name, default):
            if value is None:
                return torch.full(
                    (batch_size, width),
                    float(default),
                    device=self.device,
                    dtype=torch.float32,
                )
            value = torch.as_tensor(value, device=self.device, dtype=torch.float32)
            if value.ndim == 0:
                value = value.reshape(1, 1)
            elif value.ndim == 1:
                if value.numel() == width:
                    value = value.reshape(1, width)
                elif value.numel() == 1:
                    value = value.reshape(1, 1)
                elif value.numel() == batch_size * width:
                    value = value.reshape(batch_size, width)
                elif value.numel() == batch_size:
                    # Some PyTorch3D versions expose a scalar focal length
                    # per camera as [B].  Treat it as an isotropic focal
                    # length rather than mistaking it for one [fx, fy] pair.
                    value = value.reshape(batch_size, 1)
                else:
                    raise ValueError(
                        f"PyTorch3D camera parameter {name} has shape "
                        f"{tuple(value.shape)}, expected a scalar, {width}, "
                        f"or one value per batch item"
                    )
            elif value.ndim != 2 or value.shape[-1] not in (1, width):
                raise ValueError(
                    f"PyTorch3D camera parameter {name} must have shape "
                    f"[B, {width}], [B, 1], [{width}], or a scalar, "
                    f"got {tuple(value.shape)}"
                )
            if value.shape[0] == 1 and batch_size > 1:
                value = value.expand(batch_size, -1)
            if value.shape[-1] == 1:
                value = value.expand(-1, width)
            if value.shape[0] != batch_size:
                raise ValueError(
                    f"PyTorch3D camera parameter {name} has batch "
                    f"{value.shape[0]}, expected {batch_size}"
                )
            return value

        focal = _vector(
            getattr(camera, "focal_length", None), 2, "focal_length", 1.0
        )
        principal = _vector(
            getattr(camera, "principal_point", None), 2, "principal_point", 0.0
        )
        K = torch.zeros(batch_size, 3, 3, device=self.device, dtype=torch.float32)
        K[:, 0, 0] = focal[:, 0]
        K[:, 1, 1] = focal[:, 1]
        K[:, 0, 2] = principal[:, 0]
        K[:, 1, 2] = principal[:, 1]
        K[:, 2, 2] = 1.0
        return K

    def warp_rgb_depth(self, source_rgb, source_depth, cam_src, cam_tgt, H_tgt, W_tgt,
                       source_depth_std=None, return_uncertainty=False):
        """
        Warp RGB and depth from source view to target view using camera parameters.

        :param source_rgb: Source RGB image (B, 3, H_s, W_s)
        :param source_depth: Source depth map (B, 1, H_s, W_s)
        :param cam_src: Source camera (dict with K, R, t)
        :param cam_tgt: Target camera (dict with K, R, t)
        :param H_tgt: Target height
        :param W_tgt: Target width
        :return: (warped_rgb, warped_depth, warped_mask) - all (B, C, H_tgt, W_tgt)
        """
        source_rgb = self._as_rgb(source_rgb, 'source_rgb')
        source_depth = self._as_map(source_depth, 'source_depth')
        B, C, H_s, W_s = source_rgb.shape
        if source_depth.shape[0] != B or source_depth.shape[-2:] != (H_s, W_s):
            raise ValueError(
                "source_rgb and source_depth must have matching batch and "
                f"spatial dimensions, got {tuple(source_rgb.shape)} and "
                f"{tuple(source_depth.shape)}"
            )

        # Initialize outputs
        warped_rgb = torch.zeros(B, C, H_tgt, W_tgt, device=self.device)
        warped_depth = torch.zeros(B, 1, H_tgt, W_tgt, device=self.device)
        warped_mask = torch.zeros(B, 1, H_tgt, W_tgt, device=self.device, dtype=torch.bool)

        # Get camera parameters
        # Handle both dict format and PyTorch3D Cameras object
        src_is_pytorch3d = False
        tgt_is_pytorch3d = False
        try:
            from pytorch3d.renderer.cameras import CamerasBase
        except ImportError:
            # Dict cameras are sufficient for lightweight callers that do not
            # install the optional PyTorch3D package.
            CamerasBase = ()

        # Process cam_src
        if isinstance(cam_src, CamerasBase):
            src_is_pytorch3d = True
            K_src = self._pytorch3d_intrinsics(cam_src, B)
            if cam_src.R is not None:
                R_src = cam_src.R.to(self.device)  # (B, 3, 3)
            else:
                R_src = torch.eye(3, device=self.device).unsqueeze(0)  # Default R
            if cam_src.T is not None:
                t_src = cam_src.T.to(self.device)  # (B, 3, 1)
            else:
                t_src = torch.zeros(1, 3, 1, device=self.device)  # Default T
        elif hasattr(cam_src, 'cameras'):
            src_is_pytorch3d = True
            # Handle any camera collection type, take the first camera
            cam = cam_src.cameras[0]
            K_src = self._pytorch3d_intrinsics(cam, B)
            R_src = cam.R.to(self.device) if cam.R is not None else torch.eye(3, device=self.device).unsqueeze(0)
            t_src = cam.T.to(self.device) if cam.T is not None else torch.zeros(1, 3, 1, device=self.device)
        else:
            # Dict format
            K_src = cam_src['K']  # (B, 3, 3)
            R_src = cam_src['R']  # (B, 3, 3)
            t_src = cam_src['t']  # (B, 3, 1)

        # Process cam_tgt
        if isinstance(cam_tgt, CamerasBase):
            tgt_is_pytorch3d = True
            K_tgt = self._pytorch3d_intrinsics(cam_tgt, B)
            R_tgt = cam_tgt.R.to(self.device) if cam_tgt.R is not None else torch.eye(3, device=self.device).unsqueeze(0)
            t_tgt = cam_tgt.T.to(self.device) if cam_tgt.T is not None else torch.zeros(1, 3, 1, device=self.device)
        elif hasattr(cam_tgt, 'cameras'):
            tgt_is_pytorch3d = True
            # Handle any camera collection type, take the first camera
            cam = cam_tgt.cameras[0]
            K_tgt = self._pytorch3d_intrinsics(cam, B)
            R_tgt = cam.R.to(self.device) if cam.R is not None else torch.eye(3, device=self.device).unsqueeze(0)
            t_tgt = cam.T.to(self.device) if cam.T is not None else torch.zeros(1, 3, 1, device=self.device)
        else:
            # Dict format
            K_tgt = cam_tgt['K']  # (B, 3, 3)
            R_tgt = cam_tgt['R']  # (B, 3, 3)
            t_tgt = cam_tgt['t']  # (B, 3, 1)

        K_src = self._camera_tensor(K_src, B, 'source K', (3, 3))
        R_src = self._camera_tensor(R_src, B, 'source R', (3, 3))
        t_src = self._camera_tensor(t_src, B, 'source t', (3, 1))
        K_tgt = self._camera_tensor(K_tgt, B, 'target K', (3, 3))
        R_tgt = self._camera_tensor(R_tgt, B, 'target R', (3, 3))
        t_tgt = self._camera_tensor(t_tgt, B, 'target t', (3, 1))

        if src_is_pytorch3d:
            R_src, t_src = self._pytorch3d_to_rdf(R_src, t_src)
        if tgt_is_pytorch3d:
            R_tgt, t_tgt = self._pytorch3d_to_rdf(R_tgt, t_tgt)

        # Create grid of source pixels (B, H_s*W_s, 3)
        y_src, x_src = torch.meshgrid(torch.arange(H_s), torch.arange(W_s), indexing='ij')
        y_src_long = y_src.flatten().repeat(B, 1).to(self.device)
        x_src_long = x_src.flatten().repeat(B, 1).to(self.device)

        # Convert to float for matrix operations
        x_src_float = x_src_long.float()
        y_src_float = y_src_long.float()

        # Source pixel to camera rays (B, 3, H_s*W_s)
        pixels_src = torch.stack([x_src_float, y_src_float, torch.ones_like(x_src_float)], dim=1)  # (B, 3, N)
        dir_cam_src = torch.bmm(torch.inverse(K_src), pixels_src)  # (B, 3, N)

        # Get source depths (B, H_s*W_s)
        source_depth_flat = source_depth.squeeze(1).view(B, -1)  # (B, N)
        valid_source_depth = torch.isfinite(source_depth_flat) & (source_depth_flat > 0)
        # Keep invalid points numerically harmless; valid_source_depth below
        # still prevents them from becoming evidence.
        source_depth_safe = torch.where(
            valid_source_depth, source_depth_flat, torch.ones_like(source_depth_flat)
        )

        # Source camera space 3D points (B, 3, H_s*W_s)
        X_cam_src = dir_cam_src * source_depth_safe.unsqueeze(1)  # (B, 3, N)

        # Source camera space to world space (B, 3, H_s*W_s)
        X_world = torch.bmm(R_src.transpose(1, 2), X_cam_src - t_src)  # (B, 3, N)

        # World space to target camera space (B, 3, H_s*W_s)
        X_cam_tgt = torch.bmm(R_tgt, X_world) + t_tgt  # (B, 3, N)
        if return_uncertainty:
            std_source = 'relative_depth_assumption'
            std = source_depth * self.init_depth_rel_std
            if source_depth_std is not None:
                std = self._as_map(source_depth_std, 'source depth std')
                if std.shape != source_depth.shape:
                    raise ValueError('source depth std must match source depth shape')
                std_source = 'provided_depth_std'
            depth_direction = R_tgt @ R_src.transpose(1, 2) @ dir_cam_src
            covariance = projection_covariance(
                X_cam_tgt, depth_direction, K_tgt, std.reshape(B, 1, -1)
            )
            warped_covariance = torch.full(
                (B, 3, H_tgt, W_tgt), float('nan'), device=self.device
            )

        # A point must be valid in both cameras before it can become
        # geometric evidence.  Invalid source depths otherwise create fake
        # points near the camera center.
        valid_mask = valid_source_depth & (X_cam_tgt[:, 2, :] > 0)  # (B, N)

        # Project to target pixels (B, 3, H_s*W_s).  Use a safe denominator
        # for points that will be rejected by ``valid_mask`` anyway.
        target_z = X_cam_tgt[:, 2, :]
        safe_target_z = torch.where(
            target_z > 0, target_z, torch.ones_like(target_z)
        )
        X_cam_tgt_norm = X_cam_tgt / safe_target_z.unsqueeze(1)  # (B, 3, N)
        pixels_tgt = torch.bmm(K_tgt, X_cam_tgt_norm)  # (B, 3, N)

        # Convert to pixel coordinates
        x_tgt = pixels_tgt[:, 0, :]  # (B, N)
        y_tgt = pixels_tgt[:, 1, :]  # (B, N)

        # Clip to target image bounds and round to nearest integer
        x_tgt = x_tgt.round().long()
        y_tgt = y_tgt.round().long()

        # Check if pixels are within target bounds
        in_bounds = (x_tgt >= 0) & (x_tgt < W_tgt) & (y_tgt >= 0) & (y_tgt < H_tgt)

        # Combine all validity masks
        valid_pixels = valid_mask & in_bounds  # (B, N)

        # Resolve the z-buffer in tensor reductions, avoiding one GPU sync per
        # source pixel. Ties keep the earliest source pixel, as the scalar loop did.
        for b in range(B):
            valid = valid_pixels[b]
            indices = y_tgt[b, valid] * W_tgt + x_tgt[b, valid]
            depths = X_cam_tgt[b, 2, valid]
            count = depths.numel()
            if count == 0:
                continue
            nearest = torch.full((H_tgt * W_tgt,), float('inf'), device=self.device)
            nearest.scatter_reduce_(0, indices, depths, reduce='amin', include_self=True)
            order = torch.arange(count, device=self.device)
            candidates = torch.where(depths == nearest[indices], order, count)
            winners = torch.full((H_tgt * W_tgt,), count, device=self.device, dtype=torch.long)
            winners.scatter_reduce_(0, indices, candidates, reduce='amin', include_self=True)
            covered = winners < count
            selected = winners[covered]
            source_colors = source_rgb[b].reshape(C, -1)[:, valid]
            warped_rgb[b].view(C, -1)[:, covered] = source_colors[:, selected]
            warped_depth[b].view(-1)[covered] = depths[selected]
            warped_mask[b].view(-1)[covered] = True
            if return_uncertainty:
                warped_covariance[b].view(3, -1)[:, covered] = covariance[b, :, valid][:, selected]

        if return_uncertainty:
            return warped_rgb, warped_depth, warped_mask.float(), {
                'projection_covariance': warped_covariance,
                'uncertainty_source': std_source,
            }
        return warped_rgb, warped_depth, warped_mask.float()

    def select_reference_frames(self, t, frame_list, num_recent=None, num_ref=None, overlap_threshold=None):
        """
        Select reference frames with high geometric overlap for the current frame t.

        :param t: Current frame index
        :param frame_list: List of FrameData objects
        :param num_recent: Number of recent frames to consider
        :param num_ref: Number of reference frames to select
        :param overlap_threshold: IoU threshold for selecting reference frames
        :return: (ref_indices, per_ref_warp) - ref_indices is list of selected indices,
                 per_ref_warp contains warped_rgb and warped_mask for each reference frame
        """
        num_recent = self.num_recent if num_recent is None else num_recent
        num_ref = self.num_ref if num_ref is None else num_ref
        overlap_threshold = self.overlap_threshold if overlap_threshold is None else overlap_threshold
        if t == 0:
            # No reference frames for the first frame
            return [], {}

        # Get current frame
        current_frame = frame_list[t]
        current_rgb = self._as_rgb(current_frame.rgb, 'target rgb')
        current_mask = (self._as_map(current_frame.mask, 'target mask') > 0.5).to(
            dtype=torch.float32
        )
        H, W = current_rgb.shape[-2:]

        # Select recent frames
        candidate_start = max(0, t - num_recent)
        candidate_indices = list(range(candidate_start, t))

        # Calculate overlap for each candidate
        overlap_scores = []
        per_ref_warp = {}

        for s in candidate_indices:
            # Get source frame
            source_frame = frame_list[s]

            # Invalid source pixels must not become 3-D evidence merely
            # because their depth happens to contain a finite placeholder.
            source_depth = self._as_map(source_frame.depth, 'source depth')
            source_mask = self._as_map(source_frame.mask, 'source mask')
            if source_depth.shape != source_mask.shape:
                raise ValueError(
                    "source depth and source mask must have identical shapes, "
                    f"got {tuple(source_depth.shape)} and {tuple(source_mask.shape)}"
                )
            source_depth = source_depth * (source_mask > 0.5).to(
                dtype=source_depth.dtype
            )

            # Warp source to target
            warp_result = self.warp_rgb_depth(
                source_frame.rgb,
                source_depth,
                source_frame.camera,
                current_frame.camera,
                H,
                W,
                source_depth_std=getattr(source_frame, 'depth_std', None),
                return_uncertainty=self.init_scale_mode == 'uncertainty',
            )
            warped_rgb, warped_depth, warped_mask = warp_result[:3]

            # Calculate IoU
            intersection = (warped_mask * current_mask).sum(dim=(1, 2, 3))
            union = (warped_mask + current_mask).sum(dim=(1, 2, 3)) + 1e-8
            overlap = intersection / union  # (B,)
            overlap_value = overlap.mean().item()

            # Store results
            overlap_scores.append(overlap_value)
            per_ref_warp[s] = {
                'warped_rgb': warped_rgb,
                'warped_depth': warped_depth,
                'warped_mask': warped_mask,
                'overlap': overlap_value
            }
            if len(warp_result) == 4:
                per_ref_warp[s].update(warp_result[3])

        # Filter candidates by overlap threshold
        filtered_candidates = [(idx, score) for idx, score in zip(candidate_indices, overlap_scores) if score >= overlap_threshold]

        # Sort by overlap score descending
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top num_ref candidates
        selected_candidates = filtered_candidates[:num_ref]
        ref_indices = [c[0] for c in selected_candidates]

        return ref_indices, per_ref_warp

    def _empty_evidence_state(self, frame):
        """Return an explicit abstention state when no reference is usable."""
        depth = self._as_map(frame.depth, 'target depth')
        batch = depth.shape[0]
        zero_map = torch.zeros_like(depth, device=self.device)
        zeros = torch.zeros(batch, device=self.device, dtype=depth.dtype)
        return {
            'support_count': zero_map,
            'support_ratio': zeros,
            'admissible_ratio': zeros,
            'conflict_ratio': zeros,
            'free_space_ratio': zeros,
            'occlusion_ratio': zeros,
            'visibility_risk_ratio': zeros,
            'conflict_mask': zero_map,
            'free_space_mask': zero_map,
            'occlusion_mask': zero_map,
            'geometry_mask': zero_map,
            'routing_mask': zero_map,
            'region_state': torch.zeros_like(zero_map, dtype=torch.long),
            'per_reference_evidence': zero_map.new_zeros(
                (0, batch, zero_map.shape[1], zero_map.shape[2], zero_map.shape[3])
            ),
            'per_reference_gate': zeros.new_zeros((batch, 0)),
            'frame_gate': zeros,
            'state_strength': zeros,
            'state': torch.zeros(batch, device=self.device, dtype=torch.long),
            'reference_indices': [],
            'pair_gates': {},
            'routing_enabled': bool(self.use_evidence_routing),
        }

    def build_evidence_state(self, t, frame_list, ref_indices, per_ref_warp):
        """Verify local multi-view evidence without rewriting the scaffold.

        The returned state is consumed twice: its soft geometry mask limits
        the initial latent injection, while its scalar pair gates bias every
        temporal attention block during denoising.
        """
        current_frame = frame_list[t]
        if not self.use_evidence_routing:
            # Preserve the pre-existing overlap-based path when the new
            # controller is explicitly disabled.
            evidence = self._empty_evidence_state(current_frame)
            evidence['reference_indices'] = list(ref_indices)
            batch = self._as_map(current_frame.depth, 'target depth').shape[0]
            # The legacy controller used a hard overlap admission rule. Keep
            # that binary semantics behind the compatibility switch so an
            # ablation does not silently change the baseline path.
            evidence['pair_gates'] = {
                ref_idx: torch.ones(
                    (batch,),
                    device=self.device,
                    dtype=torch.float32,
                )
                for ref_idx in ref_indices
            }
            return evidence
        if not ref_indices:
            return self._empty_evidence_state(current_frame)

        warped_depths = []
        warped_masks = []
        valid_ref_indices = []
        for ref_idx in ref_indices:
            warp_info = per_ref_warp.get(ref_idx, {})
            if 'warped_depth' not in warp_info or 'warped_mask' not in warp_info:
                continue
            warped_depths.append(warp_info['warped_depth'])
            warped_masks.append(warp_info['warped_mask'])
            valid_ref_indices.append(ref_idx)

        if not warped_depths:
            return self._empty_evidence_state(current_frame)

        target_depth = self._as_map(current_frame.depth, 'target depth')
        target_mask = self._as_map(current_frame.mask, 'target mask')
        evidence = compute_geometry_evidence(
            torch.stack(warped_depths, dim=0),
            torch.stack(warped_masks, dim=0),
            target_depth,
            target_mask,
            depth_rel_tolerance=self.depth_rel_tolerance,
            depth_abs_tolerance=self.depth_abs_tolerance,
            conflict_threshold=self.conflict_threshold,
            min_support_ratio=self.min_support_ratio,
            admit_ratio_threshold=self.admit_ratio_threshold,
            max_conflict_ratio=self.max_conflict_ratio,
            attenuated_weight=self.attenuated_weight,
        )
        evidence['reference_indices'] = valid_ref_indices
        evidence['pair_gates'] = {
            ref_idx: evidence['per_reference_gate'][:, local_idx]
            for local_idx, ref_idx in enumerate(valid_ref_indices)
        }
        evidence['routing_enabled'] = True
        return evidence

    @staticmethod
    def export_evidence_state(evidence):
        """Keep a CPU-safe, compact record for downstream reconstruction."""
        tensor_keys = (
            'support_ratio', 'admissible_ratio', 'conflict_ratio',
            'free_space_ratio', 'visibility_risk_ratio', 'conflict_mask',
            'occlusion_ratio', 'occlusion_mask',
            'free_space_mask', 'geometry_mask', 'routing_mask',
            'region_state', 'frame_gate', 'state_strength', 'state'
        )
        integer_keys = {'state', 'region_state'}
        record = {
            key: evidence[key].detach().cpu() if key in integer_keys
            else evidence[key].detach().float().cpu()
            for key in tensor_keys
            if key in evidence
        }
        for key in ('support_count', 'per_reference_gate'):
            if key in evidence and torch.is_tensor(evidence[key]):
                record[key] = evidence[key].detach().float().cpu()
        record['reference_indices'] = list(evidence.get('reference_indices', []))
        record['routing_enabled'] = bool(evidence.get('routing_enabled', True))
        if 'initialization' in evidence:
            record['initialization'] = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in evidence['initialization'].items()
            }
        record['state_names'] = [
            ('abstain', 'attenuate', 'admit')[int(state)]
            for state in record.get('state', torch.zeros(1, dtype=torch.long)).view(-1).tolist()
        ]
        return record

    @staticmethod
    def _latent_mode(encoded):
        if isinstance(encoded, DiagonalGaussianDistribution):
            return encoded.mode()
        if hasattr(encoded, 'mode'):
            return encoded.mode()
        if hasattr(encoded, 'sample'):
            return encoded.sample()
        return encoded

    def _encode_guide_latent(self, guide_rgb):
        """Encode a guide in the same scaled latent space as the denoiser."""
        encoded = self.vae.encode(guide_rgb)
        guide_latent = self._latent_mode(encoded)
        if not torch.is_tensor(guide_latent):
            return guide_latent
        if self.diffusion_model is not None:
            scale_factor = getattr(self.diffusion_model, 'scale_factor', 1.0)
            if torch.is_tensor(scale_factor):
                scale_factor = scale_factor.to(
                    device=guide_latent.device, dtype=guide_latent.dtype
                )
            guide_latent = guide_latent * scale_factor
        return guide_latent

    def initialize_noise_with_fmi(self, t, base_noise, frame_list, ref_indices, per_ref_warp,
                                  mode="lowfreq_mix", evidence_state=None):
        """Apply evidence-gated frequency mixing to the initial latent.

        FMI (frequency-mixed initialization) keeps one fresh noise realization
        for the target frame and adds only the low-frequency residual supported
        by the local evidence maps.  When several references are available,
        their residuals are averaged with pixel-wise evidence weights rather
        than choosing a single global guide.  With no admitted support the
        function returns the untouched Gaussian latent.
        """
        del t, frame_list, mode  # retained in the signature for old callers
        diagnostics = {'mode': self.init_scale_mode, 'applied': False, 'reason': 'unavailable_support_or_encoder'}
        if evidence_state is not None:
            evidence_state['initialization'] = diagnostics
        if not self.use_fmi or not ref_indices or self.vae is None:
            if not self.use_fmi:
                diagnostics['reason'] = 'fmi_disabled'
            return base_noise

        B, C, H_l, W_l = base_noise.shape
        valid_ref_indices = [s for s in ref_indices if s in per_ref_warp]
        if not valid_ref_indices:
            return base_noise

        # Build one spatial routing map per reference.  The map is soft: an
        # attenuated state therefore changes the injection magnitude without
        # turning an entire frame into a hard on/off decision.
        evidence_maps = None
        evidence_maps_already_scaled = False
        if evidence_state is not None and evidence_state.get('routing_enabled', True):
            per_reference = evidence_state.get('per_reference_evidence')
            evidence_refs = evidence_state.get('reference_indices', valid_ref_indices)
            if torch.is_tensor(per_reference) and per_reference.ndim == 4:
                per_reference = per_reference.unsqueeze(2)
            if torch.is_tensor(per_reference) and per_reference.ndim == 5:
                ref_to_local = {ref: local for local, ref in enumerate(evidence_refs)}
                maps = []
                for ref_idx in valid_ref_indices:
                    local = ref_to_local.get(ref_idx)
                    routing_mask = evidence_state.get(
                        'routing_mask', evidence_state.get('geometry_mask')
                    )
                    if local is None or local >= per_reference.shape[0]:
                        if not torch.is_tensor(routing_mask):
                            return base_noise
                        maps.append(torch.zeros_like(routing_mask))
                    else:
                        maps.append(per_reference[local])
                evidence_maps = torch.stack(maps, dim=0)
            elif torch.is_tensor(evidence_state.get('routing_mask')):
                # Records exported by the current implementation contain the
                # final action mask but not the per-reference fields.  This
                # mask already includes the frame-level state strength.
                evidence_maps = evidence_state['routing_mask'].unsqueeze(0).expand(
                    len(valid_ref_indices), -1, -1, -1, -1
                )
                evidence_maps_already_scaled = True
            elif torch.is_tensor(evidence_state.get('geometry_mask')):
                # Compatibility with records produced before per-reference
                # maps were exported.
                evidence_maps = evidence_state['geometry_mask'].unsqueeze(0).expand(
                    len(valid_ref_indices), -1, -1, -1, -1
                )

            if evidence_maps is not None and not evidence_maps_already_scaled:
                state_strength = evidence_state.get('state_strength')
                if torch.is_tensor(state_strength):
                    evidence_maps = evidence_maps * state_strength.to(
                        device=evidence_maps.device, dtype=evidence_maps.dtype
                    ).view(1, -1, 1, 1, 1)

        if evidence_maps is None:
            if any('warped_mask' not in per_ref_warp[s] for s in valid_ref_indices):
                return base_noise
            evidence_maps = torch.stack(
                [per_ref_warp[s]['warped_mask'] for s in valid_ref_indices], dim=0
            )

        evidence_maps = evidence_maps.to(device=base_noise.device, dtype=base_noise.dtype).clamp(0.0, 1.0)
        _, map_batch, _, H, W = evidence_maps.shape
        if map_batch != B:
            raise ValueError(
                f"Evidence batch ({map_batch}) does not match latent batch ({B})"
            )
        evidence_maps_latent = F.interpolate(
            evidence_maps.reshape(-1, 1, H, W),
            (H_l, W_l),
            mode='bilinear',
            align_corners=False,
        ).reshape(len(valid_ref_indices), B, 1, H_l, W_l)
        route_mask = evidence_maps_latent.max(dim=0).values
        sigma = None
        if self.init_scale_mode != 'legacy':
            support_flat = evidence_maps_latent.reshape(-1, 1, H_l, W_l)
            if self.init_scale_mode == 'fixed':
                sigma = torch.full_like(support_flat, self.init_fixed_sigma, dtype=torch.float32)
                diagnostics['uncertainty_sources'] = ['fixed_control'] * len(valid_ref_indices)
            else:
                covariances = []
                sources = []
                for ref in valid_ref_indices:
                    info = per_ref_warp[ref]
                    covariance = info.get('projection_covariance')
                    if covariance is None:
                        covariance = torch.full((B, 3, H, W), float('nan'), device=base_noise.device)
                    covariances.append(covariance.to(base_noise.device))
                    sources.append(info.get('uncertainty_source', 'missing'))
                covariances = torch.stack(covariances)
                valid_covariance = torch.isfinite(covariances).all(dim=2, keepdim=True)
                support_flat = F.interpolate(
                    (evidence_maps * valid_covariance).reshape(-1, 1, H, W),
                    (H_l, W_l), mode='bilinear', align_corners=False,
                )
                sigma = latent_sigma(covariances.reshape(-1, 3, H, W),
                                     evidence_maps.reshape(-1, 1, H, W), (H_l, W_l))
                diagnostics['projection_covariance'] = covariances.detach()
                diagnostics['uncertainty_sources'] = sources
            eligible = torch.isfinite(sigma) & (sigma <= self.init_scales[-1])
            evidence_maps_latent = (support_flat * eligible).reshape_as(evidence_maps_latent)
            route_mask = evidence_maps_latent.max(dim=0).values
            diagnostics.update(latent_sigma=sigma.detach(),
                               excluded_scale_mask=(~eligible).detach(),
                               relative_depth_std=self.init_depth_rel_std,
                               scales=list(self.init_scales))
        diagnostics['effective_init_mask'] = route_mask.detach()
        if float(route_mask.detach().max()) <= 1e-6:
            diagnostics['reason'] = 'no_admissible_scale_support'
            return base_noise

        # Encode all candidate guides in one VAE call, then restore the
        # [reference, batch, channel, height, width] layout.
        guide_rgb = torch.stack(
            [per_ref_warp[s]['warped_rgb'] for s in valid_ref_indices], dim=0
        )
        guide_rgb = torch.clamp(guide_rgb * 2.0 - 1.0, -1.0, 1.0)
        guide_latent = self._encode_guide_latent(
            guide_rgb.reshape(-1, *guide_rgb.shape[2:])
        )
        if not torch.is_tensor(guide_latent) or guide_latent.ndim != 4:
            raise ValueError(
                "The VAE encoder must return a 4-D latent tensor shaped "
                "[reference*batch, channels, height, width]."
            )
        expected_batch = len(valid_ref_indices) * B
        if guide_latent.shape[0] != expected_batch:
            raise ValueError(
                "The VAE latent batch does not match the number of evidence "
                f"guides: {guide_latent.shape[0]} vs {expected_batch}."
            )
        if guide_latent.shape[1] != C:
            raise ValueError(
                "The VAE latent channel count must match the diffusion latent: "
                f"{guide_latent.shape[1]} vs {C}."
            )
        if guide_latent.shape[-2:] != (H_l, W_l):
            guide_latent = F.interpolate(
                guide_latent,
                size=(H_l, W_l),
                mode='bilinear',
                align_corners=False,
            )
        guide_latent = guide_latent.reshape(len(valid_ref_indices), B, C, H_l, W_l)

        base_noise_flat = base_noise.unsqueeze(0).expand(
            len(valid_ref_indices), -1, -1, -1, -1
        ).reshape(-1, C, H_l, W_l)
        guide_latent_flat = guide_latent.reshape(-1, C, H_l, W_l)

        # Keep guide noising identical when comparing FFT and spatial mixing.
        if self.diffusion_model is not None and hasattr(self.diffusion_model, 'q_sample'):
            num_timesteps = int(getattr(self.diffusion_model, 'num_timesteps', self.noise_level + 1))
            timestep_value = min(max(int(self.noise_level), 0), max(num_timesteps - 1, 0))
            timestep = torch.full(
                (guide_latent_flat.shape[0],), timestep_value,
                device=base_noise.device,
                dtype=torch.long,
            )
            guide_latent_flat = self.diffusion_model.q_sample(
                x_start=guide_latent_flat, t=timestep, noise=base_noise_flat
            )

        if self.use_freq_mix:
            LPF = self.get_freq_filter(guide_latent_flat.shape)
            mixed = freq_mix_2d(
                guide_latent_flat.to(dtype=torch.float32),
                base_noise_flat.to(dtype=torch.float32),
                LPF,
                low_freq_norm=self.low_freq_norm,
                norm_factor=1.0,
            ).to(dtype=base_noise.dtype)
        else:
            low_guide = F.avg_pool2d(guide_latent_flat, kernel_size=3, stride=1, padding=1)
            base_low = F.avg_pool2d(base_noise_flat, kernel_size=3, stride=1, padding=1)
            mixed = low_guide + (base_noise_flat - base_low)

        # Aggregate residuals with local evidence weights.  The common noise
        # is never averaged, so unsupported high-frequency detail stays fresh.
        residual = (mixed - base_noise_flat).reshape(
            len(valid_ref_indices), B, C, H_l, W_l
        )
        weights = evidence_maps_latent
        if sigma is not None:
            filtered, support, scale_weights = scale_residual(
                residual.reshape(-1, C, H_l, W_l),
                weights.reshape(-1, 1, H_l, W_l), sigma, self.init_scales,
            )
            residual = filtered.reshape_as(residual)
            weights = support.reshape_as(weights)
            diagnostics['scale_weights'] = scale_weights.detach()
        weight_sum = weights.sum(dim=0).clamp_min(1e-6)
        consensus_residual = (residual * weights).sum(dim=0) / weight_sum
        diagnostics.update(applied=True, reason='applied', reference_indices=list(valid_ref_indices))
        return (
            base_noise
            + route_mask * consensus_residual
        ).to(dtype=base_noise.dtype)

    def initialize_noise_with_pani(self, *args, **kwargs):
        """Backward-compatible alias for the former initialization name."""
        return self.initialize_noise_with_fmi(*args, **kwargs)

    def build_frame_attention_bias(self, selection_history, total_frames, batch_size):
        if not selection_history or not self.use_cga:
            return None

        # Leave ordinary temporal attention unchanged.  Only frames that were
        # considered as geometric evidence carriers receive a route-specific
        # bias; this keeps the controller local to the evidence path.
        floor = max(float(self.abstain_gate), 1e-4)
        frame_bias = torch.zeros(
            (batch_size, total_frames, total_frames),
            device=self.device,
        )
        has_routed_pair = False

        for t, data in selection_history.items():
            if t >= total_frames:
                continue
            evidence = data.get('evidence_state')
            candidate_indices = set(data.get('per_ref_warp', {}).keys())
            candidate_indices.update(data.get('ref_indices', []))
            if evidence is None:
                # Compatibility path for callers that only provide the old
                # overlap-based history.
                for s in candidate_indices:
                    if 0 <= s < total_frames:
                        if s in data.get('ref_indices', []):
                            continue
                        value = float(self.large_negative)
                        frame_bias[:, t, s] = value
                        has_routed_pair = True
                continue

            if not evidence.get('routing_enabled', True):
                # The explicit ablation switch must reproduce the former
                # binary overlap semantics, including its strong penalty for
                # rejected candidates.
                selected = set(data.get('ref_indices', []))
                for s in candidate_indices:
                    if not (0 <= s < total_frames) or s in selected:
                        continue
                    frame_bias[:, t, s] = float(self.large_negative)
                    has_routed_pair = True
                continue

            pair_gates = evidence.get('pair_gates', {})
            for s in candidate_indices:
                if not (0 <= s < total_frames):
                    continue
                gate = pair_gates.get(s, 0.0)
                gate = torch.as_tensor(gate, device=self.device, dtype=frame_bias.dtype).view(-1)
                if gate.numel() == 1 and batch_size > 1:
                    gate = gate.expand(batch_size)
                elif gate.numel() != batch_size:
                    gate = gate.mean().expand(batch_size)
                frame_bias[:, t, s] = torch.log(gate.clamp(min=floor, max=1.0))
                has_routed_pair = True

        if not has_routed_pair:
            return None

        return {
            'frame_bias': frame_bias,
            'temporal_length': total_frames,
            'abstain_gate': floor,
            'geo_early_scale': float(self.geo_early_scale),
            'geo_late_scale': float(self.geo_late_scale),
            'temporal_attention_only': True,
        }

    def build_cga_bias(self, selection_history, total_frames, batch_size):
        """Named entry point for consistency-gated temporal attention."""
        return self.build_frame_attention_bias(selection_history, total_frames, batch_size)


# Backward-compatible name for callers that imported the original engine.
WarpGuidanceEngine = EvidenceRoutingEngine
