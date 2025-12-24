import torch
import torch.nn.functional as F
import torch.fft as fft
import numpy as np
from einops import rearrange
try:
    from ViewCrafter123.lvdm.distributions import DiagonalGaussianDistribution
except ModuleNotFoundError:
    from lvdm.distributions import DiagonalGaussianDistribution


# ============== WAVE-style Frequency Domain Functions ==============

def get_freq_filter(shape, device, filter_type='gaussian', d_s=0.25):
    """
    Generate a low-pass filter for frequency domain operations.
    Ported from WAVE project.
    
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
    
    # Gaussian filter
    sigma = d_s
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
    
    LPF = (freq_dist <= d_s).float()
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
    WAVE-style frequency domain noise mixing for 2D latents.
    
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
    WAVE-style frequency domain noise mixing for 3D (video) latents.
    
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

class WarpGuidanceEngine:
    """
    Warp guidance engine that implements the WAVE-style geometric guidance mechanisms
    for enhancing multi-view/multi-frame geometric consistency in ViewCrafter's video diffusion.
    
    Now with proper FFT frequency domain mixing from WAVE paper for stable background.
    """

    def __init__(self, vae_encoder=None, device='cuda', num_recent=8, num_ref=3, 
                 overlap_threshold=0.1, large_negative=-1e4,
                 # WAVE-style frequency domain parameters
                 use_freq_mix=True,          # Enable FFT frequency mixing
                 filter_type='gaussian',      # 'gaussian' or 'ideal'
                 freq_cutoff=0.25,           # Normalized frequency cutoff (d_s)
                 low_freq_norm=True,         # Normalize low-frequency components
                 noise_level=999,            # Noise level for q_sample (from WAVE)
                 shared_noise_seed=None):    # Shared seed for consistent noise across frames
        """
        Initialize the warp guidance engine.
        :param vae_encoder: VAE encoder for latent space operations
        :param device: Device to use for computations
        :param use_freq_mix: Use WAVE-style FFT frequency mixing (recommended)
        :param filter_type: Type of low-pass filter ('gaussian' or 'ideal')
        :param freq_cutoff: Cutoff frequency for the filter (0.0-1.0, smaller = more blur)
        :param low_freq_norm: Normalize low-frequency components (key for stability)
        :param noise_level: Noise level for forward diffusion (WAVE uses 999)
        :param shared_noise_seed: Seed for shared noise generation across frames
        """
        self.vae_encoder = vae_encoder
        self.device = device
        self.num_recent = num_recent
        self.num_ref = num_ref
        self.overlap_threshold = overlap_threshold
        self.large_negative = large_negative
        
        # WAVE-style frequency domain settings
        self.use_freq_mix = use_freq_mix
        self.filter_type = filter_type
        self.freq_cutoff = freq_cutoff
        self.low_freq_norm = low_freq_norm
        self.noise_level = noise_level
        self.shared_noise_seed = shared_noise_seed
        
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

    def warp_rgb_depth(self, source_rgb, source_depth, cam_src, cam_tgt, H_tgt, W_tgt):
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
        # Ensure source_rgb has 4 dimensions (B, C, H, W)
        if len(source_rgb.shape) == 3:
            # Add batch dimension
            source_rgb = source_rgb.unsqueeze(0)
            source_depth = source_depth.unsqueeze(0)

        B, C, H_s, W_s = source_rgb.shape

        # Initialize outputs
        warped_rgb = torch.zeros(B, C, H_tgt, W_tgt, device=self.device)
        warped_depth = torch.zeros(B, 1, H_tgt, W_tgt, device=self.device)
        warped_mask = torch.zeros(B, 1, H_tgt, W_tgt, device=self.device, dtype=torch.bool)

        # Get camera parameters
        # Handle both dict format and PyTorch3D Cameras object
        from pytorch3d.renderer.cameras import CamerasBase

        # Get camera parameters
        # Handle both dict format and PyTorch3D Cameras object
        from pytorch3d.renderer.cameras import CamerasBase

        # Process cam_src
        if isinstance(cam_src, CamerasBase):
            # Single PyTorch3D Cameras object
            # Check if attributes are None
            if cam_src.K is not None:
                K_src = cam_src.K.to(self.device)  # (B, 3, 3)
            else:
                # Try alternative ways to get camera parameters
                K_src = torch.eye(3, device=self.device).unsqueeze(0)  # Default K
            if cam_src.R is not None:
                R_src = cam_src.R.to(self.device)  # (B, 3, 3)
            else:
                R_src = torch.eye(3, device=self.device).unsqueeze(0)  # Default R
            if cam_src.T is not None:
                t_src = cam_src.T.to(self.device)  # (B, 3, 1)
            else:
                t_src = torch.zeros(1, 3, 1, device=self.device)  # Default T
        elif hasattr(cam_src, 'cameras'):
            # Handle any camera collection type, take the first camera
            cam = cam_src.cameras[0]
            K_src = cam.K.to(self.device) if cam.K is not None else torch.eye(3, device=self.device).unsqueeze(0)
            R_src = cam.R.to(self.device) if cam.R is not None else torch.eye(3, device=self.device).unsqueeze(0)
            t_src = cam.T.to(self.device) if cam.T is not None else torch.zeros(1, 3, 1, device=self.device)
        else:
            # Dict format
            K_src = cam_src['K'].to(self.device)  # (B, 3, 3)
            R_src = cam_src['R'].to(self.device)  # (B, 3, 3)
            t_src = cam_src['t'].to(self.device)  # (B, 3, 1)

        # Process cam_tgt
        if isinstance(cam_tgt, CamerasBase):
            # Single PyTorch3D Cameras object
            K_tgt = cam_tgt.K.to(self.device) if cam_tgt.K is not None else torch.eye(3, device=self.device).unsqueeze(0)
            R_tgt = cam_tgt.R.to(self.device) if cam_tgt.R is not None else torch.eye(3, device=self.device).unsqueeze(0)
            t_tgt = cam_tgt.T.to(self.device) if cam_tgt.T is not None else torch.zeros(1, 3, 1, device=self.device)
        elif hasattr(cam_tgt, 'cameras'):
            # Handle any camera collection type, take the first camera
            cam = cam_tgt.cameras[0]
            K_tgt = cam.K.to(self.device) if cam.K is not None else torch.eye(3, device=self.device).unsqueeze(0)
            R_tgt = cam.R.to(self.device) if cam.R is not None else torch.eye(3, device=self.device).unsqueeze(0)
            t_tgt = cam.T.to(self.device) if cam.T is not None else torch.zeros(1, 3, 1, device=self.device)
        else:
            # Dict format
            K_tgt = cam_tgt['K'].to(self.device)  # (B, 3, 3)
            R_tgt = cam_tgt['R'].to(self.device)  # (B, 3, 3)
            t_tgt = cam_tgt['t'].to(self.device)  # (B, 3, 1)

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

        # Source camera space 3D points (B, 3, H_s*W_s)
        X_cam_src = dir_cam_src * source_depth_flat.unsqueeze(1)  # (B, 3, N)

        # Source camera space to world space (B, 3, H_s*W_s)
        X_world = torch.bmm(R_src.transpose(1, 2), X_cam_src - t_src)  # (B, 3, N)

        # World space to target camera space (B, 3, H_s*W_s)
        X_cam_tgt = torch.bmm(R_tgt, X_world) + t_tgt  # (B, 3, N)

        # Check valid depth (Z > 0)
        valid_mask = X_cam_tgt[:, 2, :] > 0  # (B, N)

        # Project to target pixels (B, 3, H_s*W_s)
        X_cam_tgt_norm = X_cam_tgt / X_cam_tgt[:, 2, :].unsqueeze(1)  # (B, 3, N)
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

        # Flatten batch indices
        batch_indices = torch.arange(B).unsqueeze(1).expand(-1, H_s*W_s).to(self.device)

        # Only process valid pixels
        for b in range(B):
            # Get valid pixels in this batch
            valid_mask_b = valid_pixels[b]
            if not valid_mask_b.any():
                continue

            # Get source pixels
            src_rgb = source_rgb[b, :, y_src_long[b][valid_mask_b], x_src_long[b][valid_mask_b]]  # (3, M)
            src_depth = source_depth[b, 0, y_src_long[b][valid_mask_b], x_src_long[b][valid_mask_b]]  # (M)

            # Get target pixels
            tgt_y = y_tgt[b][valid_mask_b]
            tgt_x = x_tgt[b][valid_mask_b]

            # Initialize z-buffer for this batch
            z_buffer = torch.full((H_tgt, W_tgt), float('inf'), device=self.device)

            # Write to target image with z-buffer
            for m in range(src_depth.shape[0]):
                y = tgt_y[m]
                x = tgt_x[m]
                depth = X_cam_tgt[b, 2, valid_mask_b][m]

                if depth < z_buffer[y, x]:
                    # Update with closer pixel
                    warped_rgb[b, :, y, x] = src_rgb[:, m]
                    warped_depth[b, 0, y, x] = src_depth[m]
                    warped_mask[b, 0, y, x] = True
                    z_buffer[y, x] = depth

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
        num_recent = num_recent or self.num_recent
        num_ref = num_ref or self.num_ref
        overlap_threshold = overlap_threshold or self.overlap_threshold
        if t == 0:
            # No reference frames for the first frame
            return [], {}

        # Get current frame
        current_frame = frame_list[t]
        H, W = current_frame.rgb.shape[2], current_frame.rgb.shape[3]

        # Select recent frames
        candidate_start = max(0, t - num_recent)
        candidate_indices = list(range(candidate_start, t))

        # Calculate overlap for each candidate
        overlap_scores = []
        per_ref_warp = {}

        for s in candidate_indices:
            # Get source frame
            source_frame = frame_list[s]

            # Warp source to target
            warped_rgb, warped_depth, warped_mask = self.warp_rgb_depth(
                source_frame.rgb,
                source_frame.depth,
                source_frame.camera,
                current_frame.camera,
                H,
                W
            )

            # Calculate IoU
            current_mask = current_frame.mask  # (B, 1, H, W)
            intersection = (warped_mask * current_mask).sum(dim=(1, 2, 3))
            union = (warped_mask + current_mask).sum(dim=(1, 2, 3)) + 1e-8
            overlap = intersection / union  # (B,)
            overlap_value = overlap.mean().item()

            # Store results
            overlap_scores.append(overlap_value)
            per_ref_warp[s] = {
                'warped_rgb': warped_rgb,
                'warped_mask': warped_mask,
                'overlap': overlap_value
            }

        # Filter candidates by overlap threshold
        filtered_candidates = [(idx, score) for idx, score in zip(candidate_indices, overlap_scores) if score >= overlap_threshold]

        # Sort by overlap score descending
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        # Select top num_ref candidates
        selected_candidates = filtered_candidates[:num_ref]
        ref_indices = [c[0] for c in selected_candidates]

        return ref_indices, per_ref_warp

    def initialize_noise_with_pani(self, t, base_noise, frame_list, ref_indices, per_ref_warp, mode="lowfreq_mix"):
        """
        Initialize diffusion latent with pose-aware noise initialization (PANI).
        
        Now uses WAVE-style FFT frequency domain mixing for better stability.

        :param t: Current frame index
        :param base_noise: Original Gaussian noise latent (B, C, H_l, W_l)
        :param frame_list: List of FrameData objects
        :param ref_indices: Selected reference frame indices
        :param per_ref_warp: Warp results from reference frames
        :param mode: PANI mode (e.g., "lowfreq_mix")
        :return: init_latent - Initialized latent with geometric guidance
        """
        if not ref_indices:
            # No reference frames, return base noise
            return base_noise

        B, C, H_l, W_l = base_noise.shape
        current_frame = frame_list[t]

        # Get warp results for all reference frames
        warped_rgbs = []
        warped_masks = []

        for s in ref_indices:
            warped_rgbs.append(per_ref_warp[s]['warped_rgb'])
            warped_masks.append(per_ref_warp[s]['warped_mask'])

        # Strategy A: Use the reference frame with highest IoU
        guide_rgb = warped_rgbs[0]  # First is highest overlap

        # Normalize guide RGB to [-1, 1] before VAE encoding
        guide_rgb_normalized = torch.clamp(guide_rgb * 2.0 - 1.0, -1.0, 1.0)

        if self.vae is None:
            return base_noise

        # Encode guide RGB to latent space
        encoded = self.vae.encode(guide_rgb_normalized)
        if isinstance(encoded, DiagonalGaussianDistribution):
            z_guide = encoded.mode()
        elif hasattr(encoded, 'mode'):
            z_guide = encoded.mode()
        elif hasattr(encoded, 'sample'):
            z_guide = encoded.sample()
        else:
            z_guide = encoded

        # Aggregate mask
        M_agg = torch.zeros_like(current_frame.mask, device=self.device)
        for mask in warped_masks:
            M_agg = torch.max(M_agg, mask)

        # Resize mask to latent size
        M_agg_latent = F.interpolate(M_agg, (H_l, W_l), mode='bilinear', align_corners=False)
        mask_expanded = (M_agg_latent > 0.1).float()
        mask_expanded = mask_expanded.expand(-1, C, -1, -1)

        # ========== WAVE-style FFT Frequency Domain Mixing ==========
        if self.use_freq_mix:
            # Get or create low-pass filter
            LPF = self.get_freq_filter(base_noise.shape)
            
            # Option 1: If we have diffusion model, use proper forward diffusion (like WAVE)
            if self.diffusion_model is not None and hasattr(self.diffusion_model, 'q_sample'):
                # Add noise to guide latent using forward diffusion process
                timestep = torch.full((z_guide.shape[0],), self.noise_level, 
                                      device=self.device, dtype=torch.long)
                z_guide_noisy = self.diffusion_model.q_sample(
                    x_start=z_guide, t=timestep, noise=base_noise
                )
                # Apply WAVE-style frequency mixing
                init_latent = freq_mix_2d(
                    z_guide_noisy.to(dtype=torch.float32),
                    base_noise.to(dtype=torch.float32),
                    LPF,
                    low_freq_norm=self.low_freq_norm,
                    norm_factor=1.0
                ).to(dtype=base_noise.dtype)
            else:
                # Option 2: Direct frequency mixing without q_sample
                # Use frequency domain mixing directly on z_guide and base_noise
                init_latent = freq_mix_2d(
                    z_guide.to(dtype=torch.float32),
                    base_noise.to(dtype=torch.float32),
                    LPF,
                    low_freq_norm=self.low_freq_norm,
                    norm_factor=1.0
                ).to(dtype=base_noise.dtype)
            
            # Apply mask: only use frequency mixing in warped regions
            # For non-warped regions, use original noise
            init_latent = base_noise * (1 - mask_expanded) + init_latent * mask_expanded
            
        else:
            # ========== Original avg_pool method (fallback) ==========
            low_guide = F.avg_pool2d(z_guide, kernel_size=3, stride=1, padding=1)
            base_low = F.avg_pool2d(base_noise, kernel_size=3, stride=1, padding=1)
            high_noise = base_noise - base_low

            # Combine low guide and high noise, only in masked regions
            init_latent = base_noise.clone()
            init_latent = init_latent * (1 - mask_expanded) + (low_guide + high_noise) * mask_expanded

        return init_latent
    def build_frame_attention_bias(self, selection_history, total_frames, batch_size):
        if not selection_history:
            return None

        frame_bias = torch.zeros(batch_size, total_frames, total_frames, device=self.device)
        has_penalty = False

        for t, data in selection_history.items():
            per_ref_warp = data.get('per_ref_warp', {})
            ref_indices = set(data.get('ref_indices', []))
            for s, warp_info in per_ref_warp.items():
                overlap_value = warp_info.get('overlap', 0.0)
                if s in ref_indices and overlap_value >= self.overlap_threshold:
                    continue
                frame_bias[:, t, s] = self.large_negative
                has_penalty = True

        if not has_penalty:
            return None

        return {
            'frame_bias': frame_bias,
            'temporal_length': total_frames
        }
