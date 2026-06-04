import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_sentinel2_bands(X: torch.Tensor) -> torch.Tensor:
    """
    Resamples 20m Sentinel-2 bands to 10m resolution.

    Sentinel-2 provides bands at different spatial resolutions.
    This function first upsamples the 20m bands to 10m resolution using a
    two-step process: nearest-neighbor to 20m (to match spatial alignment), 
    followed by bilinear interpolation to 10m.

    Args:
        X (torch.Tensor): Input tensor of shape (B, C, H, W), where C includes 
                          all Sentinel-2 bands in a specific order.

    Returns:
        torch.Tensor: Resampled tensor of shape (B, 10, H, W), combining 10m-native 
                      and upsampled 20m bands.
    """
    # Indices of 20m and 10m bands
    indices_20m = [3, 4, 5, 7, 8, 9]  # B5, B6, B7, B8A, B11, B12
    indices_10m = [0, 1, 2, 6]        # B2, B3, B4, B8

    # Separate bands by resolution
    bands_20m = X[:, indices_20m]
    bands_10m = X[:, indices_10m]

    # Step 1: Downsample 20m bands to 10m pixel count (for alignment)
    bands_20m_down = F.interpolate(bands_20m, scale_factor=0.5, mode="nearest")

    # Step 2: Upsample to 10m using bilinear interpolation for smoothness
    bands_20m_up = F.interpolate(
        bands_20m_down, scale_factor=2, mode="bilinear", antialias=True
    )

    # Concatenate upsampled 20m bands with native 10m bands
    return torch.cat([bands_20m_up, bands_10m], dim=1)


def reconstruct_sentinel2_stack(
    b10m: torch.Tensor, b20m: torch.Tensor
) -> torch.Tensor:
    """
    Reconstructs a 10-band Sentinel-2-like stack from separate 10m and 20m sources.

    Args:
        b10m (torch.Tensor): Tensor of shape (B, 4, H, W) containing B2, B3, B4, B8.
        b20m (torch.Tensor): Tensor of shape (B, 6, H, W) containing B5, B6, B7, B8A, B11, B12.

    Returns:
        torch.Tensor: Tensor of shape (B, 10, H, W), mimicking the original band order.
    """
    return torch.stack(
        [
            b10m[:, 0],  # B2 (Blue)
            b10m[:, 1],  # B3 (Green)
            b10m[:, 2],  # B4 (Red)
            b20m[:, 0],  # B5 (Red Edge 1)
            b20m[:, 1],  # B6 (Red Edge 2)
            b20m[:, 2],  # B7 (Red Edge 3)
            b10m[:, 3],  # B8 (NIR)
            b20m[:, 3],  # B8A (Narrow NIR)
            b20m[:, 4],  # B11 (SWIR 1)
            b20m[:, 5],  # B12 (SWIR 2)
        ],
        dim=1,
    )


def srmodel(
    sr_model: nn.Module,
    hard_constraint: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """
    Wraps a super-resolution model with band-specific preprocessing and postprocessing.

    The function returns a composite model that:
    1. Resamples the input Sentinel-2 bands to a uniform 10m resolution.
    2. Applies a super-resolution model.
    3. Applies a hard constraint or refinement module.
    4. Reconstructs the final 10-band Sentinel-2 output stack.

    Args:
        sr_model (nn.Module): A super-resolution model that takes a 10-band input.
        hard_constraint (nn.Module): A postprocessing constraint module applied after SR.
        device (str): The device to move models to ("cpu" or "cuda").

    Returns:
        nn.Module: A callable model ready for inference.
    """
    sr_model.to(device)
    hard_constraint.to(device)
    hard_constraint.eval()
    for param in hard_constraint.parameters():
        param.requires_grad = False

    class SRModel(nn.Module):
        def __init__(self, sr_model: nn.Module, hard_constraint: nn.Module):
            super().__init__()
            self.sr_model = sr_model
            self.hard_constraint = hard_constraint

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass of the composite super-resolution model.

            Args:
                x (torch.Tensor): Input Sentinel-2 tensor of shape (B, 10, H, W).

            Returns:
                torch.Tensor: Super-resolved tensor of shape (B, 10, H, W).
            """
            # Extract original RGB + NIR (10m bands) for reconstruction
            rgbn = x[:, [0, 1, 2, 6]].clone()

            # Resample full input to 10m uniform resolution
            x_resampled = resample_sentinel2_bands(x)

            # Apply SR model
            sr_out = self.sr_model(x_resampled)
            
            # Results must be always positive
            sr_out = torch.clamp(sr_out, min=0.0)

            # Apply hard constraint/refinement
            sr_out = self.hard_constraint(x_resampled, sr_out)

            # Reconstruct full 10-band stack
            return reconstruct_sentinel2_stack(rgbn, sr_out)

    return SRModel(sr_model, hard_constraint)
