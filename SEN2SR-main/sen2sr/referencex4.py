import torch
from torch import nn


def srmodel(
    sr_model: nn.Module,
    f2_model: nn.Module,
    reference_model_x4: nn.Module,
    reference_model_hard_constraint_x4: nn.Module,
    device: str = "cpu",
) -> nn.Module:
    """
    Wraps a multi-stage super-resolution (SR) model for Sentinel-2 imagery.

    This wrapper performs a staged upsampling pipeline:
    1. Uses a 2x SR model (`f2_model`) to bring 20m bands to 10m.
    2. Applies a 4x SR model (`sr_model`) to upsample 10m RGBN bands to 2.5m.
    3. Uses another 4x fusion model (`reference_model_x4`) to enhance SWIR bands.
    4. A hard constraint model (`reference_model_hard_constraint_x4`) ensures spectral consistency.

    Args:
        sr_model (nn.Module): 4x SR model for RGBN bands.
        f2_model (nn.Module): 2x SR model to upsample 20m bands to 10m.
        reference_model_x4 (nn.Module): 4x SR model for SWIR fusion.
        reference_model_hard_constraint_x4 (nn.Module): Spectral constraint module for SWIR bands.
        device (str): Device to place models on ('cpu' or 'cuda').

    Returns:
        nn.Module: A callable module performing the full multi-band super-resolution.
    """
    
    # Move fusion models to the correct device
    reference_model_x4 = reference_model_x4.to(device)
    reference_model_hard_constraint_x4 = reference_model_hard_constraint_x4.to(device)
    reference_model_hard_constraint_x4.eval()
    for param in reference_model_hard_constraint_x4.parameters():
        param.requires_grad = False

    class SRModel(nn.Module):
        """
        Full Sentinel-2 super-resolution pipeline.
        """
        def __init__(
            self,
            sr_model: nn.Module,
            f2_model: nn.Module,
            f4_model: nn.Module,
            f4_hard_constraint: nn.Module,
        ):
            super().__init__()
            self.sr_model = sr_model                    # SR model for RGBN bands
            self.f2_model = f2_model                    # 2x SR model for initial upsampling
            self.f4_model = f4_model                    # Fusion model for SWIR bands
            self.f4_hard_constraint = f4_hard_constraint  # Spectral constraint for SWIR

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x (torch.Tensor): Input image tensor of shape (B, 10, H, W), 
                            where 10 Sentinel-2 bands are ordered as:
                            [B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12]

            Returns:
                torch.Tensor: Super-resolved image of shape (B, 10, 4*H, 4*W)
            """
            # Define band indices
            bands_20m = [3, 4, 5, 7, 8, 9]  # RSWIR: B5, B6, B7, B8A, B11, B12
            bands_10m = [2, 1, 0, 6]        # RGBN: B4, B3, B2, B8

            # Step 1: Upsample RSWIR bands from 20m to 10m using reference SR model
            allbands10m = self.f2_model(x)

            # Extract and upsample RSWIR (10m → 2.5m) using bilinear interpolation
            rsiwr_10m = allbands10m[:, bands_20m]
            rsiwr_2dot5m_bilinear = nn.functional.interpolate(
                rsiwr_10m, scale_factor=4, mode="bilinear", antialias=True
            )

            # Step 2: Super-resolve RGBN bands (10m → 2.5m) with learned model
            rgbn_input = x[:, bands_10m]
            rgbn_2dot5m = self.sr_model(rgbn_input)

            # Reorder from RGBN → BGRN (e.g., for consistency with downstream expectations)
            rgbn_2dot5m = rgbn_2dot5m[:, [2, 1, 0, 3]]

            # Step 3: Apply fusion model to enhance RSWIR bands (10m → 2.5m)
            fusion_input = torch.cat([rsiwr_2dot5m_bilinear, rgbn_2dot5m], dim=1)
            rswirs_2dot5 = self.f4_model(fusion_input)

            # Results must be always positive
            rswirs_2dot5 = torch.clamp(rswirs_2dot5, min=0.0)

            # Step 4: Apply hard constraint model to ensure spectral consistency
            rswirs_2dot5 = self.f4_hard_constraint(rsiwr_10m, rswirs_2dot5)

            # Final step: Reconstruct full band stack in Sentinel-2 order
            return torch.stack([
                rgbn_2dot5m[:, 0],  # B2 (Blue)
                rgbn_2dot5m[:, 1],  # B3 (Green)
                rgbn_2dot5m[:, 2],  # B4 (Red)
                rswirs_2dot5[:, 0], # B5 (Red Edge 1)
                rswirs_2dot5[:, 1], # B6 (Red Edge 2)
                rswirs_2dot5[:, 2], # B7 (Red Edge 3)
                rgbn_2dot5m[:, 3],  # B8 (NIR)
                rswirs_2dot5[:, 3], # B8A (Narrow NIR)
                rswirs_2dot5[:, 4], # B11 (SWIR 1)
                rswirs_2dot5[:, 5], # B12 (SWIR 2)
            ], dim=1)

    return SRModel(sr_model, f2_model, reference_model_x4, reference_model_hard_constraint_x4)
