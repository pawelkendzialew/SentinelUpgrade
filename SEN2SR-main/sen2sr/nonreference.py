import torch
from torch import nn


def srmodel(
    sr_model: nn.Module,
    hard_constraint: nn.Module,
    device: str = "cpu"
) -> nn.Module:
    """
    Wraps a super-resolution (SR) model with a hard constraint module to enforce 
    physical consistency.

    Parameters
    ----------
    sr_model : nn.Module
        The base super-resolution model to be applied on the input tensor.
    
    hard_constraint : nn.Module
        A non-trainable constraint module that adjusts the output of the SR model
        based on prior knowledge or application-specific rules.
    
    device : str, optional
        Target device for model execution (e.g., "cpu" or "cuda"), by default "cpu".

    Returns
    -------
    nn.Module
        A composite model that applies the SR model and enforces the hard constraint during the forward pass.
    """

    # Move the SR model to the target device
    sr_model = sr_model.to(device)

    # Prepare the hard constraint module: evaluation mode, no gradients, moved to device
    hard_constraint = hard_constraint.eval()
    for param in hard_constraint.parameters():
        param.requires_grad = False
    hard_constraint = hard_constraint.to(device)

    class SRModelWithConstraint(nn.Module):
        """
        Composite model applying SR followed by a hard constraint module.
        """
        def __init__(self, sr_model: nn.Module, hard_constraint: nn.Module):
            super().__init__()
            self.sr_model = sr_model
            self.hard_constraint = hard_constraint

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass: apply SR model, then enforce hard constraint.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor representing low-resolution imagery.

            Returns
            -------
            torch.Tensor
                Super-resolved and constraint-corrected output.
            """
            
            # Apply SR model
            sr = self.sr_model(x)

            # Results must be always positive
            sr = torch.clamp(sr, min=0.0)

            return self.hard_constraint(x, sr)

    return SRModelWithConstraint(sr_model, hard_constraint)
