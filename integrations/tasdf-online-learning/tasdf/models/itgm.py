import torch
import torch.nn as nn



class ITGM(nn.Module):
    """Implicit Topology Guidance Module"""

    def __init__(self, iterations: int = 10, kernel_size: int = 3, erosion: str = "square"):
        super(ITGM, self).__init__()
        if erosion not in ("square", "cross"):
            raise ValueError(f"erosion must be 'square' or 'cross', got {erosion}")
        self.iterations = iterations
        self.erosion = erosion
        padding = kernel_size // 2
        self.pool_v = nn.MaxPool2d(kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0))
        self.pool_h = nn.MaxPool2d(kernel_size=(1, kernel_size), stride=1, padding=(0, padding))

        #max pool

        self.maxpool = nn.MaxPool2d(
            kernel_size = kernel_size, stride=1, padding=padding
        )

    def minpool(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable erosion, min(x) = -max(-x).

        'cross' reproduces Fig. 2(b) and the reference clDice soft_erode:
        min(minpool_3x1, minpool_1x3). 'square' is the legacy 3x3 erosion.
        """
        if self.erosion == "cross":
            return torch.min(-self.pool_v(-x), -self.pool_h(-x))
        return -self.maxpool(-x)

    def forward(self, x:torch.Tensor, return_intermediates: bool = False):
        """ Extract soft-skeleton through loop self.iteractions 
            Args:
                x: Input Tensor (B,C,H,W) values [0,1]
            Returns:
                skeleton: soft skeleton maps with (B,C,H,W)
                (options) History: List of states across each iteractions
        
        """
        history = []

        #Init soft skeleton 
        f = x
        f_prime = self.maxpool(self.minpool(f))

        s_0 = torch.relu(f - f_prime)
        skeleton = s_0

        if return_intermediates:
            history.append(
                {
                    "step": 0,
                    "f_eroded": f.clone(),
                    "s_residual": s_0.clone(),
                    "accumulated_skeleton": skeleton.clone()
                }
            )


        for i in range(self.iterations):
            f = self.minpool(f)
            f_prime = self.maxpool(self.minpool(f))
            s_i = torch.relu(f - f_prime)

            skeleton = skeleton + (1.0 - skeleton) * s_i
            if return_intermediates:
                history.append(
                {
                    "step": i+1,
                    "f_eroded": f.clone(),
                    "s_residual": s_i.clone(),
                    "accumulated_skeleton": skeleton.clone()
                }
            )

        if return_intermediates:
            return skeleton, history
        return skeleton



