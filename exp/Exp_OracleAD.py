import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict

from model.OracleAD import OracleAD

@dataclass
class OracleADLossConfig:
    lambda_recon: float = 0.1
    lambda_dev: float = 3.0
    use_sls_from_epoch: int = 2

def oraclead_losses(
    x: torch.Tensor,
    xhat_hist: torch.Tensor,
    xhat_last: torch.Tensor,
    D: torch.Tensor,
    sls: Optional[torch.Tensor],
    cfg: OracleADLossConfig,
    epoch_idx: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute training losses: Prediction, Reconstruction, and Deviation.
    """
    x_hist = x[:, : xhat_hist.shape[1], :]
    x_last = x[:, -1, :]

    pred_loss = F.l1_loss(xhat_last, x_last)
    recon_loss = F.mse_loss(xhat_hist, x_hist)

    dev_loss = torch.tensor(0.0, device=x.device)
    if sls is not None and epoch_idx >= cfg.use_sls_from_epoch:
        dev_loss = F.mse_loss(D, sls.unsqueeze(0).to(D.device, D.dtype))

    total = pred_loss + cfg.lambda_recon * recon_loss + cfg.lambda_dev * dev_loss
    return {"total": total, "pred": pred_loss, "recon": recon_loss, "dev": dev_loss}

@torch.no_grad()
def oraclead_scores(
    x: torch.Tensor,
    xhat_last: torch.Tensor,
    D: torch.Tensor,
    sls: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute anomaly scores during inference.
    Returns: P (Pred err), D (Deviation), A (Anomaly score), rc (Root Cause).
    """
    x_last = x[:, -1, :]
    P_score = torch.mean(torch.abs(x_last - xhat_last), dim=1)

    dev_mat = torch.abs(D - sls.unsqueeze(0).to(D.device, D.dtype))
    D_score = torch.linalg.norm(dev_mat, ord="fro", dim=(1, 2))

    A_score = P_score * D_score
    rc_score = dev_mat.sum(dim=2) # Row-sum as root cause proxy

    return {"P": P_score, "D": D_score, "A": A_score, "rc": rc_score}

def train_oraclead(
    model: OracleAD,
    train_loader,
    num_epochs: int = 10,
    lr: float = 5e-4,
    weight_decay: float = 1e-2,
    loss_cfg: OracleADLossConfig = OracleADLossConfig(),
    device: str = "cuda",
) -> torch.Tensor:
    """
    Main training loop. Returns the learned SLS matrix [N, N].
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    sls = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        
        D_sum = torch.zeros(model.N, model.N, device=device)
        D_count = 0
        meters = {"total": 0.0, "pred": 0.0, "recon": 0.0, "dev": 0.0}
        num_batches = 0

        for x in train_loader:
            x = x.to(device) # [B, L, N]

            out = model(x)
            D = model.distance_matrix(out["c_star"])

            losses = oraclead_losses(
                x=x,
                xhat_hist=out["xhat_hist"],
                xhat_last=out["xhat_last"],
                D=D,
                sls=sls,
                cfg=loss_cfg,
                epoch_idx=epoch,
            )

            opt.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()

            # Logging & SLS Accumulation
            for k in meters:
                meters[k] += float(losses[k].detach().cpu())
            num_batches += 1
            
            D_sum += D.detach().sum(dim=0)
            D_count += D.shape[0]

        # End of epoch: update SLS
        sls = (D_sum / max(D_count, 1)).detach()

        # Print progress
        for k in meters:
            meters[k] /= max(num_batches, 1)
        print(f"[Epoch {epoch:03d}] Total={meters['total']:.5f} | Pred={meters['pred']:.5f} | Dev={meters['dev']:.5f}")

    return sls