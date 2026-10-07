"""Batched 2D Aliev-Panfilov tissue with insulating scar, and the pacing protocol used for labels.

Scar cells (conductivity 0) are inert: face conductances use the harmonic mean, so no current crosses a
face that touches scar, and the reaction terms are switched off there. Fibres run along x.
"""
from __future__ import annotations

import math

import torch
from cardiograd.model import AlievPanfilov


def face_conductances(c: torch.Tensor, d_long: float = 1.0, d_trans: float = 0.5):
    """Harmonic-mean conductance on x- and y-faces; zero wherever either neighbour is scar."""
    def hm(a, b):
        return 2 * a * b / (a + b).clamp_min(1e-12)
    return d_long * hm(c[..., :, 1:], c[..., :, :-1]), d_trans * hm(c[..., 1:, :], c[..., :-1, :])


def diffusion(u, cx, cy, dx: float = 1.0):
    jx = torch.nn.functional.pad(cx * (u[..., :, 1:] - u[..., :, :-1]) / dx, (1, 1))
    jy = torch.nn.functional.pad(cy * (u[..., 1:, :] - u[..., :-1, :]) / dx, (0, 0, 1, 1))
    return (jx[..., :, 1:] - jx[..., :, :-1]) / dx + (jy[..., 1:, :] - jy[..., :-1, :]) / dx


def disc(shape, centre, radius, device=None):
    yy, xx = torch.meshgrid(torch.arange(shape[0], device=device), torch.arange(shape[1], device=device), indexing="ij")
    return ((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) <= radius ** 2


@torch.no_grad()
def run_protocol(c: torch.Tensor, intervals: list[float], *, dt: float = 0.05, tail: float = 450.0,
                 window: float = 150.0, s1_width: int = 3, s2_radius: float = 4.0, s2_shift: int = 0, threshold: float = 0.5,
                 sustained_fraction: float = 0.01, model: AlievPanfilov = AlievPanfilov()) -> dict:
    """S1 plane wave from the left edge at t = 0, then a small S2 disc at the same edge after each interval.

    c: (B, H, W) conductivity in [0, 1], 0 = scar. Every tissue is run once per interval in one batch.
    Returns per tissue: inducible (any interval sustained), per-interval sustained flags, the re-entry map
    (fraction of the final window each cell spends depolarised, for the first inducing interval) and the
    S1 activation-time map (NaN where S1 never arrived before the earliest S2).
    """
    B, H, W = c.shape
    K = len(intervals)
    dev = c.device
    cc = c.repeat_interleave(K, 0)
    alive = cc > 0
    cx, cy = face_conductances(cc)
    t_s2 = torch.tensor(intervals, device=dev, dtype=torch.float32).repeat(B)
    s1 = torch.zeros(H, W, dtype=torch.bool, device=dev)
    s1[:, :s1_width] = True
    s2 = disc((H, W), (H // 2 + s2_shift, s1_width), s2_radius, dev)
    u = torch.zeros_like(cc)
    v = torch.zeros_like(cc)
    u[:, s1] = 1.0
    u = u * alive
    first = torch.full_like(cc, math.inf)
    n_alive = alive.flatten(1).sum(1).clamp_min(1)
    win_start, win_end = t_s2 + tail - window, t_s2 + tail
    active_time = torch.zeros_like(cc)
    min_frac = torch.full((B * K,), math.inf, device=dev)
    n_steps = int(round((max(intervals) + tail) / dt))
    s2_step = torch.round(t_s2 / dt).long()
    sample = int(round(1.0 / dt))
    for i in range(n_steps):
        hit = s2_step == i
        if bool(hit.any()):
            u[hit] = torch.where(s2 & alive[hit], torch.ones_like(u[hit]), u[hit])
        du, dv = model.reaction(u, v)
        u = (u + dt * (diffusion(u, cx, cy) + du)) * alive
        v = (v + dt * dv) * alive
        if i % sample == 0:
            t = (i + 1) * dt
            dep = u > threshold
            first = torch.where(dep & torch.isinf(first), torch.full_like(first, t), first)
            inwin = (t >= win_start) & (t <= win_end)
            if bool(inwin.any()):
                frac = dep.flatten(1).sum(1) / n_alive
                min_frac = torch.where(inwin, torch.minimum(min_frac, frac), min_frac)
                active_time += dep * inwin[:, None, None]
    sustained = (min_frac > sustained_fraction).view(B, K)
    act = first.view(B, K, H, W)[:, 0]
    act = torch.where(act < min(intervals), act, torch.full_like(act, math.nan))
    reentry = (active_time / max(window, 1.0)).view(B, K, H, W)
    idx = torch.where(sustained.any(1), sustained.float().argmax(1), torch.zeros(B, dtype=torch.long, device=dev))
    re_map = reentry[torch.arange(B, device=dev), idx] * sustained.any(1)[:, None, None]
    return {"inducible": sustained.any(1), "sustained": sustained, "reentry_map": re_map, "s1_act": act}
