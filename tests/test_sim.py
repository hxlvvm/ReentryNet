import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reentrynet.sim import AlievPanfilov, diffusion, face_conductances  # noqa: E402


def propagate(c, t_end=90.0, dt=0.05):
    model = AlievPanfilov()
    alive = c > 0
    cx, cy = face_conductances(c)
    u = torch.zeros_like(c)
    u[..., :3] = 1.0
    v = torch.zeros_like(c)
    reached = torch.zeros_like(c, dtype=torch.bool)
    for _ in range(int(t_end / dt)):
        du, dv = model.reaction(u, v)
        u = (u + dt * (diffusion(u, cx, cy) + du)) * alive
        v = (v + dt * dv) * alive
        reached |= u > 0.5
    return reached


def test_one_cell_scar_line_blocks():
    c = torch.ones(1, 24, 60)
    c[..., :, 30] = 0.0
    reached = propagate(c)
    assert reached[..., :, 20].all() and not reached[..., :, 40:].any()


def test_one_cell_gap_lets_wave_through():
    c = torch.ones(1, 24, 60)
    c[..., :, 30] = 0.0
    c[..., 12, 30] = 1.0
    assert propagate(c, t_end=140.0)[..., :, 55].any()


def test_scar_faces_carry_no_current():
    c = torch.ones(1, 4, 4)
    c[..., :, 2] = 0.0
    cx, _ = face_conductances(c)
    assert (cx[..., :, 1] == 0).all() and (cx[..., :, 2] == 0).all() and (cx[..., :, 0] > 0).all()
