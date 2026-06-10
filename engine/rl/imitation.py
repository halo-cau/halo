"""Imitation learning for the layout policy (behavioral cloning from expert demonstrations).

Given one or more demonstrated layouts -- expert placements on a room -- this fits the policy to reproduce
them. A demonstration decomposes into the macro actions that build it (place the cooling unit, the rack
rows, the network rack), and the policy is trained by supervised cross-entropy to emit those actions. With
demonstrations spanning many rooms this is standard imitation learning; with a single demonstration it is a
fast supervised warm start for the reinforcement-learning policy.

Pipeline:
  1. Load a demonstration run (``voxel_grid.npy`` + ``placements.json``); derive the room (obstacle +
     ceiling) with ``twin_to_rl_input`` after stripping the movable equipment the policy will place.
  2. Convert the demonstrated AC / rows / network rack into the ``MacroPlacementEnv`` action indices.
  3. Replay them to collect the (observation, action) pairs.
  4. Supervised cross-entropy on ``SpatialMaskablePolicy`` until it matches the demonstration.
  5. Verify a deterministic rollout emits the demonstrated actions and rebuilds the layout; save the model.

Run: ``python -m engine.rl.imitation --run tools/recon_web/runs/<id> [--epochs 400] [--out model.zip]``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from engine.rl.macro_env import CH_AC, CH_NETRACK, CH_ROW0, MacroPlacementEnv
from engine.rl.twin_bridge import CELL_M, twin_to_rl_input

# Movable equipment the policy places -> stripped from the grid to recover the empty room shell.
_MOVABLE = (3, 5, 6, 7, 8, 12)
# Intake facing -> env exhaust-direction index (exhaust is opposite the intake).
_FACING_TO_DIR = {"MINUS_X": 0, "PLUS_X": 1, "MINUS_Y": 2, "PLUS_Y": 3}


def _cell(x_m: float, y_m: float) -> tuple[int, int]:
    return int(round(x_m / CELL_M)), int(round(y_m / CELL_M))


def build_demonstration(run_dir: Path):
    """Return (obstacle, ceiling_m, actions, expected, racks_per_row) for a demonstration run.

    ``actions`` is the ordered list of macro action indices that build the demonstrated layout; ``expected``
    records the entities for verification.
    """
    grid = np.load(run_dir / "voxel_grid.npy")
    manifest = json.loads((run_dir / "placements.json").read_text())

    empty = grid.copy()
    empty[np.isin(empty, _MOVABLE)] = 0
    rl = twin_to_rl_input(empty, manifest, rack_num=12)
    obstacle, ceiling_m = rl["obstacle"], rl["ceiling_m"]

    insts = manifest["instances"]
    racks = [p for p in insts if p.get("kind") == "rack"]
    acs = [p for p in insts if str(p.get("name", "")).startswith("ac_unit") or p.get("vox_id") == 3]
    nets = [p for p in insts if p.get("name") == "network rack" or p.get("vox_id") == 8]

    # Group racks into rows by facing.
    rows: dict[str, list] = {}
    for p in racks:
        rows.setdefault(p["facing"], []).append(p)

    g = obstacle.shape[0]

    def enc(gx, gy, ch):
        return int((gx * g + gy) * 6 + ch)

    actions: list[int] = []
    expected: dict = {"rows": [], "ac": None, "netrack": None}

    # 1. AC (cooling unit) -- its box center maps to a vent cell.
    ac = acs[0]
    acx, acy = _cell(ac["center"][0], ac["center"][1])
    actions.append(enc(acx, acy, CH_AC))
    expected["ac"] = (acx, acy)

    # 2-3. Rows, ordered front (smaller y) first. Anchor = the row's left-most rack's intake-face cell.
    for facing, ps in sorted(rows.items(), key=lambda kv: min(r["pos"][1] for r in kv[1])):
        d = _FACING_TO_DIR[facing]
        anchor = min(ps, key=lambda r: r["pos"][0])           # left-most along X (rows span X here)
        ax, ay = _cell(anchor["pos"][0], anchor["pos"][1])
        actions.append(enc(ax, ay, CH_ROW0 + d))
        expected["rows"].append({"facing": facing, "dir": d, "anchor": (ax, ay), "n": len(ps)})

    # 4. Network rack.
    net = nets[0]
    nx, ny = _cell(net["center"][0], net["center"][1])
    actions.append(enc(nx, ny, CH_NETRACK))
    expected["netrack"] = (nx, ny)

    racks_per_row = max(len(ps) for ps in rows.values())
    return obstacle, ceiling_m, actions, expected, racks_per_row


def collect_dataset(obstacle, ceiling_m, actions, racks_per_row):
    """Replay the demonstrated actions to gather the (obs-before-action, action) pairs."""
    env = MacroPlacementEnv(grid_size=obstacle.shape[0], racks_per_row=racks_per_row, ceiling_m=ceiling_m)
    opts = {"obstacle": obstacle, "rack_num": racks_per_row * 2, "ceiling_m": ceiling_m}
    obs, _ = env.reset(options=opts)
    states = []
    for a in actions:
        states.append(obs.copy())
        obs, _, done, _, info = env.step(a)
    return np.stack(states), env, opts, info


def clone(env, states, actions, epochs: int, lr: float):
    """Supervised cross-entropy fit of SpatialMaskablePolicy to the (state, action) pairs."""
    import torch as th
    from sb3_contrib import MaskablePPO

    from engine.rl.policy import SpatialMaskablePolicy

    model = MaskablePPO(SpatialMaskablePolicy, env, n_steps=8, batch_size=4, device="cpu")
    policy = model.policy
    obs_t = th.as_tensor(states).float()
    act_t = th.as_tensor(np.asarray(actions)).long()
    mask_t = th.ones((len(actions), env.action_space.n))
    opt = th.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        dist = policy.get_distribution(obs_t, action_masks=mask_t)
        loss = -dist.log_prob(act_t).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep == 0 or (ep + 1) % max(1, epochs // 8) == 0:
            with th.no_grad():
                pred = dist.distribution.probs.argmax(-1).tolist()
            ok = sum(int(p == a) for p, a in zip(pred, actions, strict=False))
            print(f"  epoch {ep + 1:4d}  loss={loss.item():.4f}  matched {ok}/{len(actions)}")
    return model


def verify(model, obstacle, ceiling_m, actions, expected, racks_per_row, opts):
    """Deterministic rollout: confirm the policy emits the demonstrated actions and rebuilds the layout."""
    env = MacroPlacementEnv(grid_size=obstacle.shape[0], racks_per_row=racks_per_row, ceiling_m=ceiling_m)
    obs, _ = env.reset(options=opts)
    pred, info = [], {}
    for _ in range(len(actions)):
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        pred.append(int(a))
        obs, _, done, _, info = env.step(int(a))

    match = pred == list(actions)
    print(f"\nactions predicted   : {pred}")
    print(f"actions demonstrated: {list(actions)}")
    print(f"EXACT ACTION MATCH  : {match}")
    print(f"rebuilt layout: {info.get('n_racks')} server racks, {info.get('n_cool')} AC, "
          f"{info.get('n_netrack')} network rack")
    expect_racks = racks_per_row * len(expected["rows"])
    layout_ok = (info.get("n_racks") == expect_racks and info.get("n_cool") == 1
                 and info.get("n_netrack") == 1)
    print(f"layout matches demonstration counts ({expect_racks} racks + 1 AC + 1 net): {layout_ok}")
    return match and layout_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="demonstration twin run dir")
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default=None, help="optional path to save the cloned model.zip")
    args = ap.parse_args()

    run_dir = Path(args.run)
    obstacle, ceiling_m, actions, expected, rpr = build_demonstration(run_dir)
    print(f"demonstration: {len(actions)} macro actions | racks_per_row={rpr} | ceiling={ceiling_m:.2f}m")
    print(f"  AC cell      {expected['ac']}")
    for r in expected["rows"]:
        print(f"  row {r['facing']:7s} dir={r['dir']} anchor={r['anchor']} ({r['n']} racks)")
    print(f"  netrack cell {expected['netrack']}")

    states, env, opts, _ = collect_dataset(obstacle, ceiling_m, actions, rpr)
    print(f"\nfitting on {len(states)} (state, action) pairs ...")
    model = clone(env, states, actions, args.epochs, args.lr)
    ok = verify(model, obstacle, ceiling_m, actions, expected, rpr, opts)
    print(f"\nRESULT: {'SUCCESS -- policy reproduces the demonstration' if ok else 'not matched yet'}")
    if args.out and ok:
        model.save(args.out)
        print(f"saved model -> {args.out}")


if __name__ == "__main__":
    main()
