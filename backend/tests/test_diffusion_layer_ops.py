import base64
import unittest

import numpy as np

from apply_layer_ops import apply_layer_ops_inplace


def _b64_f32(a: np.ndarray) -> str:
    return base64.b64encode(np.asarray(a, dtype=np.float32).tobytes()).decode("ascii")


def _make_payload(H: int, W: int, layers2d: dict, kinds: dict, steps: list) -> dict:
    out = {
        "version": 1,
        "H": int(H),
        "W": int(W),
        "layers": [],
        "data": {},
        "layer_ops_config": {"version": 2, "steps": steps},
    }
    for name, arr2d in layers2d.items():
        kind = str(kinds.get(name, "continuous"))
        out["layers"].append({"name": str(name), "kind": kind})
        out["data"][str(name)] = {
            "dtype": "float32",
            "b64": _b64_f32(np.asarray(arr2d, dtype=np.float32).reshape(H * W)),
        }
    return out


def _read_layer(payload: dict, name: str) -> np.ndarray:
    H = int(payload["H"])
    W = int(payload["W"])
    b64 = payload["data"][name]["b64"]
    raw = base64.b64decode(b64)
    a = np.frombuffer(raw, dtype=np.float32)
    return a.reshape(H, W)


class TestDiffusionLayerOps(unittest.TestCase):
    def test_noncell_diffuses_and_conserves_mass(self):
        H, W = 1, 3
        cell = np.array([[0, 0, 0]], dtype=np.float32)
        mol = np.array([[10, 0, 0]], dtype=np.float32)

        layers2d = {"cell": cell, "molecule_x": mol}
        kinds = {"cell": "categorical", "molecule_x": "counts"}
        steps = [
            {
                "type": "diffusion",
                "molecules": ["molecule_x"],
                "cell_layer": "cell",
                "cell_mode": "eq",
                "cell_value": 1,
                "rate": 1.0,
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        before = np.rint(_read_layer(payload, "molecule_x")).astype(int)
        apply_layer_ops_inplace(payload)
        after = np.rint(_read_layer(payload, "molecule_x")).astype(int)

        self.assertEqual(int(before.sum()), int(after.sum()))
        self.assertTrue(after[0, 1] > 0 or after[0, 2] > 0)

    def test_cells_block_diffusion(self):
        H, W = 1, 3
        cell = np.array([[0, 1, 0]], dtype=np.float32)
        mol = np.array([[10, 0, 0]], dtype=np.float32)

        layers2d = {"cell": cell, "molecule_x": mol}
        kinds = {"cell": "categorical", "molecule_x": "counts"}
        steps = [
            {
                "type": "diffusion",
                "molecules": ["molecule_x"],
                "cell_layer": "cell",
                "cell_mode": "eq",
                "cell_value": 1,
                "rate": 1.0,
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)
        after = np.rint(_read_layer(payload, "molecule_x")).astype(int)

        self.assertEqual(after[0, 2], 0)
        self.assertEqual(int(after.sum()), 10)

    def test_rate_layer_controls_where_diffusion_happens(self):
        H, W = 1, 4
        cell = np.array([[0, 0, 0, 0]], dtype=np.float32)
        mol = np.array([[10, 0, 10, 0]], dtype=np.float32)
        rate_layer = np.array([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)

        layers2d = {"cell": cell, "molecule_x": mol, "diff_rate": rate_layer}
        kinds = {"cell": "categorical", "molecule_x": "counts", "diff_rate": "continuous"}
        steps = [
            {
                "type": "diffusion",
                "molecules": ["molecule_x"],
                "cell_layer": "cell",
                "cell_mode": "eq",
                "cell_value": 1,
                "rate": 1.0,
                "rate_layer": "diff_rate",
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)
        after = np.rint(_read_layer(payload, "molecule_x")).astype(int)

        # Left pair should exchange; right pair should not change because rate is 0 there.
        self.assertEqual(int(after[:, 2:].sum()), 10)


if __name__ == "__main__":
    unittest.main()
