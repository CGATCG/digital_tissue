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


class TestTransportLayerOps(unittest.TestCase):
    def test_mass_conservation(self):
        H, W = 3, 3
        cell = np.ones((H, W), dtype=np.float32)
        mol = np.zeros((H, W), dtype=np.float32)
        mol[1, 1] = 10

        layers2d = {
            "cell": cell,
            "molecule_x": mol,
            "protein_north_exporter_x": np.full((H, W), 10, dtype=np.float32),
            "protein_south_exporter_x": np.full((H, W), 10, dtype=np.float32),
            "protein_east_exporter_x": np.full((H, W), 10, dtype=np.float32),
            "protein_west_exporter_x": np.full((H, W), 10, dtype=np.float32),
            "protein_north_importer_x": np.full((H, W), 10, dtype=np.float32),
            "protein_south_importer_x": np.full((H, W), 10, dtype=np.float32),
            "protein_east_importer_x": np.full((H, W), 10, dtype=np.float32),
            "protein_west_importer_x": np.full((H, W), 10, dtype=np.float32),
        }
        kinds = {k: "counts" for k in layers2d.keys()}

        steps = [
            {
                "type": "transport",
                "molecules": "molecule_*",
                "cell_layer": "cell",
                "cell_mode": "eq",
                "cell_value": 1,
                "protein_prefix": "protein_",
                "molecule_prefix": "molecule_",
                "dirs": ["north", "south", "east", "west"],
                "per_pair_rate": 1.0,
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        before = _read_layer(payload, "molecule_x")
        apply_layer_ops_inplace(payload)
        after = _read_layer(payload, "molecule_x")
        self.assertEqual(int(np.rint(before).sum()), int(np.rint(after).sum()))

    def test_boundary_no_leak(self):
        H, W = 2, 2
        cell = np.ones((H, W), dtype=np.float32)
        mol = np.zeros((H, W), dtype=np.float32)
        mol[0, 0] = 5

        layers2d = {
            "cell": cell,
            "molecule_x": mol,
            "protein_north_exporter_x": np.full((H, W), 10, dtype=np.float32),
            "protein_west_exporter_x": np.full((H, W), 10, dtype=np.float32),
            "protein_south_importer_x": np.full((H, W), 10, dtype=np.float32),
            "protein_east_importer_x": np.full((H, W), 10, dtype=np.float32),
        }
        kinds = {k: "counts" for k in layers2d.keys()}

        steps = [
            {
                "type": "transport",
                "molecules": ["molecule_x"],
                "cell_layer": "cell",
                "cell_mode": "eq",
                "cell_value": 1,
                "protein_prefix": "protein_",
                "molecule_prefix": "molecule_",
                "dirs": ["north", "west"],
                "per_pair_rate": 1.0,
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)
        after = _read_layer(payload, "molecule_x")
        self.assertEqual(int(np.rint(after).sum()), 5)
        self.assertEqual(int(np.rint(after[0, 0])), 5)

    def test_cell_to_noncell_uses_exporter_only(self):
        H, W = 1, 2
        cell = np.array([[1, 0]], dtype=np.float32)
        mol = np.array([[5, 0]], dtype=np.float32)

        layers2d = {
            "cell": cell,
            "molecule_x": mol,
            "protein_east_exporter_x": np.array([[3, 0]], dtype=np.float32),
        }
        kinds = {k: "counts" for k in layers2d.keys()}

        steps = [
            {
                "type": "transport",
                "molecules": ["molecule_x"],
                "cell_layer": "cell",
                "cell_mode": "eq",
                "cell_value": 1,
                "protein_prefix": "protein_",
                "molecule_prefix": "molecule_",
                "dirs": ["east"],
                "per_pair_rate": 1.0,
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)
        after = np.rint(_read_layer(payload, "molecule_x")).astype(int)
        self.assertEqual(after[0, 0], 2)
        self.assertEqual(after[0, 1], 3)

    def test_noncell_to_cell_uses_importer_only(self):
        H, W = 1, 2
        cell = np.array([[0, 1]], dtype=np.float32)
        mol = np.array([[5, 0]], dtype=np.float32)

        layers2d = {
            "cell": cell,
            "molecule_x": mol,
            "protein_west_importer_x": np.array([[0, 2]], dtype=np.float32),
        }
        kinds = {k: "counts" for k in layers2d.keys()}

        steps = [
            {
                "type": "transport",
                "molecules": ["molecule_x"],
                "cell_layer": "cell",
                "cell_mode": "eq",
                "cell_value": 1,
                "protein_prefix": "protein_",
                "molecule_prefix": "molecule_",
                "dirs": ["east"],
                "per_pair_rate": 1.0,
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)
        after = np.rint(_read_layer(payload, "molecule_x")).astype(int)
        self.assertEqual(after[0, 0], 3)
        self.assertEqual(after[0, 1], 2)

    def test_noncell_to_noncell_is_zero(self):
        H, W = 1, 2
        cell = np.array([[0, 0]], dtype=np.float32)
        mol = np.array([[5, 0]], dtype=np.float32)

        layers2d = {
            "cell": cell,
            "molecule_x": mol,
            "protein_east_exporter_x": np.array([[5, 0]], dtype=np.float32),
            "protein_west_importer_x": np.array([[0, 5]], dtype=np.float32),
        }
        kinds = {k: "counts" for k in layers2d.keys()}

        steps = [
            {
                "type": "transport",
                "molecules": ["molecule_x"],
                "cell_layer": "cell",
                "cell_mode": "eq",
                "cell_value": 1,
                "protein_prefix": "protein_",
                "molecule_prefix": "molecule_",
                "dirs": ["east"],
                "per_pair_rate": 1.0,
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)
        after = np.rint(_read_layer(payload, "molecule_x")).astype(int)
        self.assertEqual(after[0, 0], 5)
        self.assertEqual(after[0, 1], 0)


if __name__ == "__main__":
    unittest.main()
