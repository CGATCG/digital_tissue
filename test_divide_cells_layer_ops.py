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


class TestDivideCellsLayerOps(unittest.TestCase):
    def test_divide_cells_splits_and_places_nearest_empty(self):
        H, W = 3, 3
        cell = np.array(
            [
                [2, 2, 2],
                [2, 1, 0],
                [2, 2, 2],
            ],
            dtype=np.float32,
        )
        divider = np.zeros((H, W), dtype=np.float32)
        divider[1, 1] = 60

        molecule_x = np.zeros((H, W), dtype=np.float32)
        molecule_x[1, 1] = 10

        protein_a = np.zeros((H, W), dtype=np.float32)
        protein_a[1, 1] = 7

        gene_y = np.zeros((H, W), dtype=np.float32)
        gene_y[1, 1] = 3

        layers2d = {
            "cell": cell,
            "protein_divider": divider,
            "molecule_x": molecule_x,
            "protein_a": protein_a,
            "gene_y": gene_y,
        }
        kinds = {
            "cell": "categorical",
            "protein_divider": "continuous",
            "molecule_x": "counts",
            "protein_a": "continuous",
            "gene_y": "counts",
        }

        steps = [
            {
                "type": "divide_cells",
                "cell_layer": "cell",
                "cell_value": 1,
                "empty_value": 0,
                "trigger_layer": "protein_divider",
                "threshold": 50,
                "split_fraction": 0.5,
                "max_radius": 10,
                "layer_prefixes": ["molecule", "protein", "rna", "damage", "gene"],
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)

        cell_out = np.rint(_read_layer(payload, "cell")).astype(int)
        self.assertEqual(cell_out[1, 1], 1)
        self.assertEqual(cell_out[1, 2], 1)

        mol = np.rint(_read_layer(payload, "molecule_x")).astype(int)
        self.assertEqual(mol[1, 1], 5)
        self.assertEqual(mol[1, 2], 5)

        prot = _read_layer(payload, "protein_a")
        self.assertAlmostEqual(float(prot[1, 1]), 3.5, places=5)
        self.assertAlmostEqual(float(prot[1, 2]), 3.5, places=5)

        gene = np.rint(_read_layer(payload, "gene_y")).astype(int)
        self.assertEqual(gene[1, 1], 2)
        self.assertEqual(gene[1, 2], 1)

    def test_divide_cells_conflict_closest_wins(self):
        H, W = 3, 3
        cell = np.array(
            [
                [1, 2, 2],
                [1, 0, 2],
                [2, 2, 2],
            ],
            dtype=np.float32,
        )
        divider = np.zeros((H, W), dtype=np.float32)
        divider[0, 0] = 60
        divider[1, 0] = 60

        molecule_x = np.zeros((H, W), dtype=np.float32)
        molecule_x[0, 0] = 20
        molecule_x[1, 0] = 10

        layers2d = {
            "cell": cell,
            "protein_divider": divider,
            "molecule_x": molecule_x,
        }
        kinds = {
            "cell": "categorical",
            "protein_divider": "continuous",
            "molecule_x": "counts",
        }

        steps = [
            {
                "type": "divide_cells",
                "cell_layer": "cell",
                "cell_value": 1,
                "empty_value": 0,
                "trigger_layer": "protein_divider",
                "threshold": 50,
                "split_fraction": 0.5,
                "max_radius": 10,
                "layer_prefixes": ["molecule", "protein", "rna", "damage", "gene"],
                "seed": 0,
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)

        cell_out = np.rint(_read_layer(payload, "cell")).astype(int)
        self.assertEqual(cell_out[1, 1], 1)

        mol = np.rint(_read_layer(payload, "molecule_x")).astype(int)
        # (1,0) is closer to (1,1) than (0,0), so it should divide.
        self.assertEqual(mol[1, 1], 5)
        self.assertEqual(mol[1, 0], 5)
        self.assertEqual(mol[0, 0], 20)


if __name__ == "__main__":
    unittest.main()
