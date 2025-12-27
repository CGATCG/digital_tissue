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


class TestCategoricalLayerOps(unittest.TestCase):
    def test_categorical_overwrite_snaps_to_existing_values(self):
        H, W = 1, 5
        cat = np.array([[0, 0, 2, 2, 2]], dtype=np.float32)
        x = np.array([[0.0, 0.5, 1.0, 1.5, 2.0]], dtype=np.float32)

        layers2d = {"cell_type": cat, "x": x}
        kinds = {"cell_type": "categorical", "x": "continuous"}
        steps = [
            {
                "type": "op",
                "target": "cell_type",
                "expr": "10*x",  # produces 0..20, should snap to {0,2}
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)
        out = np.rint(_read_layer(payload, "cell_type")).astype(int)
        self.assertTrue(set(np.unique(out)).issubset({0, 2}))

    def test_categorical_overwrite_respects_allowed_values(self):
        H, W = 1, 4
        cat = np.array([[0, 0, 0, 0]], dtype=np.float32)
        x = np.array([[0.0, 0.4, 0.6, 1.0]], dtype=np.float32)

        layers2d = {"cell_type": cat, "x": x}
        kinds = {"cell_type": "categorical", "x": "continuous"}
        steps = [
            {
                "type": "op",
                "target": "cell_type",
                "expr": "3*x",  # 0..3
                "allowed_values": [0, 1, 3],
            }
        ]

        payload = _make_payload(H, W, layers2d, kinds, steps)
        apply_layer_ops_inplace(payload)
        out = np.rint(_read_layer(payload, "cell_type")).astype(int)
        self.assertTrue(set(np.unique(out)).issubset({0, 1, 3}))


if __name__ == "__main__":
    unittest.main()
