import argparse
import json
import os

import torch

from pilgrim import Pilgrim


def build_model(group_id, target_id, model_id, epoch, device="cpu"):
    """Rebuild a trained Pilgrim from its training log and weights, as test.py does."""
    name = f"p{int(group_id):03d}-t{int(target_id):03d}"

    with open(f"logs/model_{name}_{model_id}.json") as f:
        info = json.load(f)

    V0 = torch.load(f"targets/{name}.pt", weights_only=True, map_location=device)

    model = Pilgrim(
        num_classes=torch.unique(V0).numel(),
        state_size=V0.numel(),
        hd1=info["hd1"],
        hd2=info["hd2"],
        nrd=info["nrd"],
        dropout_rate=info.get("dropout", 0.0),
    )

    weights_path = f"weights/{name}_{model_id}_e{int(epoch):05d}.pth"
    model.load_state_dict(torch.load(weights_path, weights_only=False, map_location=device), strict=True)

    # The exported graph must stay float32: onnxruntime-web in the UI runs fp32.
    model.dtype = torch.float32
    if V0.min() < 0:
        model.z_add = -V0.min().item()

    model.eval().to(device)
    return model, V0, name, weights_path


def check_parity(onnx_path, model, V0, num_states=64):
    """Compare torch and onnxruntime predictions on random states."""
    try:
        import onnxruntime
    except ImportError:
        print("  parity check skipped (onnxruntime not installed)")
        return None

    torch.manual_seed(0)
    states = torch.stack([V0[torch.randperm(V0.numel())] for _ in range(num_states)])

    with torch.no_grad():
        expected = model(states).numpy()

    session = onnxruntime.InferenceSession(onnx_path)
    actual = session.run(["output"], {"input": states.numpy().astype("int64")})[0]

    max_diff = float(abs(expected - actual.reshape(expected.shape)).max())
    print(f"  parity vs onnxruntime: max abs diff {max_diff:.3e} over {num_states} states")
    return max_diff


def main():
    parser = argparse.ArgumentParser(description="Export a trained Pilgrim model to ONNX")
    parser.add_argument("--group_id", type=int, required=True, help="Group ID")
    parser.add_argument("--target_id", type=int, default=0, help="Target ID")
    parser.add_argument("--model_id", type=str, required=True, help="Model ID from logs/model_id.txt")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number to load")
    parser.add_argument("--out", type=str, default="", help="Output .onnx path (default: onnx/<name>_<id>_e<epoch>.onnx)")
    parser.add_argument("--moves", action="store_true", help="Also write moves.json (generators) next to the .onnx")
    parser.add_argument("--batch", type=int, default=8, help="Dummy batch size used to trace the graph")
    args = parser.parse_args()

    model, V0, name, weights_path = build_model(args.group_id, args.target_id, args.model_id, args.epoch)

    out = args.out or f"onnx/{name}_{args.model_id}_e{int(args.epoch):05d}.onnx"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    # Contract expected by ui/web/*/worker.js: int64 [B, state_size] -> float [B]
    dummy = V0.unsqueeze(0).repeat(args.batch, 1).to(torch.int64)
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    print(f"Exported {weights_path}")
    print(f"      -> {out}  ({os.path.getsize(out) / 1024:.0f} KB)")
    print(f"  state size {V0.numel()}, classes {torch.unique(V0).numel()}")

    if args.moves:
        moves_path = os.path.join(os.path.dirname(out) or ".", "moves.json")
        with open(f"generators/p{int(args.group_id):03d}.json") as f:
            generators = json.load(f)
        with open(moves_path, "w") as f:
            json.dump(generators, f)
        print(f"      -> {moves_path}")

    check_parity(out, model, V0)


if __name__ == "__main__":
    main()
