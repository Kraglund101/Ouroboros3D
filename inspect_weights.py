import torch

def inspect_checkpoint(path, max_keys=50):
    print(f"\n🔍 Loading checkpoint from:\n{path}")
    try:
        # Set weights_only=False to allow loading full objects (needed for PyTorch 2.6+)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"\n❌ Failed to load checkpoint: {e}")
        return

    print("\n📂 Top-level keys in checkpoint:")
    for k in ckpt.keys():
        print(f"  • {k}")

    # Often the actual weights are under a nested key like 'module' or 'model'
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            print("\n✅ Found 'state_dict' — inspecting parameter keys...\n")
        elif "module" in ckpt:
            state_dict = ckpt["module"]
            print("\n✅ Found 'module' — inspecting parameter keys...\n")
        else:
            state_dict = ckpt
            print("\n✅ Treating as raw state dict...\n")
    else:
        state_dict = ckpt
        print("\n✅ Loaded object is not a dict — treating as raw model.\n")

    all_keys = list(state_dict.keys())
    print(f"🧠 Total parameter keys: {len(all_keys)}")
    print(f"\n🧾 First {min(len(all_keys), max_keys)} parameter keys:\n")
    for k in all_keys[:max_keys]:
        print(f"  • {k}")

    prefixes = sorted(set(k.split('.')[0] for k in all_keys))
    print("\n🔑 Top-level prefixes (suggesting submodules):")
    for p in prefixes:
        print(f"  • {p}")

if __name__ == "__main__":
    hardcoded_path = "/home/fpk297/Ouroboros3D/outputs/o3d_independent/svd_lgm+multi-t2iadapter-rgb-ccm+plucker-o3d-independent/checkpoints/epoch=0-step=5000.ckpt/checkpoint/mp_rank_00_model_states.pt"
    inspect_checkpoint(hardcoded_path, max_keys=50)
