import os
import torch
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.system_utils import mkdir_p
from argparse import ArgumentParser, Namespace

def initialize_scene(colmap_dir, model_dir, load_iteration=30000):
    
    parser = ArgumentParser()
    model_args = ModelParams(parser)

    model_args._source_path = os.path.abspath(colmap_dir)
    model_args._model_path = os.path.abspath(model_dir)
    model_args._images = "images"
    model_args._depths = ""
    model_args._resolution = 1.0
    model_args._white_background = False
    model_args.train_test_exp = False
    model_args.data_device = "cuda"
    model_args.eval = False

    dataset = Namespace(
        sh_degree=model_args.sh_degree,
        source_path=model_args._source_path,
        model_path=model_args._model_path,
        images=model_args._images,
        depths=model_args._depths,
        resolution=model_args._resolution,
        white_background=model_args._white_background,
        train_test_exp=model_args.train_test_exp,
        data_device=model_args.data_device,
        eval=model_args.eval
    )

    gaussians = GaussianModel(sh_degree=dataset.sh_degree)
    scene = Scene(dataset, gaussians,
                  load_iteration=load_iteration,
                  shuffle=False,
                  resolution_scales=[1.0])
    return scene


def filter_gaussians_by_semantics(scene, out_ply_path, semantic_threshold=0.7, semantic_scale=1.0):
    gaussians = scene.gaussians

    if hasattr(gaussians, "semantic_mask"):
        semantics = gaussians.semantic_mask
    elif hasattr(gaussians, "semantics"):
        semantics = gaussians.semantics
    else:
        raise AttributeError("The loaded model does not contain a 'semantics' or 'semantic_mask' field.")
        
    if not isinstance(semantics, torch.Tensor):
        semantics = torch.tensor(semantics, dtype=torch.float32, device="cuda")

    # Apply sigmoid + optional scaling
    semantics = torch.sigmoid(semantics * semantic_scale)

    # Threshold
    keep_mask = semantics > semantic_threshold
    kept_count = int(keep_mask.sum().item())
    print(f"[INFO] Keeping {kept_count}/{semantics.shape[0]} Gaussians after semantic filtering.")

    # Apply mask
    for attr_name, attr in list(vars(gaussians).items()):
        if isinstance(attr, torch.Tensor) and attr.shape[0] == keep_mask.shape[0]:
            setattr(gaussians, attr_name, attr[keep_mask])

    mkdir_p(os.path.dirname(out_ply_path))
    gaussians.save_ply(out_ply_path)
    print(f"[OK] Filtered PLY saved at: {out_ply_path}")


def main():
    colmap_dir = "/home/samuelemara/colmap/samuele/lemons_only_sam2/manual_traj"
    model_dir = "/home/samuelemara/gaussian-splatting-seg/output/28a55428-1"
    mask_dir = "/home/samuelemara/Grounded-SAM-2-autodistill/samuele/Lemon_only/masks"
    ply_path = os.path.join(model_dir, "point_cloud/iteration_30000/point_cloud.ply")
    out_ply_path = os.path.join(model_dir, "point_cloud/iteration_30000/scene_semantics_filtered.ply")

    scene = initialize_scene(colmap_dir, model_dir, load_iteration=30000)

    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    print(f"[INFO] Loading model from {ply_path}")
    

    # Check and filter by semantics
    filter_gaussians_by_semantics(scene, out_ply_path)


if __name__ == "__main__":
    main()

