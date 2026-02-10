import os

from omegaconf import DictConfig


def get_parameter_order():
    """
    Define the order of parameters in directory names.

    Returns:
        List of parameter names in the order they should appear in directory names
    """
    return [
        "model_type",
        "vit_arch",
        "use_boxes",
        "subspace_dim",
        "product_metric",
        # "optimizer",
        # "warmup_steps",
        # "lr",
        # "weight_decay",
        # "batch_size",
        "entail_weight",
        "curv_init",
        "lorentz_eps",
    ]


def extract_parameters(config: DictConfig):
    """
    Extract parameters from either a config object or dictionary.

    Args:
        config: Config object or dictionary containing model parameters

    Returns:
        Dictionary of parameter names and their values
    """
    params = {}

    # Helper function to safely get nested values
    def safe_get(data, *keys):
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            elif hasattr(data, key):
                data = getattr(data, key)
            else:
                return None
        return data

    # Extract model type
    target = safe_get(config, "model", "_target_")
    if target:
        if "PHyCLIP" in target:
            params["model_type"] = "phyclip"
        elif "HyCoCLIP" in target:
            params["model_type"] = "hycoclip"
        elif "MERU" in target:
            params["model_type"] = "meru"
        elif "CLIPBaseline" in target:
            params["model_type"] = "clip"

    # Extract vision transformer architecture
    arch = safe_get(config, "model", "visual", "arch")
    if arch:
        for size, short in [("small", "s"), ("base", "b"), ("large", "l")]:
            if f"vit_{size}" in arch or f"vit_{short}" in arch:
                params["vit_arch"] = f"vit_{short}"
                break

    # Extract optimizer information
    optimizer_target = safe_get(config, "optim", "optimizer", "_target_")
    if optimizer_target:
        for opt_name in ["RiemannianAdam", "AdamW", "Adam", "SGD"]:
            if opt_name in optimizer_target:
                params["optimizer"] = opt_name
                break
        else:
            params["optimizer"] = str(optimizer_target).split(".")[-1]

    # Extract simple parameters with direct mapping
    param_mappings = {
        "lr": ("optim", "optimizer", "lr"),
        "weight_decay": ("optim", "optimizer", "weight_decay"),
        "warmup_steps": ("optim", "lr_scheduler", "warmup_steps"),
        "batch_size": ("train", "total_batch_size"),
        "subspace_dim": ("model", "subspace_dim"),
        "product_metric": ("model", "product_metric"),
        "entail_weight": ("model", "entail_weight"),
        "curv_init": ("model", "curv_init"),
        "lorentz_eps": ("model", "lorentz_eps"),
        "use_boxes": ("model", "use_boxes"),
    }

    for param_name, keys in param_mappings.items():
        value = safe_get(config, *keys)
        if value is not None:
            params[param_name] = value

    return params


def format_parameter_value(param_name, value):
    """
    Format parameter value for directory name.

    Args:
        param_name: Name of the parameter
        value: Value of the parameter

    Returns:
        Formatted string for the parameter
    """

    formatters = {
        "lr": lambda v: f"lr{v:.0e}",
        "subspace_dim": lambda v: f"subspaceDim{v}",
        "product_metric": lambda v: f"pm{v}",
        "optimizer": lambda v: f"opt{v}",
        "batch_size": lambda v: f"bs{v}",
        "weight_decay": lambda v: f"wd{v:.0e}",
        "warmup_steps": lambda v: f"warmup{v}",
        "entail_weight": lambda v: f"entailWeight{v:.0e}",
        "curv_init": lambda v: f"curvInit{v:.0e}",
        "lorentz_eps": lambda v: f"lorentzEps{v:.0e}",
        "use_boxes": lambda v: "useBoxes" if v else "noBoxes",
        "use_adaptive_entailment": lambda v: "entAda" if v else "entAvg",
    }

    return formatters.get(param_name, str)(value)


def generate_output_dir_name(_C: DictConfig) -> str:
    """
    Generate output directory name from config parameters.

    Args:
        _C: Config object containing model parameters

    Returns:
        String representing the output directory name
    """
    params = extract_parameters(_C)
    order = get_parameter_order()

    # Build directory name parts in specified order
    dir_parts = []
    simple_params = {"model_type", "vit_arch"}

    for param_name in order:
        if param_name in params:
            value = params[param_name]
            if param_name in simple_params:
                dir_parts.append(str(value))
            else:
                dir_parts.append(format_parameter_value(param_name, value))

    # Only add subdirectory if we have parameters
    if dir_parts:
        return os.path.join("train_results", "_".join(dir_parts))
    else:
        return "train_results"
