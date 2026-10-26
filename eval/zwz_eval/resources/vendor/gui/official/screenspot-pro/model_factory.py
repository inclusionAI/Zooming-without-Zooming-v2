def build_model(args):
    if args.model_type != "qwen3_5_vllm":
        raise ValueError("The preserved native GUI branch is qwen3_5_vllm only")
    from models.qwen3_5 import Qwen3_5_VLLM_Model
    model = Qwen3_5_VLLM_Model()
    model.load_model(model_name_or_path=args.model_name_or_path)
    return model
