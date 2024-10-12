import hashlib
import time

def generate_unique_run_name(hf_model_path: str, task_name: str, use_peft: str) -> str:
    """unique run name for tracking experiments."""
    timestamp = int(time.time())

    hash_input = f"{hf_model_path}_{task_name}_{timestamp}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    model_name = hf_model_path.split("/")[-1]
    return f"{model_name}_{task_name}_{use_peft}_{hash_value}"
