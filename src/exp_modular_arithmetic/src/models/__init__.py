from src.models.transformer import Decoder
from src.models.mlp import MLP_arithmetic, MLP_tokenembedding

def get_model(model_config, task_model_hparams):
    # Override the model parameters
    model_config_expt = model_config.copy()
    model_config_expt.override(task_model_hparams)
    model = Decoder(
        dim=model_config_expt.dim,
        num_layers=model_config_expt.num_layers,
        num_heads=model_config_expt.num_heads,
        num_tokens=model_config_expt.num_tokens,
        seq_len=model_config_expt.max_len
    )
    return model, model_config_expt