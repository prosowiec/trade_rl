import random, numpy as np, torch

def set_seed(seed=42, derterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(derterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
