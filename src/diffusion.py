from utils import *
from model import Model

timesteps = 200
device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def q_sample(schedule: Schedule, x_0: torch.Tensor, t: torch.LongTensor, noise: Optional[torch.Tensor]=None):
    """Sample q(x_t|x_0) for a batch

    Args:
        schedule (Schedule): The $\beta_t$ schedule 
        
        x_0 (torch.Tensor): A batch of images (N, C, W, H) 
        
        t (torch.Tensor): A 1D tensor of integers (time)
        
        noise (torch.Tensor, optional): Sampled noise of the same dimension than
        x_0; if not given, sample one. Defaults to None.
    """
    if noise is None:
        noise = torch.normal(mean=0, std=1, size=x_0.shape).to(x_0.device)
        #noise = torch.randn_like(x_0)

#     cumProd = torch.cumprod(schedule.alphas, 0)
#     alphaT = temporal_gather(cumProd, t, x_0.shape)
#     xt = torch.sqrt(alphaT) * x_0 + torch.sqrt(1 - alphaT) * noise
#     return xt

    return x_0*temporal_gather(schedule.sqrt_alphas_cumprod, t, x_0.shape)\
            + noise*temporal_gather(schedule.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

@torch.no_grad()
def p_sample(schedule: Schedule, model: Model, x: torch.Tensor, t_index: int, labels, w=0.0):
    # 1 is 0 ....
    if t_index > 0:
        # torch.rand is uniform, we want N(0, I)
        z = torch.normal(mean=0, std=1, size=x.shape).cpu()
    else:
        z = 0
    model.eval()
    # if w == 0.0, labels would be None -> second term cancels out -> no guidance
    # do if else to reduce regenating time
    eps_cls = model(x.to(device), (torch.ones(x.shape[0]) * t_index).long(), labels).cpu()
    if w > 0.0:
        eps_no_cls = model(x.to(device), (torch.ones(x.shape[0]) * t_index).long(), None).cpu()
    else:
        eps_no_cls = 0.0
    # no guidance, w=0, eps_cls == eps_no_cls since labels is None
    # guidance -> formule since labels is NOT None
    eps_t = (1+w)*eps_cls - w*eps_no_cls
    model.train()
    # Sigma_t
    Alpha_t = schedule.alphas[t_index]
    Cumprod_alpha_t_1 = schedule.alphas_cumprod[t_index-1]
    Sigma_t = schedule.posterior_variance[t_index]
    
    x_t_1 = (1/torch.sqrt(Alpha_t)) * \
            (x.cpu() - (eps_t * (1 - Alpha_t)) / schedule.sqrt_one_minus_alphas_cumprod[t_index])\
            + torch.sqrt(Sigma_t)*z
    return x_t_1.to(device)
    
@torch.no_grad()
def sample(schedule, model, batch_size=16, labels=None, w=0.0):
    """Sample images from the generation model

    Args:
        schedule (Schedule): The variance schedule
        model (Model): The noise model
        batch_size (int, optional): Number of images to generate. Defaults to 16.

    Returns:
        List[torch.Tensor]: List of images for each time step $x_{T-1}, \ldots, x_0$
    """
    image_size = model.image_size
    channels = model.channels
    device = next(model.parameters()).device
    
    # if there are labels -> batch size of vectors labels will be batch size of img generated
    batch_size = labels.shape[0] if labels is not None else batch_size
    # w == 0.0 -> no guidance
    if w == 0.0:
        labels = None
    elif labels is not None:
        labels = labels.to(device)
    else:
        labels = torch.randint(0, 9, (batch_size,)).to(device)
    
    
    
    
    # Will contain $x_{T-1}, \ldots, x_0$
    imgs = []
    # torch.rand is uniform, we want N(0, I)
    initSize = (batch_size,channels,image_size, image_size)
    x = torch.normal(mean=0, std=1, size=initSize).to(device)
    for t in range(timesteps-1, 0-1, -1):
        x = p_sample(schedule, model, x, t, labels, w)
        imgs.append(x)
    return imgs

@torch.no_grad()
def p_sample_ddim(schedule: Schedule, model: Model, x: torch.Tensor, t_index: int, labels, w=0.0):
    # 1 is 0 ....
    if t_index > 0:
        # torch.rand is uniform, we want N(0, I)
        z = torch.normal(mean=0, std=1, size=x.shape).cpu()
    else:
        z = 0
    model.eval()
    # if w == 0.0, labels would be None -> second term cancels out -> no guidance
    # do if else to reduce regenating time
    eps_cls = model(x.to(device), (torch.ones(x.shape[0]) * t_index).long(), labels).cpu()
    if w > 0.0:
        eps_no_cls = model(x.to(device), (torch.ones(x.shape[0]) * t_index).long(), None).cpu()
    else:
        eps_no_cls = 0.0
    # no guidance, w=0, eps_cls == eps_no_cls since labels is None
    # guidance -> formule since labels is NOT None
    eps_t = (1+w)*eps_cls - w*eps_no_cls
    model.train()
    # DDPM alpha = DDPM alpha cum prod
    # predict x_0
    Alpha_t = schedule.alphas_cumprod[t_index]
    Alpha_t_1 = schedule.alphas_cumprod[t_index-1]
    x_0 = (x.cpu() - torch.sqrt(1 - Alpha_t)*eps_t)/torch.sqrt(Alpha_t)
    
    sigma_t = 0.0
    dir_x_t = torch.sqrt(1-Alpha_t_1-sigma_t**2)*eps_t
    rand_noise = sigma_t*z # = 0 because sigma_t = 0
    x_t_1 = torch.sqrt(Alpha_t_1)*x_0 + dir_x_t + rand_noise
    return x_t_1.to(device)
    
@torch.no_grad()
def sample_ddim(schedule, model, batch_size=16, labels=None, w=0.0):
    """Sample images from the generation model

    Args:
        schedule (Schedule): The variance schedule
        model (Model): The noise model
        batch_size (int, optional): Number of images to generate. Defaults to 16.

    Returns:
        List[torch.Tensor]: List of images for each time step $x_{T-1}, \ldots, x_0$
    """
    image_size = model.image_size
    channels = model.channels
    device = next(model.parameters()).device
    
    # if there are labels -> batch size of vectors labels will be batch size of img generated
    batch_size = labels.shape[0] if labels is not None else batch_size
    # w == 0.0 -> no guidance
    if w == 0.0:
        labels = None
    elif labels is not None:
        labels = labels.to(device)
    else:
        labels = torch.randint(0, 9, (batch_size,)).to(device)
    
    # Will contain $x_{T-1}, \ldots, x_0$
    imgs = []
    # torch.rand is uniform, we want N(0, I)
    initSize = (batch_size,channels,image_size, image_size)
    x = torch.normal(mean=0, std=1, size=initSize).to(device)
    for t in range(timesteps-1, 0, -1):
        x = p_sample_ddim(schedule, model, x, t, labels, w)
        imgs.append(x)
    return imgs