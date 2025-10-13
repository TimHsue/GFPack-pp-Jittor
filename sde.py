import jittor as jt
import numpy as np
from types import SimpleNamespace
import copy
import functools

#----- VE SDE -----
#------------------
def ve_marginal_prob(x, t, sigma_min=0.01, sigma_max=25):
    std = sigma_min * (sigma_max / sigma_min) ** t
    mean = x
    return mean, std

def ve_sde(t, sigma_min=0.01, sigma_max=25):
    sigma = sigma_min * (sigma_max / sigma_min) ** t
    drift_coeff = jt.Var(0)
    diffusion_coeff = sigma * jt.sqrt(jt.Var(2 * (np.log(sigma_max) - np.log(sigma_min))))
    return drift_coeff, diffusion_coeff

def ve_prior(shape, sigma_min=0.01, sigma_max=25):
    return jt.randn(*shape) * sigma_max

#----- VP SDE -----
#------------------
def vp_marginal_prob(x, t, beta_0=0.1, beta_1=20):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = jt.exp(log_mean_coeff) * x
    std = jt.sqrt(1. - jt.exp(2. * log_mean_coeff))
    return mean, std

def vp_sde(t, beta_0=0.1, beta_1=20):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    diffusion_coeff = jt.sqrt(beta_t)
    return drift_coeff, diffusion_coeff

def vp_prior(shape, beta_0=0.1, beta_1=20):
    return jt.randn(*shape)

#----- sub-VP SDE -----
#----------------------
def subvp_marginal_prob(x, t, beta_0, beta_1):
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    mean = jt.exp(log_mean_coeff) * x
    std = 1 - jt.exp(2. * log_mean_coeff)
    return mean, std

def subvp_sde(t, beta_0, beta_1):
    beta_t = beta_0 + t * (beta_1 - beta_0)
    drift_coeff = -0.5 * beta_t
    discount = 1. - jt.exp(-2 * beta_0 * t - (beta_1 - beta_0) * t ** 2)
    diffusion_coeff = jt.sqrt(beta_t * discount)
    return drift_coeff, diffusion_coeff

def subvp_prior(shape, beta_0=0.1, beta_1=20):
    return jt.randn(*shape)

def init_sde(sde_mode):
    # the SDE-related hyperparameters are copied from https://github.com/yang-song/score_sde_pyjt
    if sde_mode == 've':
        sigma_min = 1.0
        sigma_max = 1000.0
        eps = 1e-5
        prior_fn = functools.partial(ve_prior, sigma_min=sigma_min, sigma_max=sigma_max)
        marginal_prob_fn = functools.partial(ve_marginal_prob, sigma_min=sigma_min, sigma_max=sigma_max)
        sde_fn = functools.partial(ve_sde, sigma_min=sigma_min, sigma_max=sigma_max)
    elif sde_mode == 'vp':
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(vp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(vp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(vp_sde, beta_0=beta_0, beta_1=beta_1)
    elif sde_mode == 'subvp':
        beta_0 = 0.1
        beta_1 = 20
        eps = 1e-3
        prior_fn = functools.partial(subvp_prior, beta_0=beta_0, beta_1=beta_1)
        marginal_prob_fn = functools.partial(subvp_marginal_prob, beta_0=beta_0, beta_1=beta_1)
        sde_fn = functools.partial(subvp_sde, beta_0=beta_0, beta_1=beta_1)
    else:
        raise NotImplementedError
    return prior_fn, marginal_prob_fn, sde_fn, eps


def lossFun(model, state, gnnFeatureData, marginalProbFunc):
    batchSize = int(state.batch.max().item() + 1)
    randomBatch = jt.rand(batchSize) * (1 - 0.01) + 0.01
    
    actionMask = (state.z.reshape(batchSize, -1) == 0).float().reshape(-1, 1)
    perturbFactor = jt.randn_like(state.x) * actionMask
    perturbedStateX = state.x.clone()
    
    polyIds = (state.y.reshape(-1, 1) * actionMask).long().reshape(-1)

    mu, std = marginalProbFunc(state.x, randomBatch)
    std = std[state.batch].view(-1, 1) + 1e-5
    perturbedStateX = mu + perturbFactor * std

    o1 = model(perturbedStateX, polyIds, state.z, state.batch, randomBatch, gnnFeatureData)
    o1 = o1 * actionMask
    
    delta = (o1) * std + perturbFactor
    lossAll = (delta**2)
    lossAll = lossAll.reshape(batchSize, -1).sum(dim=1) / state.x.shape[1]
    
    weight = state.w.reshape(-1)
    weightSum = weight.sum()
    lossAll = lossAll * weight 

    loss = jt.sum(lossAll) / weightSum
    return loss, delta.detach()


def pc_sampler_state(score_model,  
               sde_coeff,
               polyNumbers,
               polyIds,
               gnnFeatureData,
               paddingMaskData,
               batch_size=512, 
               num_steps=128,
               ):
    init_x = jt.rand(batch_size, polyNumbers, 4)

    polyIds = polyIds.unsqueeze(0).repeat(batch_size, 1)
    paddingMaskData = paddingMaskData.unsqueeze(0).repeat(batch_size, 1)
    init_x_batch = jt.array([i for i in range(batch_size) for _ in range(polyNumbers)], dtype=jt.int64)
    state = SimpleNamespace(x=init_x.reshape(-1, 4).clone(), 
                        y=polyIds.reshape(-1).clone(),
                        z=paddingMaskData.reshape(-1).clone(),
                        batch=init_x_batch)

    time_steps = jt.linspace(1, 0.01, num_steps)
    random_step = time_steps[0] - time_steps[1]
   
    actionsRes = []
    with jt.no_grad():
        feature = score_model.geo_feature(gnnFeatureData)
        actionMask = (state.z.reshape(batch_size, -1) == 0).float().reshape(-1, 1)
        
        for i in range(len(time_steps)):
            time_step = time_steps[i]
            if i + 1 > len(time_steps) - 1:
                step_size = time_steps[i - 1] - time_steps[i]
            else:
                step_size = time_steps[i] - time_steps[i + 1]
            step_size *= 14

            batch_time_step = jt.ones(batch_size) * time_step
             
            baseX = state.x.clone() * actionMask

            state.x = state.x + (jt.rand_like(state.x) - 0.5) * 0.001
            state.x = state.x * actionMask
            
            polyIdsInput = (state.y.reshape(-1, 1) * actionMask).long().reshape(-1)
            o1 = score_model(state.x, polyIdsInput, state.z, state.batch, batch_time_step, gnnFeatureData, feature)
            outputAll = o1 * actionMask

            output = outputAll.reshape(batch_size * polyNumbers, 4)
            
            batch_time_step = jt.ones(batch_size).unsqueeze(-1) * time_step
            drift, g = sde_coeff(batch_time_step)
            # print(i, g)
            g = g.view(-1, 1).repeat(1, polyNumbers).reshape(-1, 1)
            drift = drift.view(-1, 1)
            # print(output.shape, drift.shape, g.shape)
            drift = drift + g ** 2 * output
            delta = drift * step_size
            mean_x = baseX + delta + g * jt.sqrt(random_step) * jt.randn_like(baseX)
            state.x = mean_x

            actionsRes.append(mean_x.clone().reshape(batch_size, polyNumbers, 4).unsqueeze(1))

    # The last step does not include any noise
    res = jt.cat(actionsRes, dim=1)
    return res, mean_x.reshape(batch_size, polyNumbers, 4)


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                polyNumbers,
                polyIds,
                polyGrids,
                polyScales,
                polyConvex,
                batch_size=1, 
                num_steps=256,             
                eps=1e-3,
                atol=1e-5, 
                rtol=1e-5, 
                device='cuda', 
                t0=1,
               
                ):
    t = jt.ones(batch_size, device=score_model.device).unsqueeze(-1) * t0
    # t = jt.ones(batch_size, device=device).unsqueeze(-1)
    #n_box=polyNumbers
    # Create the latent code
    '''
    init_x = jt.randn(batch_size * n, 2, device=device) \
        * marginal_prob_std(t).repeat(1, n).view(-1, 1)
    '''
    actions = jt.randn(batch_size * polyNumbers, 2, device=score_model.device)
    mu, std = marginal_prob_std(actions, t)
    std = std.repeat(1, polyNumbers).view(-1, 1)
    actions = actions * std
    #init_x_batch = jt.tensor([i for i in range(batch_size) for _ in range(n)], dtype=jt.int64)
    ac=actions.reshape(-1).cpu().numpy()
    ks = [int((polyNumbers - 1) ** (0.45 ** i)) for i in range(2)]
    
    with jt.no_grad():
        feature = score_model.feature(polyGrids, polyScales)
   
    
    init_x_batch = jt.tensor([i for i in range(batch_size) for _ in range(polyNumbers)], dtype=jt.int64).cpu()

    def score_eval_wrapper(ac1, pid1, pgr1, psc1, polyConvex, time_steps):
        """A wrapper of the score-based 0model for use by the ODE solver."""
        edge_index = buildknn.buildKnnGJKSerial(polyConvex, pid1, ac1, ks, init_x_batch)
        with jt.no_grad():    
            score1, score2 = score_model(ac1, pid1, init_x_batch, pgr1, psc1, edge_index, time_steps, feature)
            score = score1 + score2
        return score.cpu().numpy().reshape((-1,))
    
    def ode_func(t, ac1, pid1, pgr1, psc1):        
        """The ODE function for use by the ODE solver."""
   
        time_steps = jt.ones(batch_size, device=t.device).unsqueeze(-1) * t
        drift, g = diffusion_coeff(jt.tensor(t))
        g = g.cpu().numpy()
        ac1=jt.FloatTensor(ac1.reshape(actions.shape)).to(t.device)
        
        return  -0.5 * (g**2) * score_eval_wrapper(ac1, pid1, pgr1, psc1, polyConvex, time_steps)
    
    # Run the black-box ODE solver.
    t_eval = None
    if num_steps is not None:                                                        
        # num_steps, from t0 -> eps
        t_eval = np.linspace(t0, eps, num_steps)
        
    ode_func0=functools.partial(ode_func, pid1=polyIds.to(score_model.device),pgr1=polyGrids.to(score_model.device),psc1=polyScales.to(score_model.device))
    x = integrate.solve_ivp(ode_func0, (t0, eps),ac,rtol=rtol, atol=atol, method='RK45', t_eval=t_eval)
    res= jt.clamp(jt.tensor(x.y[:,-1], device=score_model.device).reshape(actions.shape), min=-1.0, max=1.0)
    return res

'''
def pc_sampler_state(score_model, 
               prior,
               sde_coeff,
               polyNumbers,
               polyIds,
               polyGrids, 
               polyScales,
               snr=0.16, 
               batch_size=1, 
               num_steps=2048,             
               eps=1e-5,
               device="cuda"
               ):
    t = jt.ones(batch_size, device=device).unsqueeze(-1) * 0.1
    
    init_x = prior((batch_size, polyNumbers, 2)).to(device)
    init_x_batch = jt.tensor([i for i in range(batch_size) for _ in range(polyNumbers)], dtype=jt.int64).cpu()
    state = Data(x=init_x.reshape(-1, 2).clone(), 
                 y=polyIds,
                 batch=init_x_batch).to(device)
    
    time_steps = jt.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]
    noise_norm = np.sqrt(polyNumbers)
    
    actionsRes = []
    
    with jt.no_grad():
        f1A, f2A, f3A, f4A = score_model.feature(polyGrids, polyScales)
        for i, time_step in enumerate(time_steps[:-2]):    
            batch_time_step = jt.ones(batch_size, device=device).unsqueeze(-1) * time_step
            
            # Predictor step (Euler-Maruyama)
            x = state.x

            output_norm = jt.norm(output.reshape(batch_size, -1), dim=-1).mean() + 1e-5
            langevin_step_size = 2 * (snr * noise_norm / output_norm)**2
            x = x + langevin_step_size * output + jt.sqrt(2 * langevin_step_size) * jt.randn_like(x)  
            
            drift, diffusion = sde_coeff(batch_time_step)
            drift = drift - diffusion**2 * output
            mean_x = x + drift * step_size
            mean_x = mean_x + diffusion * jt.sqrt(step_size) * jt.randn_like(x)
            state.x = mean_x

            actionsRes.append(state.x.view(-1, 2).unsqueeze(0))

    # The last step does not include any noise
    res = jt.cat(actionsRes, dim=0)
    return res, mean_x.reshape(polyNumbers, 2)
'''

'''
class ExponentialMovingAverage:
    """
    Maintains (exponential) moving average of a set of parameters.
    """

    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
            parameters: Iterable of `jt.nn.Parameter`; usually the result of
                `model.parameters()`.
            decay: The exponential decay.
            use_num_updates: Whether to use number of updates when computing
                averages.
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                                                    for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        """
        Update currently maintained parameters.

        Call this every time the parameters are updated, such as the result of
        the `optimizer.step()` call.

        Args:
            parameters: Iterable of `jt.nn.Parameter`; usually the same set of
                parameters used to initialize this object.
        """
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with jt.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param)) # only update the ema-params

    def copy_to(self, parameters):
        """
        Copy current parameters into given collection of parameters.

        Args:
            parameters: Iterable of `jt.nn.Parameter`; the parameters to be
                updated with the stored moving averages.
        """
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        """
        Save the current parameters for restoring later.

        Args:
            parameters: Iterable of `jt.nn.Parameter`; the parameters to be
                temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.

        Args:
            parameters: Iterable of `jt.nn.Parameter`; the parameters to be
                updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                                shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']
'''
class ExponentialMovingAverage:
    """
    指数移动平均 (EMA) 用于模型参数
    """
    def __init__(self, parameters, decay, use_num_updates=True):
        """
        Args:
            parameters: 模型参数的迭代器
            decay: 衰减率
            use_num_updates: 是否使用更新次数来调整衰减率
        """
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        
        # 存储参数的影子副本
        self.shadow_params = []
        self.collected_params = []
        
        # 初始化影子参数
        for param in parameters:
            if param.requires_grad:
                # Jittor: 使用 clone() 而不是 detach().clone()
                self.shadow_params.append(param.clone())
            else:
                self.shadow_params.append(None)
    
    def update(self, parameters):
        """
        更新影子参数
        
        Args:
            parameters: 当前模型参数
        """
        decay = self.decay
        
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        one_minus_decay = 1.0 - decay
        
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad and s_param is not None:
                # Jittor: 直接更新数据
                # s_param = decay * s_param + one_minus_decay * param
                s_param.update(s_param * decay + param * one_minus_decay)
    
    def copy_to(self, parameters):
        """
        将影子参数复制到模型参数
        
        Args:
            parameters: 目标模型参数
        """
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad and s_param is not None:
                # Jittor: 使用 update() 或直接赋值
                param.update(s_param.clone())
    
    def store(self, parameters):
        """
        保存当前模型参数
        
        Args:
            parameters: 要保存的模型参数
        """
        self.collected_params = []
        for param in parameters:
            if param.requires_grad:
                # Jittor: 使用 clone()
                self.collected_params.append(param.clone())
            else:
                self.collected_params.append(None)
    
    def restore(self, parameters):
        """
        恢复之前保存的模型参数
        
        Args:
            parameters: 要恢复到的模型参数
        """
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad and c_param is not None:
                # Jittor: 使用 update()
                param.update(c_param.clone())
        
        # 清空已保存的参数
        self.collected_params = []
    
    def state_dict(self):
        """
        返回 EMA 的状态字典
        """
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow_params': [p.clone() if p is not None else None 
                            for p in self.shadow_params]
        }
    
    def load_state_dict(self, state_dict):
        """
        从状态字典加载 EMA
        """
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [p.clone() if p is not None else None 
                             for p in state_dict['shadow_params']]