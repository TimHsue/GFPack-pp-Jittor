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



class ExponentialMovingAverage:
    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        
        self.shadow_params = []
        self.collected_params = []
        
        for param in parameters:
            if param.requires_grad:
                self.shadow_params.append(param.clone())
            else:
                self.shadow_params.append(None)
    
    def update(self, parameters):
        decay = self.decay
        
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        one_minus_decay = 1.0 - decay
        
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad and s_param is not None:
                s_param.update(s_param * decay + param * one_minus_decay)
    
    def copy_to(self, parameters):
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad and s_param is not None:
                param.update(s_param.clone())
    
    def store(self, parameters):
        self.collected_params = []
        for param in parameters:
            if param.requires_grad:
                self.collected_params.append(param.clone())
            else:
                self.collected_params.append(None)
    
    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            if param.requires_grad and c_param is not None:
                param.update(c_param.clone())
        
        self.collected_params = []
    
    def state_dict(self):
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow_params': [p.clone() if p is not None else None 
                            for p in self.shadow_params]
        }
    
    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = [p.clone() if p is not None else None 
                             for p in state_dict['shadow_params']]