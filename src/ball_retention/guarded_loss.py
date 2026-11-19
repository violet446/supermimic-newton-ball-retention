"""Free-rollout position-center and strong-response direction supervision."""
import numpy as np


def center_direction_loss(model,initial,known,target,branch_mask,noise_mask,
                          baseline_first,baseline_second,lead_weights,sample_weights,config):
    n,h=initial.shape[0],known.shape[2]
    x=np.concatenate((initial,initial))
    k=known.reshape(2*n,h,model.known_dim)
    z=model.rollout(x,k).reshape(2,n,h,9)
    raw=model.denormalize(z)
    matched=branch_mask.all(axis=0)
    weights=matched*sample_weights[:,None]*lead_weights[None,:]
    if weights.sum()<=0: raise ValueError('no matched center targets')
    weights=weights/weights.sum()
    error=(raw[0,:,:,:3]+raw[1,:,:,:3]-target[0,:,:,:3]-target[1,:,:,:3])/2
    scale=config['center_position_scale_m']
    center=float(np.sum(weights[...,None]*error**2)/scale**2)
    cotangent=np.zeros_like(z)
    center_grad=config['center_position_weight']*weights[...,None]*error*model.state_scale[:3]/scale**2
    cotangent[0,:,:,:3]+=center_grad
    cotangent[1,:,:,:3]+=center_grad

    predicted=(raw[0,:,:,:3]-raw[1,:,:,:3])/2
    measured=(target[0,:,:,:3]-target[1,:,:,:3])/2
    magnitude=np.linalg.norm(measured,axis=-1)
    floor=np.maximum(1e-6,5*np.linalg.norm(baseline_first[:,:,:3]-baseline_second[:,:,:3],axis=-1)/2)
    strong=noise_mask&matched&(magnitude>=10*floor)
    dw=strong*sample_weights[:,None]*lead_weights[None,:]
    if dw.sum()>0: dw=dw/dw.sum()
    unit=measured/np.maximum(magnitude[...,None],1e-30)
    softnorm=np.sqrt(np.sum(predicted**2,axis=-1)+floor**2)
    cosine=np.sum(predicted*unit,axis=-1)/softnorm
    hinge=np.maximum(config['direction_target_cosine']-cosine,0.)
    direction=float(np.sum(dw*hinge**2))
    dcos=unit/softnorm[...,None]-cosine[...,None]*predicted/softnorm[...,None]**2
    dg=-config['direction_weight']*dw[...,None]*hinge[...,None]*dcos*model.state_scale[:3]
    cotangent[0,:,:,:3]+=dg
    cotangent[1,:,:,:3]-=dg
    _,_,grad=model.rollout_vjp(x,k,cotangent.reshape(2*n,h,9))
    loss=config['center_position_weight']*center+config['direction_weight']*direction
    if not np.isfinite(loss): raise FloatingPointError('non-finite auxiliary loss')
    return {'center':center,'direction':direction,'loss':loss,'strong_selected_leads':int(np.sum(strong&(lead_weights[None,:]>0)))},grad


def checkpoint_guard(reference,candidate,config):
    """Reject any predeclared group/amplitude or route regression before ranking."""
    failures=[]
    checks={}
    for key,before in reference['summary'].items():
        # Cohort aggregates are reports, not substitutes for individual groups.
        parts=key.split('/')
        if parts[0]!='validation' or parts[1] in ('original','coverage'): continue
        a,b=before['32'],candidate['summary'][key]['32']
        rmse=a['absolute_rmse_p_v_w_si'][0]
        passed=b['absolute_rmse_p_v_w_si'][0]<=config['H32_position_rmse_max_ratio']*rmse+1e-12
        checks[key+'/position']=passed
        for count in ('positive_cosine','cosine_ge_0_5'):
            if b['position_strong']['resolved']!=a['position_strong']['resolved']:
                raise ValueError('physical strong-response set changed')
            checks[key+'/'+count]=b['position_strong'][count]>=a['position_strong'][count]
    for episode,row in reference['s2_validation']['8'].items():
        # The caller marks route identities explicitly; edge windows stay in reports.
        if episode not in reference['complete_route_ids']: continue
        checks['route/'+episode]=candidate['s2_validation']['8'][episode]['rmse_p_v_w_si'][0]<=config['S2_complete_route_H8_position_rmse_max_ratio']*row['rmse_p_v_w_si'][0]+1e-12
    if not checks: raise ValueError('empty checkpoint guard')
    failures=[key for key,passed in checks.items() if not passed]
    return {'passed':not failures,'failures':failures,'checks':checks}
