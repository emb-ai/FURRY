"""Contract, coordinate-invariance, and exported inference regression checks."""
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .network import AnatomicalActor,NormalizedActor
from .observations import build,process_action
from .reference import ankle_support
torch.set_num_threads(1)

def test_named_regions_cover_features_once():
    a=AnatomicalActor()
    indexes=torch.cat([getattr(a,f'indices_{i}') for i in range(6)])
    assert sorted(indexes.tolist())==list(range(157))

def test_gradient_reaches_every_region_and_head():
    torch.manual_seed(6);a=AnatomicalActor();x=torch.randn(8,157,requires_grad=True)
    out=a(x);assert out.shape==(8,29)
    out.square().mean().backward()
    assert torch.isfinite(x.grad).all() and (x.grad.abs().sum(0)>0).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in a.parameters())

def test_observation_world_frame_invariance():
    rng=np.random.default_rng(42);p=rng.normal(size=(14,3));r=Rotation.random(14,random_state=rng)
    q=r.as_quat()[:,[3,0,1,2]];rp=rng.normal(size=3);rr=Rotation.random(random_state=rng)
    rq=rr.as_quat()[[3,0,1,2]];vel=rng.normal(size=(2,3));joint=rng.normal(size=(5,29))
    args=[joint[0],joint[1],rp,rq,p,q,vel[0],vel[1],joint[2],joint[3],np.zeros(29),joint[4]]
    original=build(*args)
    world=Rotation.from_euler('xyz',[.3,-.2,1.4]);shift=np.array([4.,-3.,.6]);R=world.as_matrix()
    args[2]=R@rp+shift;args[3]=(world*rr).as_quat()[[3,0,1,2]]
    args[4]=p@R.T+shift;args[5]=(world*r).as_quat()[:,[3,0,1,2]]
    args[6]=R@vel[0];args[7]=R@vel[1]
    for a,b in zip(original,build(*args)):np.testing.assert_allclose(a,b,atol=2e-6)

def test_applied_action_history_respects_target_rate():
    d=np.zeros(29);s=np.full(29,.5);target,applied=process_action(np.full(29,10.),d,d,s)
    assert target[0]==.32 and target[2]==.4 and target[15]==5.
    np.testing.assert_allclose(s*applied,target)

def test_contact_does_not_label_translation_or_invalid_stance():
    p=np.zeros((50,2,3));p[:,:,1]=.1;q=np.zeros((50,2,4));q[:,:,0]=1;v=np.ones((50,2),bool)
    assert ankle_support(p,q,v)['support'][1:].all()
    p[:,0,0]=np.arange(50)*.02*.3
    v[20:25,1]=False
    c=ankle_support(p,q,v)['support'];assert not c[:,0].any() and not c[20:26,1].any()

def test_onnx_matches_torch_unseen_batches(tmp_path):
    import onnxruntime as ort
    torch.manual_seed(7);a=NormalizedActor(AnatomicalActor(),torch.randn(157),torch.rand(157)+.2).eval()
    path=tmp_path/'policy.onnx';x=torch.randn(3,157)
    torch.onnx.export(a,x,str(path),input_names=['obs'],output_names=['actions'],dynamic_axes={'obs':{0:'batch'},'actions':{0:'batch'}},opset_version=17,dynamo=False)
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    for n in (1,7,32):
        inp=torch.randn(n,157)
        with torch.no_grad():expected=a(inp).numpy()
        np.testing.assert_allclose(session.run(None,{'obs':inp.numpy()})[0],expected,atol=3e-6,rtol=3e-5)

def test_exported_normalization_includes_rsl_epsilon():
    from rsl_rl.modules import EmpiricalNormalization
    n=EmpiricalNormalization([157]);n(torch.randn(32,157));n.eval()
    a=AnatomicalActor().eval();wrapped=NormalizedActor(a,n.mean,n.std,n.eps)
    x=torch.randn(4,157)
    torch.testing.assert_close(wrapped(x),a(n(x)),atol=1e-6,rtol=1e-5)

def test_real_ppo_updates_actor_critic_and_all_minibatches():
    from rsl_rl.algorithms import PPO
    from .network import actor_critic_class
    torch.manual_seed(123)
    policy=actor_critic_class()(157,286,29,critic_hidden_dims=[512,256,128],activation='elu')
    alg=PPO(policy,num_learning_epochs=5,num_mini_batches=4,learning_rate=.001,schedule='adaptive',desired_kl=.01,entropy_coef=.005,gamma=.99,lam=.95,device='cpu')
    alg.init_storage('rl',16,24,[157],[286],[29])
    before={k:v.clone() for k,v in policy.state_dict().items()}
    count=[0];original=alg.optimizer.step
    def counted(*args,**kwargs):count[0]+=1;return original(*args,**kwargs)
    alg.optimizer.step=counted
    for iteration in range(2):
        with torch.inference_mode():
            for _ in range(24):
                action=alg.act(torch.randn(16,157),torch.randn(16,286))
                alg.process_env_step(-action.square().mean(-1),torch.zeros(16,dtype=torch.bool),{})
            alg.compute_returns(torch.randn(16,286))
        losses=alg.update();assert all(np.isfinite(x) for x in losses.values())
    assert count[0]==40
    for prefix in ('actor.','critic.'):
        assert any(not torch.equal(before[k],v) for k,v in policy.state_dict().items() if k.startswith(prefix))
