"""Minimal fixed-tool rigid-ball diagnostic. Newton is imported only on construction.

This is not the full robot environment or a policy action interface.
"""
import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation
from ball_retention.geometry import load_tool,support_point,sphere_clearance
from ball_retention.task import tool_pose_for_anchor

def initial_geometry(root,cfg):
    mesh=load_tool(root/'assets/shovel.STL')
    point,normal,_=support_point(mesh,cfg['initial']['anchor_xy_tool_m'])
    R=np.asarray(cfg['initial']['tool_rotation_world'])
    tool=tool_pose_for_anchor(cfg['initial']['anchor_world_m'],point,R)
    center_local=point+normal*(cfg['ball']['radius_m']+cfg['initial']['gap_m'])
    if sphere_clearance(mesh,center_local,cfg['ball']['radius_m']) < -cfg['admission']['max_initial_penetration_m']:
        raise ValueError('initial ball penetrates mesh')
    center=tool[:3,:3]@center_local+tool[:3,3]
    diagnostic=cfg.get('diagnostic',{})
    if diagnostic.get('geometry')=='flat_patch':
        import trimesh
        mesh=trimesh.creation.box(extents=[.3,.3,.01])
        mesh.apply_translation(np.asarray(cfg['initial']['anchor_world_m'])-[0,0,.005])
        mesh.apply_transform(np.linalg.inv(tool))
    if 'ball_center_world_m' in diagnostic:
        center=np.asarray(diagnostic['ball_center_world_m'])
    local=tool[:3,:3].T@(center-tool[:3,3])
    if sphere_clearance(mesh,local,cfg['ball']['radius_m']) < -cfg['admission']['max_initial_penetration_m']:
        raise ValueError('diagnostic ball penetrates geometry')
    return mesh,tool,center

def diagnostic_metrics(body_q,body_qd,ball_id,tool_id,mesh,cfg):
    if not np.isfinite(body_q).all() or not np.isfinite(body_qd).all():
        raise FloatingPointError('nonfinite body state')
    if np.max(abs(np.linalg.norm(body_q[:,3:],axis=1)-1))>1e-3:
        raise FloatingPointError('nonunit body quaternion')
    tool=body_q[tool_id]; ball=body_q[ball_id]
    local=Rotation.from_quat(tool[3:]).inv().apply(ball[:3]-tool[:3])
    clearance=sphere_clearance(mesh,local,cfg['ball']['radius_m'])
    return {'ball_local_m':local.tolist(),'surface_clearance_m':clearance,
            'floor_reached':bool(ball[2]<=cfg['physics']['floor_z_m']+cfg['ball']['radius_m']+.001)}

class FixedToolProbe:
    def __init__(self,root,cfg,device,report_stage=None):
        started=time.monotonic(); self.phase_times_s={}
        def stage(name):
            self.phase_times_s[name]=time.monotonic()-started
            if report_stage: report_stage(name)
        stage('before_physics_imports')
        import warp as wp
        import newton
        stage('imports_complete')
        self.wp=wp; self.newton=newton; self.cfg=copy.deepcopy(cfg)
        self.mesh,tool,center=initial_geometry(root,cfg)
        self.base_tool=tool.copy()
        stage('geometry_complete')
        builder=newton.ModelBuilder(gravity=wp.vec3(0,0,-cfg['physics']['gravity_m_s2']))
        material=newton.ModelBuilder.ShapeConfig(density=0,mu=cfg['ball']['friction'],restitution=0,
                    mu_torsional=0,mu_rolling=0,ka=0,margin=0,gap=.001,
                    has_shape_collision=True,has_particle_collision=False)
        transform=wp.transform(wp.vec3(*tool[:3,3]),wp.quat(*Rotation.from_matrix(tool[:3,:3]).as_quat()))
        self.tool_id=builder.add_body(xform=transform,is_kinematic=True,label='tool')
        builder.add_shape_mesh(self.tool_id,mesh=newton.Mesh(vertices=np.asarray(self.mesh.vertices,np.float32),
                     indices=np.asarray(self.mesh.faces,np.int32).ravel()),cfg=material,label='shovel')
        radius=cfg['ball']['radius_m']; mass=cfg['ball']['mass_kg']; inertia=2*mass*radius**2/5
        self.ball_id=builder.add_body(xform=wp.transform(wp.vec3(*center),wp.quat_identity()),mass=mass,
                  inertia=wp.mat33(inertia,0,0,0,inertia,0,0,0,inertia),lock_inertia=True,label='ball')
        builder.add_shape_sphere(self.ball_id,radius=radius,cfg=material,label='ball')
        builder.add_ground_plane(height=cfg['physics']['floor_z_m'],cfg=material,label='floor')
        stage('builder_complete')
        self.model=builder.finalize(device=device)
        stage('finalize_complete')
        self.a=self.model.state(); self.b=self.model.state(); self.steps=0
        self._new_solver()
        stage('solver_pipeline_complete')
        self.initial=self.snapshot()
        self.mass_inertia={'mass_kg':self.model.body_mass.numpy().tolist(),
                           'inertia_kg_m2':self.model.body_inertia.numpy().tolist(),
                           'shape_mu':self.model.shape_material_mu.numpy().tolist(),
                           'shape_labels':list(self.model.shape_label)}
        self.mass_inertia['material_spec']={'mu':cfg['ball']['friction'],'mu_torsional':0,'mu_rolling':0,
                    'ka':0,'margin_m':0,'gap_m':.001,'restitution_enabled':False,'particle_collision':False}
        if not np.isclose(self.mass_inertia['mass_kg'][self.ball_id],mass,rtol=1e-6):
            raise ValueError('ball mass differs from requested value')

    def _new_solver(self):
        self.solver=self.newton.solvers.SolverXPBD(self.model,iterations=self.cfg['physics']['iterations'],enable_restitution=False)
        self.pipeline=self.newton.CollisionPipeline(self.model)
        self.contacts=self.pipeline.contacts()

    def snapshot(self):
        # Only this fixed-tool, no-controller probe. Recreate solver/pipeline on restore;
        # exact continuation is explicitly tested, never assumed from these arrays alone.
        arrays={name:value.numpy().copy() for name,value in vars(self.a).items() if isinstance(value,self.wp.array)}
        return {'schema':'fixed-tool-probe-snapshot-v1','steps':self.steps,'arrays':arrays}

    def restore(self,saved):
        if saved['schema']!='fixed-tool-probe-snapshot-v1': raise ValueError('snapshot mismatch')
        for state in (self.a,self.b):
            for name,data in saved['arrays'].items(): getattr(state,name).assign(data)
        self.steps=saved['steps']; self._new_solver()

    def reset(self):
        self.restore(self.initial)
        return self.read()

    def read(self):
        q=self.a.body_q.numpy(); qd=self.a.body_qd.numpy()
        metrics=diagnostic_metrics(q,qd,self.ball_id,self.tool_id,self.mesh,self.cfg)
        return {'step':self.steps,'time_s':self.steps*self.cfg['physics']['dt_s'],
                'body_q':q.tolist(),'body_qd_com_world':qd.tolist(),**metrics}

    def step(self):
        self.a.clear_forces()
        self.pipeline.collide(self.a,self.contacts)
        count=int(self.contacts.rigid_contact_count.numpy()[0])
        if count>=self.contacts.rigid_contact_max: raise RuntimeError('rigid contact buffer saturated')
        self.solver.step(self.a,self.b,None,self.contacts,self.cfg['physics']['dt_s'])
        self.a,self.b=self.b,self.a; self.steps+=1
        result=self.read(); result['contacts_before_step']=count
        return result

    def prescribe_tilt(self,time_s,sign):
        """Diagnostic only: smooth pulse about world X through the material anchor."""
        phase=np.clip(time_s-1.,0.,2.); u=phase if phase<=1 else 2-phase
        angle=sign*np.deg2rad(5)*(10*u**3-15*u**4+6*u**5)
        rate=sign*np.deg2rad(5)*(30*u**2-60*u**3+30*u**4)*(1 if phase<=1 else -1)
        R=Rotation.from_rotvec([angle,0,0]).as_matrix()@self.base_tool[:3,:3]
        anchor=np.asarray(self.cfg['initial']['anchor_world_m'])
        local=self.base_tool[:3,:3].T@(anchor-self.base_tool[:3,3])
        origin=anchor-R@local; omega=np.array([rate,0,0]); velocity=-np.cross(omega,R@local)
        q=self.a.body_q.numpy(); qd=self.a.body_qd.numpy()
        q[self.tool_id]=np.r_[origin,Rotation.from_matrix(R).as_quat()]
        qd[self.tool_id]=np.r_[velocity,omega]
        self.a.body_q.assign(q); self.a.body_qd.assign(qd)

    def place_ball_outside(self):
        q=self.a.body_q.numpy(); local=np.array([self.mesh.bounds[1,0]+.1,0,.1])
        q[self.ball_id,:3]=self.base_tool[:3,:3]@local+self.base_tool[:3,3]
        self.a.body_q.assign(q)
        self.initial=self.snapshot()
