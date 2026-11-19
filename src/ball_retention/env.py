"""S1 full EC66 + rigid ball physical runner, with a prescribed joint path.

Not yet the S2 action/reward/termination API. Kinematic poses are prescribed at
the substep endpoint before contact generation, as in the isolated tool probe.
Dynamic timestep sensitivity must be measured for this discretization.
"""
import copy
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .clearance import excluded_pairs
from .geometry import load_tool, support_point, sphere_clearance
from .prescribed import PrescribedPath, collision_exclusions
from .robot import Robot, origin
from .task import reference, origin_velocity


class PrescribedArmEnv:
    def __init__(self, root, cfg, device='cuda:0', moving=False, report_stage=None):
        def stage(name):
            if report_stage:
                report_stage(name)
        stage('before_physics_imports')
        import newton
        import warp as wp
        stage('imports_complete')
        self.newton, self.wp = newton, wp
        self.cfg = copy.deepcopy(cfg)
        self.robot = Robot(root / 'assets/ec66_description.urdf')
        self.path = PrescribedPath(root / cfg['prescribed']['joint_path'], cfg, moving)
        self.mesh = load_tool(root / 'assets/shovel.STL')
        self.anchor_local = support_point(self.mesh, cfg['initial']['anchor_xy_tool_m'])[0]
        self.steps = 0
        self.max_fk_ball_change = 0.
        xml = ET.parse(root / 'assets/ec66_description.urdf').getroot()
        # Newton 1.6 only supports root kinematic bodies in articulations.
        # Keep the original six-joint tree in a geometry-free FK model, and copy
        # its poses/twists to independent prescribed bodies in the physical scene.
        # The FK model is never stepped; the physical ball is absent from it.
        fk_xml = copy.deepcopy(xml)
        for link in fk_xml.findall('link'):
            for tag in ('visual', 'collision'):
                for element in list(link.findall(tag)):
                    link.remove(element)
        fk_builder = newton.ModelBuilder()
        fk_builder.add_urdf(ET.tostring(fk_xml, encoding='unicode'), floating=False, collapse_fixed_joints=False)
        if len(fk_builder.joint_q) != 6 or len(fk_builder.joint_qd) != 6:
            raise ValueError('unexpected robot joint coordinate layout')
        self.fk_model = fk_builder.finalize(device=device)
        self.fk_state = self.fk_model.state()
        self.fk_com_local = self.fk_model.body_com.numpy()
        self.arm_names = [name.split('/')[-1] for name in fk_builder.body_label]
        self.arm_body_count = len(self.arm_names)
        builder = newton.ModelBuilder(gravity=wp.vec3(0, 0, -cfg['physics']['gravity_m_s2']))
        material = newton.ModelBuilder.ShapeConfig(
            density=0, mu=cfg['ball']['friction'], restitution=0, mu_torsional=0,
            mu_rolling=0, ka=0, margin=0, gap=.001, collision_filter_parent=False,
            has_shape_collision=True, has_particle_collision=False)
        import trimesh
        for name in self.arm_names:
            body = builder.add_body(is_kinematic=True, label=name)
            link = next(link for link in xml.findall('link') if link.get('name') == name)
            for element in link.findall('collision'):
                mesh_element = element.find('geometry/mesh')
                if mesh_element is None:
                    box = element.find('geometry/box')
                    if box is not None and np.all(np.fromstring(box.get('size'), sep=' ') == 0):
                        continue
                    raise ValueError('unsupported collision geometry')
                mesh = trimesh.load(root/'assets'/mesh_element.get('filename'), force='mesh')
                mesh.apply_scale(np.fromstring(mesh_element.get('scale', '1 1 1'), sep=' '))
                mesh.apply_transform(origin(element))
                builder.add_shape_mesh(body, mesh=newton.Mesh(vertices=np.asarray(mesh.vertices, np.float32),
                    indices=np.asarray(mesh.faces, np.int32).ravel()), cfg=material, label=name)
        self.tool_id = self.arm_names.index('flan')
        mount = wp.transform(wp.vec3(0, 0, 0), wp.quat(*Rotation.from_matrix(self.robot.mount).as_quat()))
        self.shovel_shape = builder.add_shape_mesh(self.tool_id, xform=mount,
            mesh=newton.Mesh(vertices=np.asarray(self.mesh.vertices, np.float32),
                             indices=np.asarray(self.mesh.faces, np.int32).ravel()), cfg=material, label='shovel')
        radius, mass = cfg['ball']['radius_m'], cfg['ball']['mass_kg']
        inertia = 2 * mass * radius ** 2 / 5
        center = np.asarray(cfg['initial']['ball_center_world_m'])
        tool, _, _ = self.robot.fk(self.path.command(0)[0])
        if sphere_clearance(self.mesh, tool[:3, :3].T @ (center-tool[:3, 3]), radius) < -cfg['admission']['max_initial_penetration_m']:
            raise ValueError('initial sphere penetrates shovel')
        self.ball_id = builder.add_body(xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
            mass=mass, inertia=wp.mat33(inertia, 0, 0, 0, inertia, 0, 0, 0, inertia),
            lock_inertia=True, label='ball')
        self.ball_shape = builder.add_shape_sphere(self.ball_id, radius=radius, cfg=material, label='ball')
        self.floor_shape = builder.add_ground_plane(height=cfg['physics']['floor_z_m'], cfg=material, label='floor')
        self.shape_names = [builder.body_label[b].split('/')[-1] if b >= 0 else 'floor' for b in builder.shape_body]
        self.shape_names[self.shovel_shape] = 'shovel'
        pairs = collision_exclusions(self.shape_names, builder.shape_body, excluded_pairs())
        builder.shape_collision_filter_pairs = pairs
        if any(self.ball_shape in pair or self.floor_shape in pair for pair in pairs):
            raise ValueError('ball or floor contact was filtered')
        if not all(int(flags) & int(newton.ShapeFlags.COLLIDE_SHAPES) for flags in builder.shape_flags):
            raise ValueError('physical shape collision disabled')
        self.assembly = {'body_labels': list(builder.body_label), 'body_flags': list(builder.body_flags),
            'shape_names': self.shape_names, 'shape_bodies': list(builder.shape_body),
            'excluded_shape_pairs': pairs, 'ball_id': self.ball_id, 'tool_body_id': self.tool_id,
            'shovel_shape': self.shovel_shape, 'floor_shape': self.floor_shape,
            'arm_is_kinematic': True, 'self_collision_physics_certified': False,
            'execution': 'separate geometry-free Newton FK tree -> prescribed world-root bodies; no torque dynamics',
            'self_clearance_evidence': 'runs/s1_reachability_20261111/paths.json'}
        stage('builder_complete')
        self.model = builder.finalize(device=device)
        stage('finalize_complete')
        self.a, self.b = self.model.state(), self.model.state()
        self.com_local = self.model.body_com.numpy()
        self._new_solver()
        self._prescribe(0)
        if 'ball_velocity_world_m_s' in cfg['initial']:
            velocity = np.asarray(cfg['initial']['ball_velocity_world_m_s'], dtype=float)
            if velocity.shape != (3,) or not np.isfinite(velocity).all():
                raise ValueError('invalid initial sphere velocity')
            qd = self.a.body_qd.numpy()
            qd[self.ball_id, :3] = velocity
            self.a.body_qd.assign(qd)
        self.initial = self.snapshot()
        self.assembly['ball_mass_kg'] = float(self.model.body_mass.numpy()[self.ball_id])
        self.assembly['ball_inertia_kg_m2'] = self.model.body_inertia.numpy()[self.ball_id].tolist()
        if not np.isclose(self.assembly['ball_mass_kg'], mass, rtol=1e-6):
            raise ValueError('incorrect ball mass')
        stage('solver_pipeline_complete')

    def _new_solver(self):
        self.solver = self.newton.solvers.SolverXPBD(self.model, iterations=self.cfg['physics']['iterations'], enable_restitution=False)
        self.pipeline = self.newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()

    def _prescribe(self, t, command=None):
        q, qd, qdd = self.path.command(t) if command is None else command
        if any(np.asarray(x).shape != (6,) or not np.isfinite(x).all() for x in (q, qd, qdd)):
            raise ValueError('invalid joint command')
        if np.any(q < self.robot.low) or np.any(q > self.robot.high):
            raise ValueError('joint position constraint')
        if np.max(abs(qd)) > self.cfg['robot']['joint_speed_rad_s'] or np.max(abs(qdd)) > self.cfg['robot']['joint_acceleration_rad_s2']:
            raise ValueError('joint speed/acceleration constraint')
        self.fk_state.joint_q.assign(q)
        self.fk_state.joint_qd.assign(qd)
        ball_before = np.r_[self.a.body_q.numpy()[self.ball_id], self.a.body_qd.numpy()[self.ball_id]]
        self.newton.eval_fk(self.fk_model, self.fk_state.joint_q, self.fk_state.joint_qd, self.fk_state)
        fk_q, fk_qd = self.fk_state.body_q.numpy(), self.fk_state.body_qd.numpy()
        poses, velocities = self.a.body_q.numpy(), self.a.body_qd.numpy()
        for i in range(self.arm_body_count):
            r = Rotation.from_quat(fk_q[i, 3:]).as_matrix()
            # Physical prescribed bodies have origin COM; convert the URDF COM twist.
            poses[i] = fk_q[i]
            velocities[i, :3] = fk_qd[i, :3]+np.cross(fk_qd[i, 3:], r @ (self.com_local[i]-self.fk_com_local[i]))
            velocities[i, 3:] = fk_qd[i, 3:]
        self.a.body_q.assign(poses)
        self.a.body_qd.assign(velocities)
        ball_after = np.r_[self.a.body_q.numpy()[self.ball_id], self.a.body_qd.numpy()[self.ball_id]]
        error = float(np.max(abs(ball_after-ball_before)))
        self.max_fk_ball_change = max(self.max_fk_ball_change, error)
        if error != 0:
            raise RuntimeError('FK overwrote dynamic sphere')
        self.command_q, self.command_qd, self.command_qdd = q.copy(), qd.copy(), qdd.copy()

    def read(self, target_rotation=None):
        q, qd = self.a.body_q.numpy(), self.a.body_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise FloatingPointError('nonfinite rigid state')
        if np.max(abs(np.linalg.norm(q[:, 3:], axis=1)-1)) > 1e-3:
            raise FloatingPointError('nonunit rigid quaternion')
        p = q[self.tool_id, :3]
        flange_r = Rotation.from_quat(q[self.tool_id, 3:]).as_matrix()
        r = flange_r @ self.robot.mount
        omega = qd[self.tool_id, 3:]
        v = origin_velocity(qd[self.tool_id, :3], omega, p, p+flange_r @ self.com_local[self.tool_id])
        anchor = p+r @ self.anchor_local
        time_s = self.steps*self.cfg['physics']['dt_s']
        expected = reference(time_s if self.path.moving else 0, self.cfg)
        local = r.T @ (q[self.ball_id, :3]-p)
        expected_tool, jacobian, _ = self.robot.fk(self.command_q)
        desired_twist = jacobian @ self.command_qd
        if target_rotation is None:
            target_rotation = self.cfg['initial']['tool_rotation_world']
        return {'step': self.steps, 'time_s': time_s, 'body_q': q.tolist(), 'body_qd_com_world': qd.tolist(),
            'command_q_rad': self.command_q.tolist(), 'command_qd_rad_s': self.command_qd.tolist(),
            'command_qdd_rad_s2': self.command_qdd.tolist(),
            'tool_position_world_m': p.tolist(), 'tool_quaternion_xyzw': Rotation.from_matrix(r).as_quat().tolist(),
            'tool_origin_velocity_world_m_s': v.tolist(), 'tool_angular_velocity_world_rad_s': omega.tolist(),
            'anchor_position_world_m': anchor.tolist(),
            'anchor_error_m': float(np.linalg.norm(anchor-expected['anchor_position'])),
            'anchor_velocity_error_m_s': float(np.linalg.norm(v+np.cross(omega, r @ self.anchor_local)-expected['anchor_velocity'])),
            'tool_rotation_error_rad': float(Rotation.from_matrix(r @ np.asarray(target_rotation).T).magnitude()),
            'fk_position_error_m': float(np.linalg.norm(p-expected_tool[:3, 3])),
            'fk_velocity_error': float(np.max(abs(np.r_[v, omega]-desired_twist))),
            'fk_ball_max_change': self.max_fk_ball_change, 'ball_local_m': local.tolist(),
            'surface_clearance_m': sphere_clearance(self.mesh, local, self.cfg['ball']['radius_m']),
            'floor_reached': bool(q[self.ball_id, 2] <= self.cfg['physics']['floor_z_m']+self.cfg['ball']['radius_m']+.001)}

    def step(self, command=None, target_rotation=None, allow_floor_contact=False):
        self._prescribe((self.steps+1)*self.cfg['physics']['dt_s'], command)
        self.a.clear_forces()
        self.pipeline.collide(self.a, self.contacts)
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        if count >= self.contacts.rigid_contact_max:
            raise RuntimeError('contact buffer saturated')
        shape0 = self.contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = self.contacts.rigid_contact_shape1.numpy()[:count]
        pairs = sorted({tuple(sorted((int(i), int(j)))) for i, j in zip(shape0, shape1)})
        # Near-gap contacts are candidates, not proof of penetration. Stop to inspect
        # unexpected pairs; never call kinematic-body silence a self-collision proof.
        allowed = {tuple(sorted((self.ball_shape, self.shovel_shape)))}
        if allow_floor_contact:
            allowed.add(tuple(sorted((self.ball_shape, self.floor_shape))))
        if any(pair not in allowed for pair in pairs):
            raise RuntimeError(f'unexpected contact candidates: {pairs}')
        self.solver.step(self.a, self.b, None, self.contacts, self.cfg['physics']['dt_s'])
        # Physical-scene free-joint coordinates are not used as ball truth or for FK.
        self.a, self.b = self.b, self.a
        self.steps += 1
        row = self.read(target_rotation)
        row['contacts_before_step'] = count
        row['contact_shape_pairs'] = pairs
        row['contact_tool_time_s'] = row['time_s']
        row['contact_ball_time_s'] = (self.steps-1)*self.cfg['physics']['dt_s']
        if row['anchor_error_m'] > self.cfg['admission']['motion_position_tolerance_m'] or row['tool_rotation_error_rad'] > np.deg2rad(self.cfg['admission']['motion_angle_tolerance_deg']):
            raise RuntimeError('reference tracking mismatch')
        if row['fk_position_error_m'] > 2e-5 or row['fk_velocity_error'] > 2e-5:
            raise RuntimeError('Newton FK differs from independent FK')
        return row

    def snapshot(self):
        return {'schema': 'prescribed-arm-v1', 'steps': self.steps,
            'arrays': {name: value.numpy().copy() for name, value in vars(self.a).items() if isinstance(value, self.wp.array)},
            'command': tuple(x.copy() for x in (self.command_q, self.command_qd, self.command_qdd))}

    def restore(self, saved):
        if saved['schema'] != 'prescribed-arm-v1':
            raise ValueError('snapshot schema mismatch')
        for state in (self.a, self.b):
            for name, value in saved['arrays'].items():
                getattr(state, name).assign(value)
        self.steps = saved['steps']
        self.command_q, self.command_qd, self.command_qdd = (x.copy() for x in saved['command'])
        self._new_solver()

    def reset(self):
        self.restore(self.initial)
        return self.read()
