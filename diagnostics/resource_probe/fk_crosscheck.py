"""Newton URDF FK crosscheck, called only inside an admitted worker."""
def crosscheck(root,cfg):
    import xml.etree.ElementTree as ET
    import numpy as np
    import warp as wp
    import newton
    from scipy.spatial.transform import Rotation
    from ball_retention.robot import Robot
    from ball_retention.task import origin_velocity
    xml=ET.parse(root/'assets/ec66_description.urdf').getroot()
    for mesh in xml.findall('.//mesh'): mesh.set('filename',str(root/'assets'/mesh.get('filename')))
    for link in xml.findall('link'):
        for tag in ('visual','collision'):
            for element in list(link.findall(tag)):
                box=element.find('geometry/box')
                if box is not None and np.all(np.fromstring(box.get('size'),sep=' ')==0): link.remove(element)
    builder=newton.ModelBuilder()
    builder.add_urdf(ET.tostring(xml,encoding='unicode'),floating=False,collapse_fixed_joints=False)
    model=builder.finalize(device='cuda:0'); state=model.state()
    if model.joint_coord_count!=6 or model.joint_dof_count!=6: raise ValueError('unexpected URDF joint schema')
    flange=next(i for i,n in enumerate(model.body_label) if n.split('/')[-1]=='flan')
    robot=Robot(root/'assets/ec66_description.urdf'); q=np.array(cfg['initial']['joint_q_rad'])
    qd=np.array([.1,-.08,.07,-.06,.05,-.04]); state.joint_q.assign(q.astype(np.float32)); state.joint_qd.assign(qd.astype(np.float32))
    newton.eval_fk(model,state.joint_q,state.joint_qd,state)
    pose=state.body_q.numpy()[flange]; velocity=state.body_qd.numpy()[flange]
    R=Rotation.from_quat(pose[3:]).as_matrix(); actual=np.eye(4); actual[:3,:3]=R@robot.mount; actual[:3,3]=pose[:3]
    expected,J,poses=robot.fk(q); com=pose[:3]+R@model.body_com.numpy()[flange]
    v_origin=origin_velocity(velocity[:3],velocity[3:],pose[:3],com)
    result={'position_error_m':float(np.linalg.norm(actual[:3,3]-expected[:3,3])),
            'orientation_error_rad':float(Rotation.from_matrix(actual[:3,:3]@expected[:3,:3].T).magnitude()),
            'linear_velocity_error_m_s':float(np.linalg.norm(v_origin-(J@qd)[:3])),
            'angular_velocity_error_rad_s':float(np.linalg.norm(velocity[3:]-(J@qd)[3:]))}
    if max(result.values())>2e-5: raise RuntimeError(f'FK mismatch: {result}')
    return {'fk_errors':result,'physics_steps':0,'solver_step_executed':False}
