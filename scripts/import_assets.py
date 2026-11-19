"""Copy exactly the S0 manifest assets, after verifying every source hash."""
from pathlib import Path
import hashlib, json, shutil

ROOT=Path(__file__).resolve().parents[1]
audit=json.loads((ROOT/'runs/s0_20261111/static_inventory.json').read_text(encoding='utf-8'))
source=Path('C:/newton/New/Particle_Spreading/experiments/release_distribution_ec66_v2/assets')
rows=[]
for row in audit['assets']:
    src=Path(row['path']); relative=src.relative_to(source)
    if hashlib.sha256(src.read_bytes()).hexdigest()!=row['sha256']:
        raise RuntimeError(f'source changed: {src}')
    dst=ROOT/'assets'/relative
    if dst.exists() and hashlib.sha256(dst.read_bytes()).hexdigest()!=row['sha256']:
        raise RuntimeError(f'refusing to overwrite changed asset: {dst}')
    rows.append((src,dst,row))
for src,dst,row in rows:
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists(): shutil.copyfile(src,dst)
manifest={'source_head':audit['source_head'],'asset_commit':audit['asset_commit'],
          'license':'unknown; no asset-specific license verified',
          'tool_transform':'(z,x-138,y-30)*0.001, then flange Rx(90deg)@Rz(90deg), no mount translation',
          'robot_units':'URDF loader interprets meters/radians; CAD provenance unverified',
          'collision_plan':'full triangle meshes; sphere-shape contacts required; runtime filters not yet validated',
          'assets':[dict(row,local_path=str(dst.relative_to(ROOT)),copied=True) for _,dst,row in rows]}
(ROOT/'assets/manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'Verified and imported {len(rows)} assets, {sum(r[2]["bytes"] for r in rows)} bytes')
