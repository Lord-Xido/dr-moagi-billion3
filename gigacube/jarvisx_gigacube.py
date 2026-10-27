#!/usr/bin/env python3
"""JARVIS-X Gigacube: disk-backed, exact reversible 3-D bit lattice.

Logical domain: (8_000_000_000 bit coordinates)^3. Only explicitly stored,
fixed-size 3-D bricks are computed; the absent background is zero and frozen.
This is a *virtual logical address space*, not 64 ronnabytes of physical RAM.

The checkerboard transformation is bijective on the fixed allocated region:
  A' = A XOR F(B)  /  B' = B XOR G(A')
  B  = B' XOR G(A') / A  = A' XOR F(B)
Each phase reads only the opposite parity. Neighbor faces between bricks are
loaded from SQLite. The pair of phases + iteration counter commit atomically.

Dependencies: Python 3.10+, NumPy. No GPU or quantum hardware needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

import numpy as np

AXIS_BITS = 8_000_000_000  # decimal gigabyte: 8e9 bit positions *per axis*
DEFAULT_BRICK = 64


def morton_encode(x: int, y: int, z: int) -> int:
    """Interleave 33 bits of each global coordinate -> 99-bit address."""
    if any(v < 0 or v >= AXIS_BITS for v in (x, y, z)):
        raise ValueError('out of logical bit-coordinate bounds')
    result = 0
    for k in range(33):
        result |= ((x >> k) & 1) << (3*k)
        result |= ((y >> k) & 1) << (3*k+1)
        result |= ((z >> k) & 1) << (3*k+2)
    return result


def morton_decode(value: int) -> tuple[int, int, int]:
    """Morton decode; caller validates coordinates fall within logical cube."""
    if value < 0 or value >= (1 << 99):
        raise ValueError('Morton address outside 99-bit space')
    xyz = [0, 0, 0]
    for k in range(33):
        for axis in range(3):
            xyz[axis] |= ((value >> (3*k+axis)) & 1) << k
    return tuple(xyz)


def brick_pos(sequence: int, side: int) -> tuple[int, int, int]:
    c = morton_decode(sequence)
    if any((v+1)*side > AXIS_BITS for v in c):
        raise ValueError('input has exceeded the configured 3D virtual volume')
    return c


def sha_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


class Gigacube:
    def __init__(self, dbpath: str | Path):
        self.path = Path(dbpath)
        if not self.path.is_file():
            raise FileNotFoundError(f'Gigacube database not found: {self.path}')
        self.cn = sqlite3.connect(str(self.path), timeout=60.0)
        self.cn.execute('PRAGMA busy_timeout=60000')
        self.cn.execute('PRAGMA synchronous=FULL')
        self.meta = {k: json.loads(v) for k, v in self.cn.execute('SELECT key,value FROM meta')}
        self.side = int(self.meta['brick_side'])
        self.pagebytes = self.side**3 // 8
        self.zero = bytes(self.pagebytes)
        self.coord_rows = list(self.cn.execute('SELECT seq,x,y,z FROM tiles ORDER BY seq'))
        self.coords = [tuple(row[1:]) for row in self.coord_rows]
        gx = np.arange(self.side, dtype=np.uint16)[:, None, None]
        gy = np.arange(self.side, dtype=np.uint16)[None, :, None]
        gz = np.arange(self.side, dtype=np.uint16)[None, None, :]
        self.even = (gx + gy + gz) % 2 == 0
        self.odd = ~self.even

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.cn.close()

    def update_meta(self, key, value):
        self.cn.execute('UPDATE meta SET value=? WHERE key=?', (json.dumps(value), key))
        self.meta[key] = value

    def get_raw(self, coord: tuple[int,int,int]) -> bytes:
        row = self.cn.execute('SELECT data FROM tiles WHERE x=? AND y=? AND z=?', coord).fetchone()
        return row[0] if row is not None else self.zero

    def unpack(self, raw: bytes) -> np.ndarray:
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder='little').reshape(
            (self.side, self.side, self.side))

    @staticmethod
    def packed(a: np.ndarray) -> bytes:
        return np.packbits(a, bitorder='little').tobytes()

    def neighbor_halo(self, coord: tuple[int,int,int], center: np.ndarray) -> np.ndarray:
        """Zero exterior boundary; six remote brick faces when allocated."""
        n = self.side
        x,y,z = coord
        halo = np.zeros((n+2, n+2, n+2), dtype=np.uint8)
        halo[1:-1,1:-1,1:-1] = center
        for delta, dest, src in (
            ((-1,0,0), (0, slice(1,-1),slice(1,-1)), (-1, slice(None),slice(None))),
            (( 1,0,0), (-1,slice(1,-1),slice(1,-1)), (0, slice(None),slice(None))),
            ((0,-1,0), (slice(1,-1),0,slice(1,-1)), (slice(None),-1,slice(None))),
            ((0, 1,0), (slice(1,-1),-1,slice(1,-1)), (slice(None),0,slice(None))),
            ((0,0,-1), (slice(1,-1),slice(1,-1),0), (slice(None),slice(None),-1)),
            ((0,0, 1), (slice(1,-1),slice(1,-1),-1), (slice(None),slice(None),0)),
        ):
            target = (x+delta[0], y+delta[1], z+delta[2])
            raw = self.get_raw(target)
            if raw != self.zero:
                halo[dest] = self.unpack(raw)[src]
        return halo

    def phase(self, parity: int) -> int:
        """Return changed-bit count. Update only target parity, read other parity."""
        n = self.side
        mask = self.even if parity == 0 else self.odd
        changed = 0
        for _, x, y, z in self.coord_rows:
            coord = (x,y,z)
            core = self.unpack(self.get_raw(coord))
            p = self.neighbor_halo(coord, core)
            xp = p[2:, 1:-1, 1:-1]
            xm = p[:-2, 1:-1, 1:-1]
            yp = p[1:-1, 2:, 1:-1]
            ym = p[1:-1, :-2, 1:-1]
            zp = p[1:-1, 1:-1, 2:]
            zm = p[1:-1, 1:-1, :-2]
            if parity == 0:
                flux = xp ^ xm ^ yp ^ ym ^ zp ^ zm ^ (xp & yp)
            else:
                flux = xp ^ ym ^ zm ^ (xm & zp)
            changed += int(np.count_nonzero(flux[mask]))
            core[mask] ^= flux[mask]
            self.cn.execute('UPDATE tiles SET data=? WHERE x=? AND y=? AND z=?',
                            (self.packed(core), x,y,z))
        return changed

    def digest_state(self) -> tuple[str, int]:
        h = hashlib.sha256()
        ones = 0
        for (raw,) in self.cn.execute('SELECT data FROM tiles ORDER BY seq'):
            h.update(raw)
            ones += int(np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).sum())
        return h.hexdigest(), ones

    def iterations(self, count: int, reverse: bool=False, quiet: bool=False) -> list[dict]:
        if count < 0:
            raise ValueError('count must be nonnegative')
        level = int(self.meta['iteration'])
        if reverse and count > level:
            raise ValueError(f'cannot reverse {count} iterations: only {level} recorded')
        trace = []
        for _ in range(count):
            old = level
            t0 = time.perf_counter()
            try:
                self.cn.execute('BEGIN IMMEDIATE')
                changes = 0
                for parity in ((1,0) if reverse else (0,1)):
                    changes += self.phase(parity)
                level = level - 1 if reverse else level + 1
                digest, ones = self.digest_state()
                self.update_meta('iteration', level)
                self.update_meta('state_sha256', digest)
                self.cn.execute('INSERT INTO journal(iteration,direction,changed_bits,ones,sha256) VALUES (?,?,?,?,?)',
                                (level, 'inverse' if reverse else 'forward', changes, ones, digest))
                self.cn.commit()
            except Exception:
                self.cn.rollback()
                level = old
                self.meta['iteration'] = old
                raise
            elapsed = time.perf_counter() - t0
            item = {'iteration':level,'direction':'inverse' if reverse else 'forward',
                    'changed_bits':changes,'one_bits':ones,'state_sha256':digest,
                    'seconds':round(elapsed,6)}
            trace.append(item)
            if not quiet:
                print(json.dumps(item))
        return trace

    def export_bytes(self, path: str | Path):
        """Export payload in ingest brick order, dropping final zero padding."""
        remain = int(self.meta['input_bytes'])
        with open(path, 'wb') as out:
            for (raw,) in self.cn.execute('SELECT data FROM tiles ORDER BY seq'):
                if remain <= 0: break
                part = raw[:remain]
                out.write(part)
                remain -= len(part)
        if remain != 0:
            raise RuntimeError('database missing input bricks')

    def recover(self, output: str | Path) -> dict:
        """Copy SQLite state, invert in copy, verify exact original file digest."""
        with tempfile.TemporaryDirectory(prefix='gigacube_recovery_') as td:
            snapshot = Path(td) / 'snapshot.sqlite'
            dst = sqlite3.connect(snapshot)
            self.cn.backup(dst)
            dst.close()
            with Gigacube(snapshot) as scratch:
                scratch.iterations(int(scratch.meta['iteration']), reverse=True, quiet=True)
                scratch.export_bytes(output)
        result_sha = sha_file(output)
        expected = self.meta['input_sha256']
        ok = result_sha == expected
        return {'sha256_expected':expected,'sha256_actual':result_sha,
                'exact_reconstruction':ok,'output':str(output)}

    def status(self) -> dict:
        state, ones = self.digest_state()
        return {'logical_axes_bit_positions':[AXIS_BITS]*3,
                'logical_cells_bits':str(AXIS_BITS**3),
                'fully_dense_bytes':str(AXIS_BITS**3//8),
                'brick_side_bits':self.side,
                'packed_bytes_per_brick':self.pagebytes,
                'allocated_bricks':len(self.coords),
                'allocated_voxel_positions':len(self.coords)*self.side**3,
                'allocated_packed_bytes':len(self.coords)*self.pagebytes,
                'input_bytes':self.meta['input_bytes'],
                'iteration':self.meta['iteration'],
                'state_sha256':state,
                'one_bits':ones,
                'sqlite_file_bytes':self.path.stat().st_size,
                'source_sha256':self.meta['input_sha256'],
                'boundary':'zero and frozen outside allocated bricks',
                'address_bits':99}


def initialize(input_path: str | Path, db_path: str | Path, side:int=DEFAULT_BRICK,
               max_mib: float=64):
    if side < 8 or side > 128 or side % 8 or (side & (side-1)):
        raise ValueError('brick side must be one of 8,16,32,64,128')
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    nbytes = source.stat().st_size
    if nbytes == 0:
        raise ValueError('input file must contain at least 1 byte')
    if nbytes > max_mib * (1024**2):
        raise ValueError('input exceeds configured --max-mib safety limit; increase explicitly')
    path = Path(db_path)
    if path.exists():
        raise FileExistsError(f'won\'t replace database: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    pagebytes = side**3//8
    cn = sqlite3.connect(path)
    try:
        cn.execute('PRAGMA synchronous=FULL')
        cn.execute('PRAGMA journal_mode=DELETE')
        cn.execute('CREATE TABLE meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        cn.execute('CREATE TABLE tiles (seq INTEGER PRIMARY KEY, x INTEGER NOT NULL, y INTEGER NOT NULL, z INTEGER NOT NULL, data BLOB NOT NULL, UNIQUE(x,y,z))')
        cn.execute('CREATE TABLE journal (id INTEGER PRIMARY KEY AUTOINCREMENT, iteration INTEGER, direction TEXT,changed_bits INTEGER, ones INTEGER, sha256 TEXT)')
        h = hashlib.sha256()
        current = hashlib.sha256()
        seq = 0
        with open(source,'rb') as inp:
            while (chunk:=inp.read(pagebytes)):
                h.update(chunk)
                brick = chunk.ljust(pagebytes,b'\0')
                current.update(brick)
                coords = brick_pos(seq,side)
                cn.execute('INSERT INTO tiles(seq,x,y,z,data) VALUES(?,?,?,?,?)',
                           (seq,*coords,brick))
                seq += 1
        for k,v in {'schema':1,'logical_axis_bits':AXIS_BITS,'brick_side':side,
                    'input_bytes':nbytes,'input_sha256':h.hexdigest(),
                    'state_sha256':current.hexdigest(),'iteration':0,
                    'brick_count':seq}.items():
            cn.execute('INSERT INTO meta(key,value) VALUES(?,?)', (k,json.dumps(v)))
        cn.commit()
        return {'db':str(path),'input_bytes':nbytes,'bricks':seq,
                'packed_brick_bytes':pagebytes,'sha256':h.hexdigest()}
    except Exception:
        cn.rollback()
        cn.close()
        path.unlink(missing_ok=True)
        raise
    finally:
        try: cn.close()
        except Exception: pass


def render_html(eng:Gigacube, output: str | Path, max_points:int=4000):
    """Stand-alone HTML viewer that embeds sampled *actual* 1-bits from SQLite."""
    rng = np.random.default_rng(827)
    max_points = max(1,min(max_points,30000))
    # Reserve a fair quota for each of the first bricks when there are many.
    sampled = []
    max_bricks = min(len(eng.coord_rows), min(512,max_points))
    for seq,x,y,z in eng.coord_rows[:max_bricks]:
        arr = eng.unpack(eng.get_raw((x,y,z)))
        ones = np.flatnonzero(arr)
        if not len(ones): continue
        quota = max(1,max_points//max_bricks)
        if len(ones)>quota:
            ones = rng.choice(ones,size=quota,replace=False)
        ix, iy, iz = np.unravel_index(ones,arr.shape)
        for a,b,c in zip(ix.tolist(),iy.tolist(),iz.tolist()):
            sampled.append((int(x*eng.side+a),int(y*eng.side+b),int(z*eng.side+c),int(seq)))
    sampled = sampled[:max_points]
    bounds = [max(1,max((pt[axis] for pt in sampled),default=0)) for axis in range(3)]
    meta = eng.status()
    payload = json.dumps({'points':sampled,'maxes':bounds,'status':meta},separators=(',',':')).replace('</','<\\/')
    template = '''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>JARVIS-X 3D Gigacube — Verified Bit State</title>
<style>*{box-sizing:border-box}body{margin:0;background:#050b13;color:#dcecff;font-family:system-ui,Arial;overflow:hidden}#viewer{position:absolute;inset:0}#panel{position:absolute;top:14px;left:14px;max-width:375px;background:rgba(5,15,28,.90);border:1px solid #31536d;border-radius:12px;padding:16px;z-index:3;line-height:1.5}h2{margin:0 0 6px;font-size:17px;letter-spacing:.04em;color:#67e8f9}small{color:#97b3c8}p{font-size:12px;margin:6px 0;overflow-wrap:anywhere}button{background:#113049;color:#dbf8ff;border:1px solid #297d94;border-radius:7px;padding:7px 12px;cursor:pointer}button.selected{background:#0c6674}#loading{position:absolute;bottom:16px;left:16px;color:#8fa6c0;z-index:2;font-size:12px}@media(max-width:550px){#panel{top:8px;left:8px;right:8px;max-width:none;padding:10px}p{margin:3px 0}}</style></head><body><div id="viewer"></div><div id="panel"><h2>JARVIS-X / 3D GIGACUBE</h2><small>Real binary-state sample · virtual 99-bit address space</small><p id="state"></p><p id="verification"></p><p><button class="selected" id="lattice">Lattice</button> <button id="torus">Torus projection</button></p><small>Drag to orbit • mouse wheel to zoom • torus is a projection, not quantum hardware</small></div><div id="loading">3D rendering requires an internet connection for Three.js.</div><script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script><script>
const DATA=__DATA__;
document.getElementById('state').textContent='Iteration '+DATA.status.iteration+' · '+DATA.status.allocated_bricks+' allocated bricks · '+DATA.status.one_bits+' one-bits · '+DATA.points.length+' sampled points';
document.getElementById('verification').textContent='State SHA-256: '+DATA.status.state_sha256+' | original payload SHA-256: '+DATA.status.source_sha256;
if(typeof THREE!=='undefined'){
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});renderer.setPixelRatio(Math.min(window.devicePixelRatio,2));renderer.setSize(window.innerWidth,window.innerHeight);document.getElementById('viewer').appendChild(renderer.domElement);
 const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(60,innerWidth/innerHeight,.1,300);let angle=.6,pitch=.35,dist=42;
 const grid=new THREE.GridHelper(37,18,0x1d506a,0x173047);grid.position.y=-15;scene.add(grid);const axes=new THREE.AxesHelper(19);scene.add(axes);
 const geo=new THREE.BufferGeometry();const pos=new Float32Array(DATA.points.length*3),clr=new Float32Array(DATA.points.length*3);let mode='lattice';const mat=new THREE.PointsMaterial({size:.22,vertexColors:true,sizeAttenuation:true});const pts=new THREE.Points(geo,mat);scene.add(pts);
 function project(){const mx=DATA.maxes; DATA.points.forEach((p,i)=>{let x,y,z;if(mode==='lattice'){x=26*(p[0]/mx[0]-.5);y=26*(p[1]/mx[1]-.5);z=26*(p[2]/mx[2]-.5);}else{const u=2*Math.PI*((p[0]/(mx[0]+1)+p[2]/(mx[2]+1))%1),v=2*Math.PI*(p[1]/(mx[1]+1));x=(11+4*Math.cos(v))*Math.cos(u);y=4*Math.sin(v);z=(11+4*Math.cos(v))*Math.sin(u)}pos.set([x,y,z],i*3);const f=p[3]%7/7;clr.set([.2+.4*f,.7+.2*(1-f),.95],i*3)});geo.setAttribute('position',new THREE.BufferAttribute(pos,3));geo.setAttribute('color',new THREE.BufferAttribute(clr,3));grid.visible=(mode==='lattice');}
 project();for(const m of ['lattice','torus'])document.getElementById(m).onclick=()=>{mode=m;document.querySelectorAll('button').forEach(b=>b.classList.remove('selected'));document.getElementById(m).classList.add('selected');project()};
 let drag=false,lastX=0,lastY=0;renderer.domElement.addEventListener('pointerdown',e=>{drag=true;lastX=e.clientX;lastY=e.clientY;renderer.domElement.setPointerCapture(e.pointerId)});renderer.domElement.addEventListener('pointerup',()=>drag=false);renderer.domElement.addEventListener('pointermove',e=>{if(!drag)return;angle+=(e.clientX-lastX)*.007;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-lastY)*.007));lastX=e.clientX;lastY=e.clientY});renderer.domElement.addEventListener('wheel',e=>{dist=Math.max(10,Math.min(90,dist+e.deltaY*.025))},{passive:true});
 const tick=()=>{requestAnimationFrame(tick);camera.position.set(dist*Math.sin(angle)*Math.cos(pitch),dist*Math.sin(pitch),dist*Math.cos(angle)*Math.cos(pitch));camera.lookAt(0,0,0);renderer.render(scene,camera)};tick();window.addEventListener('resize',()=>{renderer.setSize(innerWidth,innerHeight);camera.aspect=innerWidth/innerHeight;camera.updateProjectionMatrix()});document.getElementById('loading').textContent='Rendering actual sampled 1-bits from stored state.';
}</script></body></html>'''
    Path(output).write_text(template.replace('__DATA__',payload),encoding='utf-8')
    return {'html':str(output),'sampled_one_bits':len(sampled),'actual_one_bits':meta['one_bits']}


def selftest() -> dict:
    tests = []
    def check(name, assertion):
        if not assertion: raise AssertionError(name)
        tests.append(name)
    rng = np.random.default_rng(1234)
    check('morton canonical 1,2,3 -> 53',morton_encode(1,2,3)==53)
    for coords in ((0,0,0),(1,2,3),(AXIS_BITS-1,AXIS_BITS-2,AXIS_BITS-3)):
        check('99-bit Morton roundtrip '+str(coords),morton_decode(morton_encode(*coords))==coords)
    with tempfile.TemporaryDirectory(prefix='gigacube_test_') as td:
        # Exact reversible forward/inverse for different tile counts + input sizes.
        for side,size,k in ((8,17,0),(8,64,1),(8,65,3),(8,154,5),
                            (16,512,2),(16,1037,3),(32,4121,2)):
            file = Path(td) / f'input_{side}_{size}_{k}.bin'
            db = Path(td) / f'state_{side}_{size}_{k}.sqlite'
            outfile = Path(td) / f'out_{side}_{size}_{k}.bin'
            file.write_bytes(rng.integers(0,256,size,dtype=np.uint8).tobytes())
            init = initialize(file,db,side=side,max_mib=2)
            with Gigacube(db) as core:
                check(f'input SHA side={side} size={size}',core.meta['input_sha256']==sha_file(file))
                start_digest=core.meta['state_sha256']
                core.iterations(k,quiet=True)
                result=core.recover(outfile)
                check(f'forward+inverse exact side={side} size={size} k={k}',result['exact_reconstruction'] and file.read_bytes()==outfile.read_bytes())
                # inverse applied to actual primary DB should also work
                core.iterations(k,reverse=True,quiet=True)
                restored_digest=core.status()['state_sha256']
                check(f'in-place inverse side={side} size={size} k={k}',restored_digest==start_digest)
        # Exercise a tile boundary specifically: only one set bit on the x+ border,
        # allocated x-adjacent brick. The deterministic even sweep crosses it.
        boundaryfile=Path(td)/'boundary.bin'
        data=bytearray(8**3//8*2)
        # The Morton tile sequence 0 -> (0,0,0), 1 -> (1,0,0).
        bit=((7*8+4)*8+4)  # (x=7, y=4, z=4): odd source -> even neighbor
        data[bit//8] |= 1<<(bit%8)
        boundaryfile.write_bytes(data)
        boundarydb=Path(td)/'boundary.sqlite'
        initialize(boundaryfile,boundarydb,side=8)
        with Gigacube(boundarydb) as core:
            core.iterations(1,quiet=True)
            check('cross-tile flux changed data in neighboring tile',
                  core.get_raw((1,0,0))!=bytes(core.pagebytes))
            core.iterations(1,reverse=True,quiet=True)
            recovered=Path(td)/'boundary.out'
            core.export_bytes(recovered)
            check('cross-tile exact inversion',recovered.read_bytes()==boundaryfile.read_bytes())
        with Gigacube(boundarydb) as core:
            htmlfile=Path(td)/'viewer.html'
            sample=render_html(core,htmlfile,max_points=100)
            check('3D HTML generated',htmlfile.is_file() and 'THREE' in htmlfile.read_text())
        # Exercise every face direction, not just x; 3D Morton brick order:
        # x-neighbor sequence=1, y-neighbor=2, z-neighbor=4.
        for axis, neighbor_seq, source_xyz, target in (
            ('y', 2, (4,7,4), (0,1,0)),
            ('z', 4, (4,4,7), (0,0,1)),
        ):
            payload=bytearray((neighbor_seq+1)*64)
            idx=(source_xyz[0]*8+source_xyz[1])*8+source_xyz[2]
            payload[idx//8] |= 1<<(idx%8)
            sf=Path(td)/f'face_{axis}.bin';sf.write_bytes(payload)
            sd=Path(td)/f'face_{axis}.sqlite';initialize(sf,sd,side=8)
            with Gigacube(sd) as core:
                core.iterations(1,quiet=True)
                check(f'{axis}-face halo transmitted actual bit',core.get_raw(target)!=bytes(64))
                core.iterations(1,reverse=True,quiet=True)
                rec=Path(td)/f'face_{axis}.out';core.export_bytes(rec)
                check(f'{axis}-face inverse recovers source',rec.read_bytes()==sf.read_bytes())

        # Eight connected bricks fill a 2x2x2 genuinely three-dimensional cube.
        full_input=Path(td)/'eight_3d.bin'
        full_input.write_bytes(rng.integers(0,256,512,dtype=np.uint8).tobytes())
        whole_db=Path(td)/'eight_3d.sqlite';initialize(full_input,whole_db,side=8)
        with Gigacube(whole_db) as core:
            check('8 bricks cover all 3 spatial directions', set(core.coords)=={
                (x,y,z) for x in (0,1) for y in (0,1) for z in (0,1)})
            core.iterations(10,quiet=True)
            rec=Path(td)/'eight_3d.out'
            check('10-step 3D inverse',core.recover(rec)['exact_reconstruction'])

        # Restart/resume must match uninterrupted execution exactly.
        interrupted=Path(td)/'restart.sqlite'
        uninterrupted=Path(td)/'uninterrupted.sqlite'
        initialize(full_input,interrupted,side=8)
        initialize(full_input,uninterrupted,side=8)
        with Gigacube(interrupted) as core: core.iterations(3,quiet=True)
        with Gigacube(interrupted) as core:
            core.iterations(5,quiet=True)
            resumed=core.status()['state_sha256']
        with Gigacube(uninterrupted) as core:
            core.iterations(8,quiet=True)
            continuous=core.status()['state_sha256']
        check('checkpoint resume equals continuous 8 iterations',resumed==continuous)

        # Explicit injected phase failure must roll back partial writes and iteration.
        with Gigacube(uninterrupted) as core:
            start=core.status()['state_sha256'];start_index=core.meta['iteration']
            original_phase=core.phase
            def fail_after_one_phase(parity):
                if parity==1: raise RuntimeError('injected phase failure')
                return original_phase(parity)
            core.phase=fail_after_one_phase
            try: core.iterations(1,quiet=True)
            except RuntimeError: pass
            else: raise AssertionError('injected phase failure did not raise')
            check('failed iteration rolls back both phases',core.status()['state_sha256']==start and core.meta['iteration']==start_index)

        # A tampered brick must not pass original SHA check.
        conn=sqlite3.connect(boundarydb)
        raw=conn.execute('SELECT data FROM tiles WHERE seq=0').fetchone()[0]
        changed=bytearray(raw);changed[0]^=1
        conn.execute('UPDATE tiles SET data=? WHERE seq=0',(bytes(changed),))
        conn.commit();conn.close()
        with Gigacube(boundarydb) as core:
            result=core.recover(Path(td)/'tampered.out')
            check('bit flip detection by SHA-256',not result['exact_reconstruction'])
    return {'passed':len(tests),'tests':tests}


def main(argv=None):
    p=argparse.ArgumentParser(description='JARVIS-X sparse 3D gigascale reversible bit engine')
    commands=p.add_subparsers(dest='command',required=True)
    a=commands.add_parser('init',help='pack an input file into disk-backed 3D bricks')
    a.add_argument('--input',required=True);a.add_argument('--db',required=True)
    a.add_argument('--brick',type=int,default=64);a.add_argument('--max-mib',type=float,default=64)
    a=commands.add_parser('iterate',help='perform N reversible iterations on stored tiles')
    a.add_argument('--db',required=True);a.add_argument('--count',type=int,default=1)
    a=commands.add_parser('reverse',help='undo N reversible iterations in stored tiles')
    a.add_argument('--db',required=True);a.add_argument('--count',type=int,default=1)
    a=commands.add_parser('recover',help='recover original bytes from an independently reversed snapshot')
    a.add_argument('--db',required=True);a.add_argument('--output',required=True)
    a=commands.add_parser('status',help='view logical and measured storage statistics')
    a.add_argument('--db',required=True)
    a=commands.add_parser('render',help='create interactive 3D HTML visualizer with actual state')
    a.add_argument('--db',required=True);a.add_argument('--output',required=True)
    a.add_argument('--points',type=int,default=4000)
    a=commands.add_parser('selftest',help='execute randomized, boundary and corruption tests')
    args=p.parse_args(argv)
    if args.command=='init': answer=initialize(args.input,args.db,args.brick,args.max_mib)
    elif args.command=='selftest': answer=selftest()
    else:
        with Gigacube(args.db) as eng:
            if args.command=='iterate': answer={'trace':eng.iterations(args.count)}
            elif args.command=='reverse': answer={'trace':eng.iterations(args.count,reverse=True)}
            elif args.command=='recover': answer=eng.recover(args.output)
            elif args.command=='status': answer=eng.status()
            elif args.command=='render': answer=render_html(eng,args.output,args.points)
    print(json.dumps(answer,indent=2))
    if args.command=='recover' and not answer['exact_reconstruction']:
        raise SystemExit(2)

if __name__=='__main__':
    try: main()
    except (ValueError,sqlite3.Error,OSError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        sys.exit(1)
