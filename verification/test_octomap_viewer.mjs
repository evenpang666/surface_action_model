// Run with: node verification/test_octomap_viewer.mjs
// Exercise the real viewer scripts with a small DOM/Plotly adapter; no browser,
// camera, robot, npm dependencies, or network access is required.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

class Element {
  constructor() { this.style={}; this.listeners={}; this.controls={}; this.textContent=''; }
  querySelector(selector) {
    if (!this.controls[selector]) {
      const input=new Element();
      input.value=selector.endsWith('-style')?'native':selector.endsWith('-color')?'probability':'';
      input.checked=selector.endsWith('-points');
      this.controls[selector]=input;
    }
    return this.controls[selector];
  }
  querySelectorAll() { return Object.entries(this.controls).filter(([s])=>/-(style|color|free|points)$/.test(s)).map(([,e])=>e); }
  addEventListener(event, handler) { this.listeners[event]=handler; }
  setAttribute() {}
  append() {}
}
const elements=new Map();
const element=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const draws=[];
const context=vm.createContext({
  document:{createElement:()=>new Element(),getElementById:element,querySelector:element,addEventListener(){},visibilityState:'visible'},
  Plotly:{react:async(_element,traces,layout)=>draws.push({traces,layout})},
  location:{search:''}, URLSearchParams, AbortController,
  performance:{now:()=>performance.now()}, fetch:async()=>({ok:true,status:204}),setTimeout:()=>0,
});
const helper=await readFile(new URL('../real_scripts/octomap_viewer.js',import.meta.url),'utf8');
const source=await readFile(new URL('../real_scripts/live_dual_ur7e_obstacle_model.py',import.meta.url),'utf8');
const html=source.match(/VIEWER_HTML = r"""([\s\S]*?)"""/)[1];
assert.ok(html.includes('<script src="/octomap-viewer.js"></script>'));
vm.runInContext(helper,context);
vm.runInContext(await readFile(new URL('../real_scripts/voxblox_viewer.js',import.meta.url),'utf8'),context);
vm.runInContext(await readFile(new URL('../real_scripts/ellipsoid_viewer.js',import.meta.url),'utf8'),context);
vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
const viewer=vm.runInContext('octomapViewer',context);
const group=(center,size,probability,depth)=>({count:1,shown:1,sampled:false,
  centers:[center],sizes_m:[size],probabilities:[probability],depths:[depth],depth_counts:{[depth]:1}});
const view={max_depth:16,resolution_m:.05,axes:[[0,-1,0],[1,0,0],[0,0,1]],
  occupied:group([2,3,.5],.2,.8,14),free:group([-1,1,1],.4,.2,13)};
const corners=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]];
context.snapshot={frame:1,host_timestamp_ns:1,status:'ok',environment:{points:[[0,0,0]],colors:[[10,20,30]]},
  robot:{points:[],colors:[]},obbs:[{corners}],diagnostics:{table_normal_left_base:[0,0,1],table_offset_m:0},octomap_view:view};
const draw=async()=>{await vm.runInContext('draw(snapshot)',context);return draws.at(-1).traces;};
let traces=await draw();
assert.equal(viewer.element.hidden,false);
let mesh=traces.find(t=>t.type==='mesh3d');
assert.equal(mesh.name,'OctoMap 占据叶');
assert.equal(mesh.x.length,8);
assert.equal(mesh.i.length,12);
assert.ok(mesh.i.concat(mesh.j,mesh.k).every(i=>i>=0&&i<8));
assert.ok(Math.abs(Math.min(...mesh.x)-1.9)<1e-12);
assert.ok(Math.abs(Math.max(...mesh.x)-2.1)<1e-12);
assert.ok(Math.abs(Math.max(...mesh.z)-.6)<1e-12);
assert.equal(mesh.customdata[0][2],200);
assert.equal(mesh.intensity[0],.8);
assert.equal(traces.filter(t=>t.type==='mesh3d').length,1);
assert.ok(viewer.status.textContent.includes('深度 13–14'));

viewer.free.checked=true;
viewer.free.listeners.change();
traces=await draw();
assert.equal(viewer.revision,1);
assert.equal(traces.filter(t=>t.type==='mesh3d').length,2);
const free=traces.find(t=>t.name==='OctoMap 空闲叶');
assert.ok(Math.abs(Math.max(...free.x)-Math.min(...free.x)-.4)<1e-12);
assert.equal(free.customdata[0][0],.2);
assert.equal(traces.filter(t=>t.showscale).length,1);

viewer.color.value='depth';
traces=await draw();
mesh=traces.find(t=>t.type==='mesh3d');
assert.equal(mesh.intensity[0],14);
assert.equal(mesh.cmin,13);
assert.equal(mesh.cmax,16);
viewer.color.value='state';
traces=await draw();
assert.equal(traces.find(t=>t.type==='mesh3d').color,'#ff9d45');
assert.equal(traces.some(t=>t.showscale),false);

viewer.points.checked=false;
traces=await draw();
assert.equal(traces.some(t=>t.name==='environment'),false);
viewer.style.value='boxes';
traces=await draw();
assert.equal(traces.some(t=>t.type==='mesh3d'),false);
assert.ok(traces.some(t=>t.name==='体素框'));
assert.equal(viewer.free.disabled,true);

viewer.style.value='native';
context.snapshot.viewer_filter={state:'applied',obbs:[{corners}],voxel_count:1,annotation_frame:1};
vm.runInContext('filterVoxels=true',context);
traces=await draw();
assert.equal(viewer.style.disabled,true);
assert.equal(traces.some(t=>t.type==='mesh3d'),false);
assert.ok(viewer.status.textContent.includes('人工筛选'));

vm.runInContext('filterVoxels=false;showVoxels=false',context);
traces=await draw();
assert.equal(traces.some(t=>t.type==='mesh3d'),false);
vm.runInContext('showVoxels=true',context);
view.occupied.sampled=true;view.occupied.count=99;
await draw();
assert.ok(viewer.status.textContent.includes('已抽样'));
delete context.snapshot.octomap_view;
traces=await draw();
assert.equal(viewer.element.hidden,true);
assert.ok(traces.some(t=>t.name==='environment'));
assert.ok(traces.some(t=>t.name==='体素框'));
console.log('OctoMap viewer: native geometry, probability/depth/state colors, free/cloud toggles, filtering, sampling and NumPy fallback passed.');

const tsdf=vm.runInContext('voxbloxViewer',context);
const tsdfGroup=(center,distance,weight)=>({count:1,shown:1,sampled:false,centers:[center],
  sizes_m:[.02],distances_m:[distance],weights:[weight]});
context.snapshot.voxblox_view={schema:'voxblox_tsdf_view_v1',axes:[[1,0,0],[0,1,0],[0,0,1]],
  resolution_m:.02,truncation_m:.08,surface_band_m:.02,max_weight:32,
  surface:tsdfGroup([0,0,.5],.01,3),field:tsdfGroup([0,0,.3],.08,12)};
traces=await draw();
assert.equal(tsdf.element.hidden,false);
assert.equal(viewer.element.hidden,true);
mesh=traces.find(t=>t.type==='mesh3d');
assert.equal(mesh.name,'Voxblox TSDF 表面体素');
assert.equal(mesh.intensity[0],.01);
assert.equal(mesh.cmin,-.08);assert.equal(mesh.cmax,.08);
assert.equal(mesh.customdata[0][1],3);
assert.ok(!mesh.hovertemplate.includes('占据概率'));
assert.ok(Math.abs(Math.max(...mesh.x)-Math.min(...mesh.x)-.02)<1e-12);
tsdf.free.checked=true;tsdf.color.value='weight';
traces=await draw();
assert.equal(traces.filter(t=>t.type==='mesh3d').length,2);
assert.equal(traces.find(t=>t.type==='mesh3d').intensity[0],3);
assert.equal(traces.find(t=>t.type==='mesh3d').cmax,32);
tsdf.style.value='boxes';
traces=await draw();
assert.equal(traces.some(t=>t.type==='mesh3d'),false);
vm.runInContext('filterVoxels=true',context);
await draw();
assert.equal(tsdf.style.disabled,true);
console.log('Voxblox viewer: native TSDF metre/weight colors, shared geometry, field toggle, compatibility and filtered display passed.');

vm.runInContext('filterVoxels=false',context);
tsdf.style.value='native';tsdf.color.value='distance';
context.snapshot.voxblox_view.esdf_enabled=true;
context.snapshot.voxblox_view.esdf_max_distance_m=2;
context.snapshot.voxblox_view.esdf={surface:tsdfGroup([0,0,.5],.01,3),
  field:tsdfGroup([0,0,.3],.7,12)};
traces=await draw();
assert.equal(tsdf.field.value,'tsdf'); // ESDF availability never overrides the default.
tsdf.field.value='esdf';tsdf.free.checked=true;tsdf.field.listeners.change();
traces=await draw();
assert.equal(tsdf.field.value,'esdf');
assert.equal(tsdf.free.checked,true);
mesh=traces.find(t=>t.name==='Voxblox ESDF 已观测距离场');
assert.equal(mesh.intensity[0],.7);
assert.equal(mesh.cmin,-2);assert.equal(mesh.cmax,2);
assert.equal(mesh.colorbar.title.text,'ESDF (m)');
assert.ok(mesh.hovertemplate.includes('ESDF'));
assert.ok(!mesh.hovertemplate.includes('概率'));
assert.ok(tsdf.status.textContent.includes('距离上限 2 m'));
assert.ok(tsdf.legend.textContent.includes('full Euclidean'));
tsdf.field.value='tsdf';tsdf.field.listeners.change();
traces=await draw();
mesh=traces.find(t=>t.name==='Voxblox 已观测距离场');
assert.equal(mesh.intensity[0],.08);
assert.equal(mesh.cmax,.08);
assert.equal(mesh.colorbar.title.text,'TSDF (m)');
tsdf.field.value='esdf';tsdf.color.value='weight';
traces=await draw();
assert.equal(traces.find(t=>t.type==='mesh3d').colorbar.title.text,'源 TSDF 权重');
vm.runInContext('filterVoxels=true',context);
traces=await draw();
assert.equal(tsdf.field.disabled,true);
assert.equal(traces.some(t=>t.type==='mesh3d'),false);
vm.runInContext('filterVoxels=false',context);
delete context.snapshot.voxblox_view.esdf;
await draw();
assert.equal(tsdf.field.value,'tsdf');
assert.equal(tsdf.field.disabled,true);
console.log('Voxblox ESDF viewer: Euclidean distance range, TSDF switch, source weights and filtered display passed.');

// Whole-volume overlays share the same table transform and are independently toggled.
const ellipsoids=vm.runInContext('ellipsoidViewer',context);
assert.equal(ellipsoids.groups(),'');
const body=(key,arm)=>({key,arm,link:2,center:[2,3,.5],radii:[.1,.2,.3],
  axes:[[0,-1,0],[1,0,0],[0,0,1]]});
context.snapshot.ellipsoid_view={scope:'selected_cells',robot:[body('left',0),body('right',1)],
  obstacles:[body('obstacle',-1)],messages:[]};
vm.runInContext('filterVoxels=false;showRobot=false;showVoxels=false',context);
traces=await draw();
assert.equal(traces.some(t=>t.name==='障碍椭球'),false);
ellipsoids.obstacles.checked=true;ellipsoids.robot.checked=true;
traces=await draw();
assert.equal(ellipsoids.groups(),'obstacles,robot');
assert.equal(traces.filter(t=>/椭球$/.test(t.name)).length,3);
mesh=traces.find(t=>t.name==='障碍椭球');
assert.equal(mesh.type,'mesh3d');
assert.equal(mesh.opacity,.28);
assert.ok(Math.abs(Math.min(...mesh.x)-1.8)<1e-12);
assert.ok(Math.abs(Math.max(...mesh.y)-3.1)<1e-12);
assert.ok(Math.abs(Math.max(...mesh.z)-.8)<1e-12);
assert.ok(mesh.i.concat(mesh.j,mesh.k).every(i=>i>=0&&i<mesh.x.length));
assert.equal(traces.some(t=>t.name==='dual robots'),false);
assert.equal(traces.some(t=>t.name==='体素框'),false);
assert.ok(ellipsoids.status.textContent.includes('连杆/夹爪 2'));
ellipsoids.style.value='wire';
traces=await draw();
assert.equal(traces.find(t=>t.name==='左臂椭球').type,'scatter3d');
assert.ok(traces.find(t=>t.name==='左臂椭球').x.includes(null));
// B changes voxel display only; the ellipsoid set is always selected-only.
vm.runInContext('filterVoxels=true',context);
traces=await draw();
assert.equal(traces.filter(t=>/椭球$/.test(t.name)).length,3);
context.snapshot.ellipsoid_view.scope='all_occupied';
traces=await draw();
assert.equal(traces.some(t=>/椭球$/.test(t.name)),false);
context.snapshot.ellipsoid_view.scope='selected_cells';
traces=await draw();
assert.equal(traces.filter(t=>/椭球$/.test(t.name)).length,3);
assert.ok(ellipsoids.status.textContent.includes('筛选单元内实时占据体素'));
ellipsoids.robot.checked=false;
traces=await draw();
assert.equal(traces.filter(t=>/椭球$/.test(t.name)).length,1);
context.snapshot.ellipsoid_view={error:'test failure',robot:[],obstacles:[]};
traces=await draw();
assert.equal(traces.some(t=>/椭球$/.test(t.name)),false);
assert.ok(ellipsoids.status.textContent.includes('test failure'));
console.log('Ellipsoid viewer: exact rotated axes, independent toggles, mesh/wire, scope changes and failures passed.');
