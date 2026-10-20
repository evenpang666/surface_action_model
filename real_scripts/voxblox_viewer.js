/* Reuse cube geometry; signed fields remain distinct from occupancy probability. */
class VoxbloxViewer extends OctomapLeafViewer {
  constructor(onChange) {
    super(onChange,'voxblox');
    this.element.querySelector('b').textContent='Voxblox TSDF / ESDF';
    this.style.querySelector('option[value="native"]').textContent='原生距离场体素';
    this.style.value='native';
    this.color.innerHTML='<option value="distance">有符号距离</option><option value="weight">源 TSDF 权重</option><option value="state">表面/距离场</option>';
    this.color.value='distance';
    this.points.checked=true;
    this.element.querySelector('#voxblox-free-label').textContent='显示其余已观测距离场';
    const label=document.createElement('label');
    label.textContent='距离场 ';
    this.field=document.createElement('select');
    this.field.innerHTML='<option value="tsdf">TSDF 截断距离</option><option value="esdf">ESDF 欧氏距离</option>';
    this.field.value='tsdf';
    label.append(this.field);this.element.append(label);
    this.field.addEventListener('change',()=>{this.revision++;onChange();});
  }

  isEsdf(view) { return this.field.value==='esdf'&&!!view.esdf; }

  update(view,filtered,showVoxels) {
    this.element.hidden=!view;
    if(!view)return;
    if(!view.esdf)this.field.value='tsdf';
    this.field.disabled=filtered||!view.esdf||this.style.value!=='native';
    this.style.disabled=filtered;
    this.color.disabled=this.free.disabled=filtered||this.style.value!=='native';
    const esdf=this.isEsdf(view), selected=esdf?view.esdf:view;
    const surface=selected.surface, field=selected.field, kind=esdf?'ESDF':'TSDF';
    const sampled=surface.sampled||(this.free.checked&&field.sampled);
    const note=filtered?' · 人工筛选使用精确兼容体素框':
      this.style.value!=='native'?' · 显示 TSDF 表面与当前端点的兼容体素框':
      `${sampled?' · 显示已抽样，距离场完整':''} · 未观测区域不绘制`;
    this.status.textContent=`${kind} 表面 ${surface.shown}/${surface.count} · 其余距离场 ${field.shown}/${field.count}${this.free.checked?'':'（隐藏）'} · 体素 ${(view.resolution_m*1000).toFixed(0)} mm · 表面带 ±${(view.surface_band_m*1000).toFixed(0)} mm${esdf?` · 距离上限 ${view.esdf_max_distance_m} m`:''}${note}${showVoxels?'':' · 体素已隐藏'}`;
    this.legend.textContent=this.color.value==='weight'?'颜色表示源 TSDF 融合权重；悬停查看距离、权重和边长。':
      this.color.value==='state'?`橙：${kind} 表面带 · 蓝：其余已观测距离场。当前端点补充框可切换到兼容体素框查看。`:
      `蓝：表面后方（负距离）· 白：零距离表面 · 红：表面前方（正距离）。${esdf?'原生 full Euclidean ESDF；到达上限的值为饱和值。':'TSDF 在截断距离处饱和。'}`;
  }

  geometryView(view) {
    // The shared renderer uses two scalar channels for vertex/hover data.
    // Substitute native metres and source weights, then set field legends below.
    const group=g=>({...g,probabilities:g.distances_m,depths:g.weights,depth_counts:{0:g.count}});
    const selected=this.isEsdf(view)?view.esdf:view;
    return {...view,max_depth:1,occupied:group(selected.surface),free:group(selected.field)};
  }

  *boundsPoints(view) { yield* super.boundsPoints(this.geometryView(view)); }

  traces(view,transform) {
    const traces=super.traces(this.geometryView(view),transform);
    const esdf=this.isEsdf(view), kind=esdf?'ESDF':'TSDF';
    for(const trace of traces) {
      const surface=trace.name.startsWith('OctoMap 占据叶');
      const label=surface?`Voxblox ${kind} 表面体素`:`Voxblox ${esdf?'ESDF ':''}已观测距离场`;
      trace.name=label+(trace.type==='mesh3d'?'':'边界');
      if(trace.type!=='mesh3d')continue;
      trace.hovertemplate=label+`<br>${kind} %{customdata[0]:.4f} m<br>源 TSDF 权重 %{customdata[1]:.2f}<br>边长 %{customdata[2]:.1f} mm<extra></extra>`;
      if(this.color.value==='state')continue;
      const weight=this.color.value==='weight';
      trace.intensity=trace.customdata.map(d=>d[weight?1:0]);
      const range=esdf?view.esdf_max_distance_m:view.truncation_m;
      trace.cmin=weight?0:-range;
      trace.cmax=weight?view.max_weight:range;
      trace.colorscale=weight?'Viridis':[[0,'#2166ac'],[.5,'#f7f7f7'],[1,'#b2182b']];
      trace.colorbar.title.text=weight?'源 TSDF 权重':`${kind} (m)`;
    }
    return traces;
  }
}
