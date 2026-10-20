/* Native OcTree leaves, in scene coordinates. Geometry never enters motion checks. */
class OctomapLeafViewer {
  constructor(onChange, prefix='octomap') {
    this.revision = 0;
    this.element = document.createElement('div');
    this.element.id = `${prefix}-controls`;
    this.element.hidden = true;
    this.element.style.pointerEvents = 'auto';
    this.element.style.maxWidth = '760px';
    this.element.innerHTML = `<div style="margin-top:7px;border-top:1px solid #536171;padding-top:6px">
      <b>OctoMap 八叉树</b>
      <label>显示 <select id="${prefix}-style"><option value="native">原生叶节点</option><option value="boxes">兼容体素框</option></select></label>
      <label>着色 <select id="${prefix}-color"><option value="probability">占据概率</option><option value="depth">树深度</option><option value="state">占据/空闲</option></select></label><br>
      <label><input id="${prefix}-free" type="checkbox"><span id="${prefix}-free-label">显示已观测空闲节点</span></label>
      <label><input id="${prefix}-points" type="checkbox" checked>显示点云</label>
      <div id="${prefix}-status" style="color:#9eb4ca;font-size:12px"></div>
      <div id="${prefix}-legend" style="color:#9eb4ca;font-size:12px"></div>
    </div>`;
    this.style = this.element.querySelector(`#${prefix}-style`);
    this.color = this.element.querySelector(`#${prefix}-color`);
    this.free = this.element.querySelector(`#${prefix}-free`);
    this.points = this.element.querySelector(`#${prefix}-points`);
    this.status = this.element.querySelector(`#${prefix}-status`);
    this.legend = this.element.querySelector(`#${prefix}-legend`);
    for (const input of this.element.querySelectorAll('select,input')) {
      input.style.pointerEvents = 'auto';
      input.style.margin = '2px 5px';
      input.addEventListener('change', () => { this.revision++; onChange(); });
    }
  }

  useNative(view, filtered) {
    return Boolean(view && !filtered && this.style.value === 'native');
  }

  update(view, filtered, showVoxels) {
    this.element.hidden = !view;
    if (!view) return;
    this.style.disabled = filtered;
    this.color.disabled = filtered || this.style.value !== 'native';
    this.free.disabled = filtered || this.style.value !== 'native';
    const occupied = view.occupied, free = view.free;
    const depths = [...Object.keys(occupied.depth_counts), ...Object.keys(free.depth_counts)].map(Number);
    const minDepth = depths.length ? Math.min(...depths) : view.max_depth;
    const maxDepth = depths.length ? Math.max(...depths) : view.max_depth;
    const size = depth => (view.resolution_m * 2 ** (view.max_depth - depth) * 1000).toFixed(0);
    const sampled = occupied.sampled || (this.free.checked && free.sampled);
    const note = filtered ? ' · 人工筛选使用精确选中体素框' :
      this.style.value !== 'native' ? ' · 当前显示兼容体素框' :
      `${sampled ? ' · 显示已抽样，地图完整' : ''} · 未观测区域不绘制`;
    this.status.textContent = `占据叶 ${occupied.shown}/${occupied.count} · 空闲叶 ${free.shown}/${free.count}${this.free.checked ? '' : '（隐藏）'} · 深度 ${minDepth}–${maxDepth} · 边长 ${size(maxDepth)}–${size(minDepth)} mm${note}${showVoxels ? '' : ' · 体素已隐藏'}`;
    this.legend.textContent = this.color.value === 'state' ? '橙：占据 · 蓝：已观测空闲' :
      this.color.value === 'probability' ? '颜色表示占据概率 0–1；≥0.5 为占据。悬停查看边长、概率和深度。' :
      '颜色表示原生叶节点深度；层级越深，节点越小。悬停查看边长和概率。';
  }

  corners(view, group, index) {
    const center = group.centers[index], half = group.sizes_m[index] / 2, axes = view.axes;
    return [[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]].map(
      sign => center.map((v, row) => v + half * (axes[row][0]*sign[0] + axes[row][1]*sign[1] + axes[row][2]*sign[2])));
  }

  *boundsPoints(view) {
    for (const name of this.free.checked ? ['occupied','free'] : ['occupied']) {
      const group = view[name];
      for (let index = 0; index < group.shown; index++) yield* this.corners(view, group, index);
    }
  }

  traces(view, transform) {
    const triangles = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]];
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    const result = [];
    let hasColorbar = false;
    const depthKeys = [...Object.keys(view.occupied.depth_counts), ...Object.keys(view.free.depth_counts)].map(Number);
    const minDepth = depthKeys.length ? Math.min(...depthKeys) : 0;
    for (const name of this.free.checked ? ['occupied','free'] : ['occupied']) {
      const group = view[name];
      if (!group.shown) continue;
      const x=[], y=[], z=[], i=[], j=[], k=[], intensity=[], customdata=[], ex=[], ey=[], ez=[];
      for (let index=0; index<group.shown; index++) {
        const base=x.length, corners=this.corners(view, group, index).map(transform);
        const data=[group.probabilities[index], group.depths[index], group.sizes_m[index]*1000];
        for (const point of corners) {
          x.push(point[0]); y.push(point[1]); z.push(point[2]); customdata.push(data);
          intensity.push(this.color.value==='depth' ? group.depths[index] : group.probabilities[index]);
        }
        for (const t of triangles) { i.push(base+t[0]); j.push(base+t[1]); k.push(base+t[2]); }
        for (const [a,b] of edges) {
          ex.push(corners[a][0],corners[b][0],null); ey.push(corners[a][1],corners[b][1],null); ez.push(corners[a][2],corners[b][2],null);
        }
      }
      const occupied=name==='occupied', label=occupied?'OctoMap 占据叶':'OctoMap 空闲叶';
      const mesh={type:'mesh3d',name:label,x,y,z,i,j,k,customdata,flatshading:true,
        opacity:occupied ? .72 : .10, color:occupied?'#ff9d45':'#57baff',
        hovertemplate:label+'<br>占据概率 %{customdata[0]:.3f}<br>树深度 %{customdata[1]}<br>边长 %{customdata[2]:.1f} mm<extra></extra>',
        lighting:{ambient:.8,diffuse:.4,specular:0}};
      if (this.color.value!=='state') {
        Object.assign(mesh,{intensity,intensitymode:'vertex',
          cmin:this.color.value==='depth'?minDepth:0, cmax:this.color.value==='depth'?Math.max(minDepth+1,view.max_depth):1,
          colorscale:this.color.value==='depth'?'Turbo':[[0,'#2166ac'],[.49,'#b9def2'],[.5,'#ffe6a6'],[1,'#c43d16']],
          showscale:!hasColorbar,
          colorbar:{title:{text:this.color.value==='depth'?'树深度':'占据概率',font:{color:'#e6edf3'}},tickfont:{color:'#e6edf3'},thickness:12,len:.42}});
        hasColorbar=true;
      }
      result.push(mesh,{type:'scatter3d',mode:'lines',name:label+'边界',x:ex,y:ey,z:ez,
        line:{color:occupied?'#ffd29b':'#57baff',width:occupied?1.5:1},opacity:occupied ? .8 : .28,hoverinfo:'skip'});
    }
    return result;
  }
}
