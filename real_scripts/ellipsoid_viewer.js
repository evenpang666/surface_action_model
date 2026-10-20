/* Replay ellipsoids in the existing Plotly scene, independent of voxel/point toggles. */
class EllipsoidViewer {
  constructor(onChange) {
    this.revision=0;
    this.element=document.createElement('div');
    this.element.style.pointerEvents='auto';
    this.element.innerHTML=`<div style="margin-top:7px;border-top:1px solid #536171;padding-top:6px">
      <b>回放椭球模型</b>
      <label><input id="ellipsoid-obstacles" type="checkbox">障碍椭球（O）</label>
      <label><input id="ellipsoid-robot" type="checkbox">连杆椭球（L）</label>
      <label>绘制 <select id="ellipsoid-style"><option value="surface">半透明表面</option><option value="wire">线框</option></select></label>
      <div style="color:#9eb4ca;font-size:12px">绿：障碍 · 蓝：左臂 · 紫：右臂；完整碰撞包络，未叠加安全距离</div>
      <div style="color:#9eb4ca;font-size:12px">每个连杆 / 整个夹爪各一个椭球；B 关闭时拟合全部占据体素，开启时仅拟合筛选体素</div>
      <div id="ellipsoid-status" style="color:#9eb4ca;font-size:12px"></div></div>`;
    this.obstacles=this.element.querySelector('#ellipsoid-obstacles');
    this.robot=this.element.querySelector('#ellipsoid-robot');
    this.style=this.element.querySelector('#ellipsoid-style');
    this.style.value='surface';
    this.status=this.element.querySelector('#ellipsoid-status');
    for(const input of [this.obstacles,this.robot,this.style])input.addEventListener('change',()=>{this.revision++;onChange()});
    document.addEventListener('keydown',e=>{
      if(e.repeat||['INPUT','TEXTAREA','SELECT','BUTTON','A'].includes(e.target?.tagName))return;
      const input=e.code==='KeyO'?this.obstacles:e.code==='KeyL'?this.robot:null;
      if(input){e.preventDefault();input.checked=!input.checked;this.revision++;onChange()}
    });
  }
  groups(){return [this.obstacles.checked?'obstacles':'',this.robot.checked?'robot':''].filter(Boolean).join(',')}
  update(view,filtered,frozen){
    const scope=frozen?(filtered?'frozen_selection':'frozen_all_cells'):(filtered?'selected_cells':'all_cells');
    this.current=view?.scope===scope?view:null;
    if(!this.groups()){this.status.textContent='按需计算；开启显示后建立椭球';return}
    if(view?.error){this.status.textContent='椭球显示失败：'+view.error;return}
    if(!this.current){this.status.textContent='正在建立当前范围的椭球…';return}
    const labels={all_cells:'全部实时占据体素',selected_cells:'仅筛选单元内实时占据体素',frozen_all_cells:'全部冻结占据体素（非实时）',frozen_selection:'仅冻结筛选体素（非实时）'};
    this.status.textContent=`${labels[scope]} · 障碍 ${view.obstacles.length} · 连杆/夹爪 ${view.robot.length} 个椭球`+(view.messages?.length?' · '+view.messages.join('；'):'');
  }
  point(body,u,transform){
    return transform(body.center.map((c,r)=>c+body.axes[r].reduce((s,a,j)=>s+a*body.radii[j]*u[j],0)));
  }
  *boundsPoints(){
    if(!this.current||!this.groups())return;
    for(const body of this.selected()){
      // Axis-aligned bounds in source coordinates, enclosing rotated ellipsoids.
      const h=body.axes.map(row=>Math.hypot(...row.map((a,j)=>a*body.radii[j])));
      for(const x of [-1,1])for(const y of [-1,1])for(const z of [-1,1])yield body.center.map((c,i)=>c+[x,y,z][i]*h[i]);
    }
  }
  selected(){return this.current?[...(this.obstacles.checked?this.current.obstacles:[]),...(this.robot.checked?this.current.robot:[])]:[]}
  traces(transform){
    const result=[];
    for(const [kind,items,color] of [
      ['障碍椭球',this.obstacles.checked?this.current?.obstacles:[],'#45ff9b'],
      ['左臂椭球',this.robot.checked?this.current?.robot.filter(b=>b.arm===0):[],'#50bfff'],
      ['右臂椭球',this.robot.checked?this.current?.robot.filter(b=>b.arm===1):[],'#c690ff']]){
      if(!items?.length)continue;
      const x=[],y=[],z=[],i=[],j=[],k=[],names=[];
      const push=(b,u)=>{const p=this.point(b,u,transform);x.push(p[0]);y.push(p[1]);z.push(p[2]);names.push(b.key+'<br>半轴 '+b.radii.map(r=>(r*1000).toFixed(1)).join(' / ')+' mm')};
      for(const body of items){
        if(this.style.value==='wire'){
          // Latitude and longitude lines reveal individual envelope boundaries.
          for(let axis=0;axis<3;axis++)for(const latitude of [-.65,0,.65]){
            for(let t=0;t<=32;t++){
              const u=[0,0,0],r=Math.sqrt(1-latitude**2),a=t*Math.PI/16;
              u[axis]=latitude;u[(axis+1)%3]=r*Math.cos(a);u[(axis+2)%3]=r*Math.sin(a);push(body,u);
            }
            x.push(null);y.push(null);z.push(null);names.push('');
          }
        }else{
          const base=x.length,nlat=12,nlon=24;
          for(let lat=0;lat<=nlat;lat++)for(let lon=0;lon<nlon;lon++){
            const a=lat*Math.PI/nlat,b=lon*2*Math.PI/nlon;
            push(body,[Math.sin(a)*Math.cos(b),Math.sin(a)*Math.sin(b),Math.cos(a)]);
          }
          for(let lat=0;lat<nlat;lat++)for(let lon=0;lon<nlon;lon++){
            const a=base+lat*nlon+lon,b=base+lat*nlon+(lon+1)%nlon,c=a+nlon,d=b+nlon;
            if(lat>0){i.push(a);j.push(c);k.push(b)}
            if(lat<nlat-1){i.push(b);j.push(c);k.push(d)}
          }
        }
      }
      const common={name:kind,x,y,z,text:names,hovertemplate:kind+'<br>%{text}<extra></extra>'};
      result.push(this.style.value==='wire'?{...common,type:'scatter3d',mode:'lines',line:{color,width:2}}:
        {...common,type:'mesh3d',i,j,k,color,opacity:.28,flatshading:false,lighting:{ambient:.7,diffuse:.5,specular:.15}});
    }
    return result;
  }
}
