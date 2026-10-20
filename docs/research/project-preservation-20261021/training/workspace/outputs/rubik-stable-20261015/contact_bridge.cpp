#include "../rubik-regrasp-20261015/grasp_bridge.cpp"
// Read-only observation of actual solver contacts; no forward() or dynamics edits.
// row: hand, hand body, cubie body, hand geom, cubie geom, pos[3],
// inward normal on cube[3], world force on cube[3], normal force, distance, mu.
extern "C" int cube_contacts(void*p,double*out,int capacity){
 auto*s=static_cast<Simulation*>(p);auto*m=s->model;auto*d=s->data;int count=0;
 for(int i=0;i<d->ncon&&count<capacity;i++){
  auto&c=d->contact[i];if(c.geom[0]<0||c.geom[1]<0)continue;
  for(int k=0;k<2;k++){
   int gb=c.geom[k],gh=c.geom[1-k],cb=m->geom_bodyid[gb],hb=m->geom_bodyid[gh];
   const char*cn=mj_id2name(m,mjOBJ_BODY,cb);const char*hn=mj_id2name(m,mjOBJ_BODY,hb);
   if(!cn||!hn||std::string(cn).rfind("rubik_cubie_",0)!=0)continue;
   int side=std::string(hn).rfind("left_hand_",0)==0||std::string(hn)=="left_wrist_yaw_link"?0:std::string(hn).rfind("right_hand_",0)==0||std::string(hn)=="right_wrist_yaw_link"?1:-1;if(side<0)continue;
   double f[6];mj_contactForce(m,d,i,f);double sign=k==1?1.:-1.;double*row=out+17*count++;
   row[0]=side;row[1]=hb;row[2]=cb;row[3]=gh;row[4]=gb;
   for(int j=0;j<3;j++){row[5+j]=c.pos[j];row[8+j]=sign*c.frame[j];row[11+j]=0;for(int a=0;a<3;a++)row[11+j]+=sign*c.frame[3*a+j]*f[a];}
   row[14]=f[0];row[15]=c.dist;row[16]=c.friction[0];break;
  }
 }
 return count;
}
extern "C" void actual_ctrl(void*p,double*out){auto*s=static_cast<Simulation*>(p);std::copy_n(s->data->ctrl,s->model->nu,out);}
extern "C" void actuator_force(void*p,double*out){auto*s=static_cast<Simulation*>(p);std::copy_n(s->data->actuator_force,s->model->nu,out);}
// Expand only position-command authority to the existing physical joint limits.
// Joint limits, effort limits, gains, masses and contacts are unchanged.
extern "C" void thumb_position_authority(void*p,int enabled){
 auto*s=static_cast<Simulation*>(p);auto*m=s->model;
 for(int j=0;j<2;j++){
  std::string name="right_hand_thumb_"+std::to_string(j)+"_joint";int a=mj_name2id(m,mjOBJ_ACTUATOR,name.c_str());int q=m->actuator_trnid[2*a];
  m->actuator_ctrlrange[2*a]=enabled?m->jnt_range[2*q]:(j==0?0:-.8);
  m->actuator_ctrlrange[2*a+1]=enabled?m->jnt_range[2*q+1]:0;
 }
}
