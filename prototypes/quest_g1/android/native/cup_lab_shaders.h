#pragma once

// The preview script compiles these exact sources with a desktop GLSL preamble.
namespace cuplab {
inline constexpr const char* VertexShader = R"(#version 320 es
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
uniform mat4 vp;
uniform mat4 model;
uniform mat4 sceneModel;
uniform mat3 normalMatrix;
out vec3 worldNormal;
out vec3 worldPosition;
out vec3 scenePosition;
void main(){
    vec4 p=model*vec4(position,1.);
    gl_Position=vp*p;
    worldPosition=p.xyz;
    worldNormal=normalize(normalMatrix*normal);
    scenePosition=(sceneModel*vec4(position,1.)).xyz;
}
)";
inline constexpr const char* FragmentShader = R"(#version 320 es
precision highp float;
in vec3 worldNormal;
in vec3 worldPosition;
in vec3 scenePosition;
uniform vec4 color;
uniform vec3 eyePosition;
uniform vec3 keyDirection;
uniform vec3 upDirection;
uniform vec3 materialParams; // specular, shininess, emission
uniform vec4 cupShade;       // current origin xyz, placement cue
uniform vec4 cupGoal;        // xyz, acceptance radius
uniform vec2 robotPosition;
uniform int surfaceKind;    // 1 floor, 2 wood, 3 ceramic, 4 cue
uniform int labStyle;
out vec4 fragColor;
void main(){
    vec3 n=normalize(worldNormal);
    if(labStyle==0){
        float light=.4+.6*max(dot(n,normalize(vec3(.5,1,.4))),0.);
        fragColor=vec4(color.rgb*light,color.a);return;
    }
    vec3 base=color.rgb;
    float ao=1.;
    if(surfaceKind==1){
        // Broad concrete variation; no high-frequency checkerboard in VR.
        float grain=sin(scenePosition.x*19.+sin(scenePosition.y*7.))*sin(scenePosition.y*23.);
        base*=.98+.02*grain;
        vec2 footShadow=(scenePosition.xy-robotPosition)/vec2(.20,.29);
        ao=1.-.22*exp(-dot(footShadow,footShadow));
        for(int i=0;i<4;i++){
            vec2 leg=vec2(cupGoal.x+((i<2)?-.26:.26),((i%2==0)?-.39:.39));
            vec2 d=(scenePosition.xy-leg)/vec2(.075,.075);
            ao*=1.-.24*exp(-dot(d,d));
        }
    }
    if(surfaceKind==2){
        float g=scenePosition.y*420.+3.2*sin(scenePosition.x*5.)+sin(scenePosition.x*19.);
        float grain=sin(g)*.013+sin(g*.37)*.018+sin(g*.083)*.013;
        base*=1.+grain;
        if(abs(scenePosition.z-cupGoal.z)<.012){
            vec2 shadow=scenePosition.xy-cupShade.xy;
            float height=max(0.,cupShade.z-cupGoal.z);
            // Analytic soft contact cue; this is not a shadow map.
            ao=1.-.28*exp(-dot(shadow,shadow)/(.0018+height*.018))*exp(-height*5.);
        }
    }
    if(surfaceKind==3){
        float cavity=clamp((scenePosition.z-cupShade.z)/.096,0.,1.);
        ao=.78+.22*cavity;
    }
    if(surfaceKind==4){base=mix(base,vec3(.22,.87,.58),cupShade.w);}
    if(surfaceKind==5){base=mix(vec3(.77,.86,.82),vec3(.43,.66,.78),clamp((scenePosition.z-1.3)/1.1,0.,1.));}
    float hemi=.5+.5*dot(n,normalize(upDirection));
    vec3 ambient=mix(vec3(.40,.42,.41),vec3(.74,.77,.75),hemi);
    float diffuse=max(dot(n,normalize(keyDirection)),0.);
    vec3 v=normalize(eyePosition-worldPosition);
    vec3 h=normalize(normalize(keyDirection)+v);
    float spec=pow(max(dot(n,h),0.),mix(12.,96.,materialParams.y))*materialParams.x*.10;
    vec3 linear=pow(base,vec3(2.2));
    vec3 lit=linear*(ambient+vec3(.38,.35,.29)*diffuse)*ao+vec3(.98,.94,.84)*spec;
    lit=mix(lit,linear,clamp(materialParams.z,0.,1.));
    lit=pow(clamp(lit,0.,1.),vec3(1./2.2));
    fragColor=vec4(lit,color.a);
}
)";
} // namespace cuplab
