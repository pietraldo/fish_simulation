#version 330 core

in float vID;

out vec4 FragColor;

void main()
{
    if(vID == 0.0f){
       FragColor = vec4(0.4f, 0.0f, 0.2f, 1.0f);}
    else if(vID == 1.0f){
       FragColor = vec4(0.0f, 1.0f, 0.0f, 1.0f);}
    else if(vID == 2.0f){
       FragColor = vec4(0.0f, 0.0f, 1.0f, 1.0f);}
}
