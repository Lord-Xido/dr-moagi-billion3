#pragma once
#include <vector>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#include <windows.h>
#include <winioctl.h>
#include <d3d11.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
class GPU {
 ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context;
 ComPtr<ID3D11ComputeShader> shader; ComPtr<IDXGIAdapter3> adapter;
 static void check(HRESULT h) { if(FAILED(h)) throw std::runtime_error("D3D11 operation failed"); }
public:
 bool ready=false; std::string name="CPU";
 GPU(bool enabled) {
  if(!enabled) return;
  try {
   D3D_FEATURE_LEVEL level;
   check(D3D11CreateDevice(nullptr,D3D_DRIVER_TYPE_HARDWARE,nullptr,0,nullptr,0,D3D11_SDK_VERSION,&device,&level,&context));
   if(level<D3D_FEATURE_LEVEL_11_0) throw std::runtime_error("Compute feature level unavailable");
   ComPtr<IDXGIDevice> dx; check(device.As(&dx)); ComPtr<IDXGIAdapter> a; check(dx->GetAdapter(&a)); a.As(&adapter);
   DXGI_ADAPTER_DESC desc{}; check(a->GetDesc(&desc)); name="D3D11 hardware compute";
   const char* source=R"(RWStructuredBuffer<float> z:register(u0); StructuredBuffer<float> target:register(t0);
[numthreads(64,1,1)] void main(uint3 id:SV_DispatchThreadID){if(id.x<4096) z[id.x]=0.5*z[id.x]+0.5*target[id.x];})";
   ComPtr<ID3DBlob> code,error; check(D3DCompile(source,strlen(source),nullptr,nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS,0,&code,&error));
   check(device->CreateComputeShader(code->GetBufferPointer(),code->GetBufferSize(),nullptr,&shader)); ready=true;
  } catch(...) { ready=false; name="CPU (GPU initialization unavailable)"; }
 }
 uint64_t available() const {
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  if(adapter && SUCCEEDED(adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info))) return info.Budget>info.CurrentUsage?info.Budget-info.CurrentUsage:0;
  return 0;
 }
 void refine(std::vector<float>& z,const std::vector<float>& target,unsigned iterations,uint64_t budget) {
  const UINT bytes=static_cast<UINT>(z.size()*sizeof(float));
  if(z.size()!=4096 || target.size()!=4096 || uint64_t(bytes)*3>budget || uint64_t(bytes)*3>available()) throw std::runtime_error("GPU allocation guard");
  D3D11_BUFFER_DESC d{}; d.ByteWidth=bytes; d.Usage=D3D11_USAGE_DEFAULT; d.BindFlags=D3D11_BIND_UNORDERED_ACCESS|D3D11_BIND_SHADER_RESOURCE; d.MiscFlags=D3D11_RESOURCE_MISC_BUFFER_STRUCTURED; d.StructureByteStride=4;
  D3D11_SUBRESOURCE_DATA init{}; init.pSysMem=z.data(); ComPtr<ID3D11Buffer> zb,tb,read;
  check(device->CreateBuffer(&d,&init,&zb)); init.pSysMem=target.data(); check(device->CreateBuffer(&d,&init,&tb));
  D3D11_UNORDERED_ACCESS_VIEW_DESC ud{}; ud.ViewDimension=D3D11_UAV_DIMENSION_BUFFER; ud.Buffer.NumElements=4096;
  ComPtr<ID3D11UnorderedAccessView> u; check(device->CreateUnorderedAccessView(zb.Get(),&ud,&u));
  D3D11_SHADER_RESOURCE_VIEW_DESC sd{}; sd.ViewDimension=D3D11_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements=4096;
  ComPtr<ID3D11ShaderResourceView> s; check(device->CreateShaderResourceView(tb.Get(),&sd,&s));
  context->CSSetShader(shader.Get(),nullptr,0); auto up=u.Get(); auto sp=s.Get(); context->CSSetUnorderedAccessViews(0,1,&up,nullptr); context->CSSetShaderResources(0,1,&sp);
  for(unsigned k=0;k<iterations;++k) context->Dispatch(64,1,1);
  ID3D11UnorderedAccessView* nullU=nullptr; ID3D11ShaderResourceView* nullS=nullptr;
  context->CSSetUnorderedAccessViews(0,1,&nullU,nullptr); context->CSSetShaderResources(0,1,&nullS);
  d.Usage=D3D11_USAGE_STAGING; d.BindFlags=0; d.CPUAccessFlags=D3D11_CPU_ACCESS_READ; d.MiscFlags=0; d.StructureByteStride=0;
  check(device->CreateBuffer(&d,nullptr,&read)); context->CopyResource(read.Get(),zb.Get());
  D3D11_MAPPED_SUBRESOURCE mapped{}; check(context->Map(read.Get(),0,D3D11_MAP_READ,0,&mapped)); memcpy(z.data(),mapped.pData,bytes); context->Unmap(read.Get(),0);
 }
};
#else
class GPU { public: bool ready=false; std::string name="CPU"; GPU(bool){} uint64_t available() const{return 0;} void refine(std::vector<float>&,const std::vector<float>&,unsigned,uint64_t){throw std::runtime_error("GPU unavailable");} };
#endif

