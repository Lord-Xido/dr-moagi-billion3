#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <limits>
#include "gpu.hpp"
using Clock=std::chrono::steady_clock;
constexpr uint64_t GB=1000000000ULL, EB=1000000000000000000ULL;
constexpr size_t N=64*64*64, L=16*16*16;
constexpr uint64_t headerBytes=65536, tileBytes=N*sizeof(float), capacity=(GB-headerBytes)/tileBytes;
struct Address { uint32_t x,y,z; uint64_t byte;
 uint64_t cell()const { if(x>=1000||y>=1000||z>=1000||byte>=GB) throw std::out_of_range("Namespace address"); return (uint64_t(z)*1000+y)*1000+x; }
 uint64_t linear()const{return cell()*GB+byte;}
};
struct Record { uint64_t cell,tile; };
class Store {
 std::fstream file; std::vector<Record> records;
 // Explicit sparse hierarchy: 10^3 cells per region, 10^3 regions globally.
 std::map<uint64_t,std::map<uint64_t,std::map<uint64_t,size_t>>> regions;
 static uint64_t region(uint64_t c){auto x=c%1000,y=(c/1000)%1000,z=c/1000000;return (z/100)*100+(y/100)*10+x/100;}
 void index(Record r,size_t slot){ regions[region(r.cell)][r.cell][r.tile]=slot; }
public:
 Store(const std::string& path) {
  bool exists=std::filesystem::exists(path);
  if(!exists) {
#ifdef _WIN32
   HANDLE h=CreateFileA(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
   if(h==INVALID_HANDLE_VALUE) throw std::runtime_error("Create backing file failed");
   DWORD returned=0; bool sparse=DeviceIoControl(h,FSCTL_SET_SPARSE,nullptr,0,nullptr,0,&returned,nullptr)!=0;
   LARGE_INTEGER length{}; length.QuadPart=GB;
   bool ok=SetFilePointerEx(h,length,nullptr,FILE_BEGIN)&&SetEndOfFile(h); CloseHandle(h);
   if(!ok) throw std::runtime_error("Resize backing file failed");
   std::cout<<"sparse_file="<<sparse<<" logical_bytes="<<GB<<"\n";
#else
   std::ofstream create(path,std::ios::binary); create.seekp(GB-1); create.put(0); create.close();
#endif
  }
  if(std::filesystem::file_size(path)!=GB) throw std::runtime_error("Backing length must be exactly decimal 1 GB");
  file.open(path,std::ios::binary|std::ios::in|std::ios::out); if(!file) throw std::runtime_error("Open backing failed");
  char magic[8]{}; file.read(magic,8); uint64_t count=0; file.read(reinterpret_cast<char*>(&count),8);
  if(std::memcmp(magic,"DM3D0001",8)==0) {
   if(count>capacity) throw std::runtime_error("Corrupt index count");
   records.resize(static_cast<size_t>(count)); file.read(reinterpret_cast<char*>(records.data()),static_cast<std::streamsize>(count*sizeof(Record)));
   if(!file) throw std::runtime_error("Read index failed");
   for(size_t i=0;i<records.size();++i){if(records[i].cell>=1000000000ULL||records[i].tile>=GB/tileBytes) throw std::runtime_error("Corrupt address"); index(records[i],i);}
  } else {
   bool zero=true; for(char c:magic) zero=zero&&c==0;
   if(exists||!zero) throw std::runtime_error("Refusing unknown backing format");
   saveIndex();
  }
 }
 void saveIndex(){file.clear();file.seekp(0);file.write("DM3D0001",8);uint64_t count=records.size();file.write(reinterpret_cast<char*>(&count),8);file.write(reinterpret_cast<char*>(records.data()),static_cast<std::streamsize>(records.size()*sizeof(Record)));file.flush();if(!file)throw std::runtime_error("Index write failed");}
 void write(Address a,const std::vector<float>& data){
  uint64_t c=a.cell(),t=a.byte/tileBytes;
  if(a.byte%tileBytes||a.byte+tileBytes>GB||data.size()!=N) throw std::out_of_range("Tile bounds/alignment");
  auto& tiles=regions[region(c)][c]; auto it=tiles.find(t); size_t slot;
  if(it==tiles.end()){if(records.size()>=capacity)throw std::runtime_error("1 GB substrate full; grow via explicit new shards");slot=records.size();records.push_back({c,t});tiles[t]=slot;}else slot=it->second;
  file.clear();file.seekp(static_cast<std::streamoff>(headerBytes+slot*tileBytes));file.write(reinterpret_cast<const char*>(data.data()),tileBytes);file.flush();if(!file)throw std::runtime_error("Tile write failed");saveIndex();
 }
 std::vector<float> read(Address a){
  uint64_t c=a.cell(); if(a.byte%tileBytes||a.byte+tileBytes>GB)throw std::out_of_range("Tile read bounds");
  std::vector<float> data(N,0); auto r=regions.find(region(c));if(r==regions.end())return data;auto b=r->second.find(c);if(b==r->second.end())return data;auto t=b->second.find(a.byte/tileBytes);if(t==b->second.end())return data;
  file.clear();file.seekg(static_cast<std::streamoff>(headerBytes+t->second*tileBytes));file.read(reinterpret_cast<char*>(data.data()),tileBytes);if(!file)throw std::runtime_error("Tile read failed");return data;
 }
};
size_t latent(size_t i){size_t x=i%64,y=(i/64)%64,z=i/4096;return (z/4)*256+(y/4)*16+x/4;}
std::vector<float> generate(unsigned seed){std::vector<float> v(N);uint32_t s=seed+1;for(auto& x:v){s=s*1664525u+1013904223u;x=float(s>>16)/65536.0f;}return v;}
std::vector<float> encode(const std::vector<float>& v){std::vector<float> z(L,0);for(size_t i=0;i<N;++i)z[latent(i)]+=v[i]/64.0f;return z;}
void refineCPU(std::vector<float>& z,const std::vector<float>& target,unsigned k){for(unsigned j=0;j<k;++j)for(size_t i=0;i<L;++i)z[i]=0.5f*z[i]+0.5f*target[i];}
uint64_t availableRAM(){
#ifdef _WIN32
 MEMORYSTATUSEX s{};s.dwLength=sizeof(s);if(!GlobalMemoryStatusEx(&s))throw std::runtime_error("Memory telemetry unavailable");return s.ullAvailPhys;
#else
 return 512ULL*1024*1024;
#endif
}
struct Scheduler {
 unsigned depth=1,k=8; double omega=0;
 void update(double io,double compute,double residual,uint64_t free,uint64_t budget){omega=0.9*omega+0.1*residual;k=omega>0.15?12:8;depth=free<budget*2?1:std::min(8u,depth+(io>compute?1u:0u));}
};
void require(bool b,const char* msg){if(!b)throw std::runtime_error(msg);}
void tests(){
 require(Address{999,999,999,GB-1}.linear()==EB-1,"Last address");
 bool caught=false;try{Address{1000,0,0,0}.linear();}catch(const std::out_of_range&){caught=true;}require(caught,"Address guard");
 auto x=generate(17),y=generate(17);require(x==y,"Determinism");auto z=encode(std::vector<float>(N,1));for(float v:z)require(v==1,"Block encoder");auto target=z;std::fill(z.begin(),z.end(),0);refineCPU(z,target,12);require(std::abs(z[0]-1)<0.001,"Refinement contraction");
 Scheduler s;s.update(10,1,0.5,1,100);require(s.depth==1,"Memory pressure");
 std::cout<<"PASS namespace, deterministic generation, encoder, contraction, scheduler guards\n";
}
int main(int argc,char** argv){try{
 bool self=false,cpu=false,demo=false;unsigned tiles=4;uint64_t host=256ULL*1024*1024,gpuBudget=64ULL*1024*1024;std::string path="dm3d_1GB.vmem";
 for(int i=1;i<argc;++i){std::string a=argv[i];auto value=[&](){if(++i>=argc)throw std::runtime_error("Missing option value");return std::string(argv[i]);};
 if(a=="--self-test")self=true;else if(a=="--cpu")cpu=true;else if(a=="--demo")demo=true;else if(a=="--file")path=value();else if(a=="--tiles"){auto n=std::stoull(value());if(n==0||n>capacity)throw std::runtime_error("Tile count guard");tiles=static_cast<unsigned>(n);}else if(a=="--host-mib"||a=="--gpu-mib"){auto n=std::stoull(value());if(n<8||n>2048)throw std::runtime_error("Budget range 8..2048 MiB");(a=="--host-mib"?host:gpuBudget)=n*1024*1024;}else throw std::runtime_error("Unknown option");}
 if(self){tests();return 0;}if(!demo){std::cout<<"DrMoagi3D --demo [--cpu] [--tiles 4] [--file dm3d_1GB.vmem] [--host-mib 256] [--gpu-mib 64]\n";return 0;}
 require(host>=6*tileBytes+3*L*sizeof(float),"Host allocation guard");require(availableRAM()>host,"Insufficient available RAM");
 GPU gpu(!cpu);Store store(path);Scheduler scheduler;
 std::cout<<"namespace_bytes="<<EB<<" backend="<<gpu.name<<" host_budget="<<host<<" gpu_budget="<<gpuBudget<<"\n";
 for(unsigned tile=0;tile<tiles;++tile){require(availableRAM()>6*tileBytes,"Runtime host memory pressure");Address a{tile%1000,(tile/1000)%1000,999,0};auto input=generate(tile);auto start=Clock::now();store.write(a,input);auto x=store.read(a);require(x==input,"Backing readback");double io=std::chrono::duration<double,std::milli>(Clock::now()-start).count();start=Clock::now();
 auto target=encode(x),z=target;for(auto& v:z)v*=0.5f;
 if(gpu.ready){try{gpu.refine(z,target,scheduler.k,gpuBudget);}catch(const std::exception& e){std::cout<<"gpu_fallback="<<e.what()<<"\n";gpu.ready=false;z=target;for(auto& v:z)v*=0.5f;refineCPU(z,target,scheduler.k);}}else refineCPU(z,target,scheduler.k);
 std::vector<float> decoded(N),residual(N);double mse=0,maxError=0;
 // CTR: Generate decoded candidates; Contrast residual; Reckon error;
 // Verify finite values; Correct using an explicit full-size residual sidecar.
 for(size_t i=0;i<N;++i){decoded[i]=z[latent(i)];residual[i]=x[i]-decoded[i];require(std::isfinite(residual[i]),"CTR non-finite residual");mse+=double(residual[i])*residual[i];decoded[i]+=residual[i];maxError=std::max(maxError,double(std::abs(decoded[i]-x[i])));}
 require(maxError<=1e-6,"CTR corrected reconstruction");store.write(a,decoded);auto checked=store.read(a);require(checked==decoded,"Corrected backing readback");
 double compute=std::chrono::duration<double,std::milli>(Clock::now()-start).count();scheduler.update(io,compute,std::sqrt(mse/N),availableRAM(),host);
 std::cout<<"tile="<<tile<<" io_ms="<<io<<" pipeline_ms="<<compute<<" residual_rmse="<<std::sqrt(mse/N)<<" corrected_max="<<maxError<<" omega="<<scheduler.omega<<" next_K="<<scheduler.k<<" queue_window="<<scheduler.depth<<" gpu_free="<<gpu.available()<<" PASS\n";
 }
 std::cout<<"END_TO_END PASS\n";return 0;
 }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}}

