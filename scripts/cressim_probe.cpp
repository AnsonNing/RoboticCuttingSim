// Independent headless API harness for yafei-ou/CRESSim-MPM (BSD-3-Clause).
// The upstream example_cutting is the API reference; see THIRD_PARTY_NOTICES.md.
#include "simulation_factory.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

using namespace crmpm;

// input: xyzm rows, trajectory: t xyz qxyzw; output: xyz velocity_xyz mass float32.
int main(int argc, char** argv) {
    if (argc != 10) { std::cerr << "particles trajectory output frames frame_dt subdt dx mode E\n"; return 2; }
    const int frames=std::stoi(argv[4]); const float dt=std::stof(argv[5]);
    const float subdt=std::stof(argv[6]), dx=std::stof(argv[7]);
    const std::string mode=argv[8]; const float E=std::stof(argv[9]), nu=.30f;
    std::ifstream in(argv[1]); std::vector<std::array<float,4>> particles;
    float x,y,z,m;
    while(in >> x >> y >> z >> m) particles.push_back({x,y,z,m});
    if(particles.empty()) return 3;
    std::ifstream traj(argv[2]); std::string line; std::getline(traj,line);
    std::vector<std::array<float,8>> trajectory;
    while(std::getline(traj,line)) {
        std::replace(line.begin(),line.end(),',',' '); std::istringstream row(line);
        std::array<float,8> a{}; for(float& v:a) row>>v; trajectory.push_back(a);
    }
    if(trajectory.size()<2) return 4;
    const auto poseAt=[&](float t) {
        size_t j=0; while(j+2<trajectory.size() && t>=trajectory[j+1][0]) ++j;
        const auto& a=trajectory[j]; const auto& b=trajectory[j+1];
        float u=std::clamp((t-a[0])/(b[0]-a[0]),0.f,1.f);
        Transform out{}; out.position=Vec3f(a[1]+u*(b[1]-a[1]),a[2]+u*(b[2]-a[2]),a[3]+u*(b[3]-a[3]));
        // Shared reference trajectories have fixed identity orientation. Reject rotation.
        if(a[4]!=0 || a[5]!=0 || a[6]!=0 || a[7]!=1 || b[4]!=0 || b[5]!=0 || b[6]!=0 || b[7]!=1)
            throw std::runtime_error("Headless CRESSim reference adapter supports fixed identity input orientation only");
        if(mode=="slicer") { // local XZ slicer plane -> world XY; lower blade edge z.
            out.rotation=Quat(.70710678f,0,0,.70710678f);
            out.position-=Vec3f(0,0,.010f);
        }
        return out;
    };
    auto* factory=createFactory(particles.size()+16,16,16,1024,true);
    if(!factory || !factory->isGpu()) return 5;
    SceneDesc sd{}; sd.solverType=MpmSolverType::eGpuMlsMpmSolver;
    sd.numMaxParticles=particles.size()+16; sd.gravity=Vec3f(0,0,-9.81f);
    // C++ Bounds3 takes minimum/maximum, unlike the C API's center/extents struct.
    sd.gridBounds=Bounds3(Vec3f(-.10f,-.08f,-.05f),Vec3f(.10f,.08f,.10f));
    for(const auto& p:particles) {
        if(!sd.gridBounds.contains(Vec3f(p[0],p[1],p[2]))) {
            std::cerr << "Initial particle outside CRESSim grid bounds\n"; return 8;
        }
    }
    sd.gridCellSize=dx; sd.solverIntegrationStepSize=subdt;
    auto* scene=factory->createScene(sd); if(!scene) return 6;
    unsigned offset=0; if(!scene->allocateParticles(particles.size(),offset)) return 7;
    auto& data=scene->getParticleData(); auto& mat=scene->getParticleMaterialData();
    const float lambda=E*nu/((1+nu)*(1-2*nu)), mu=E/(2*(1+nu));
    for(unsigned i=0;i<particles.size();++i) {
        auto a=particles[i]; unsigned j=offset+i;
        data.positionMass[j]=make_float4(a[0],a[1],a[2],a[3]); data.velocity[j]=Vec3f(0);
        mat.type[j]=ParticleMaterialType::eCoRotational; mat.params0[j]=make_float4(lambda,mu,0,0);
        scene->getComputeInitialDataIndices().pushBack(j);
    }
    scene->setActiveMaskRange(offset,particles.size(),1); scene->markDirty(SceneDataDirtyFlags::eAll);
    std::vector<Shape*> shapes;
    auto addBox=[&](Vec3f c,Vec3f h) {
        auto* g=factory->createGeometry(GeometryType::eBox,make_float4(h.x(),h.y(),h.z(),0));
        ShapeDesc d{}; d.type=ShapeType::eKinematic; d.transform.position=c;
        d.sdfSmoothDistance=0; d.sdfFatten=0; d.drag=1; d.friction=.4f;
        auto* s=factory->createShape(*g,d); g->release(); scene->addShape(s); shapes.push_back(s);
    };
    addBox(Vec3f(0,0,-.005f),Vec3f(.068f,.048f,.005f));
    addBox(Vec3f(-.064f,0,.0075f),Vec3f(.004f,.044f,.0075f));
    addBox(Vec3f(.064f,0,.0075f),Vec3f(.004f,.044f,.0075f));
    addBox(Vec3f(0,-.044f,.0075f),Vec3f(.060f,.004f,.0075f));
    addBox(Vec3f(0,.044f,.0075f),Vec3f(.060f,.004f,.0075f));
    auto* geometry=factory->createGeometry(mode=="slicer"?GeometryType::eQuadSlicer:GeometryType::eBox,
        mode=="slicer"?make_float4(.010f,.025f,0,0):make_float4(.001f,.025f,.010f,0));
    ShapeDesc bd{}; bd.type=ShapeType::eKinematic; bd.transform=poseAt(0);
    bd.sdfSmoothDistance=0; bd.sdfFatten=mode=="slicer"?.0005f:0; bd.drag=.5f; bd.friction=.4f;
    auto* blade=factory->createShape(*geometry,bd); geometry->release(); scene->addShape(blade);
    std::ofstream out(argv[3],std::ios::binary);
    const auto dump=[&]() {
        const auto& d=scene->getParticleData();
        for(unsigned i=0;i<particles.size();++i) {
            const auto p=d.positionMass[offset+i]; const auto v=d.velocity[offset+i];
            float row[]={p.x,p.y,p.z,v.x(),v.y(),v.z(),p.w};
            for(float value:row) if(!std::isfinite(value)) throw std::runtime_error("Nonfinite CRESSim state");
            out.write(reinterpret_cast<const char*>(row),sizeof(row));
        }
    };
    dump();
    for(int frame=0;frame<frames;++frame) {
        blade->setKinematicTarget(poseAt((frame+1)*dt),dt);
        scene->advance(dt); scene->fetchResults(); dump();
        if((frame+1)%15==0) std::cout<<"frame="<<frame+1<<"/"<<frames<<std::endl;
    }
    blade->release(); for(auto* s:shapes) s->release(); scene->release(); releaseFactory(factory);
    return 0;
}
