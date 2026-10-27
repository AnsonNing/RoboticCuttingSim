// Host diagnostics call the pinned CRESSim headers directly. No solver reimplementation.
// Build via scripts/verify_cressim_candidate.py; upstream notices remain in its source copy.
#include "mls_mpm_algorithm.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace crmpm;

namespace {
float matrixError(const Mat3x3f& a, const Mat3x3f& b) {
    float e = 0;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j)
        e=std::max(e,std::abs(a(i,j)-b(i,j)));
    return e;
}
float vectorError(const Vec3f& a, const Vec3f& b) {
    return std::max({std::abs(a.x()-b.x()),std::abs(a.y()-b.y()),std::abs(a.z()-b.z())});
}
bool allPass=true;
void check(const std::string& name, float error, float tolerance) {
    const bool pass=std::isfinite(error) && error<=tolerance;
    allPass &= pass;
    std::cout << "{\"test\":\"" << name << "\",\"error\":" << error
              << ",\"tolerance\":" << tolerance << ",\"status\":\""
              << (pass?"PASS":"FAIL") << "\"}\n";
}
struct Contact {
    ShapeType type=ShapeType::eKinematic;
    GeometryType geomType=GeometryType::ePlane;
    int id=0;
    Vec3f position{}, invScale{1}, linearVelocity{}, angularVelocity{};
    Quat rotation{};
    float4 params=make_float4(0,1,.25,0);
    float4 geomParams=make_float4(.01,.01,0,0);
    ShapeData shapes{};
    GeometryData geometry{};
    GeometrySdfData sdf{};
    Contact() {
        shapes.type=&type; shapes.position=&position; shapes.rotation=&rotation;
        shapes.invScale=&invScale; shapes.linearVelocity=&linearVelocity;
        shapes.angularVelocity=&angularVelocity; shapes.params0=&params;
        shapes.geometryIdx=&id; shapes.size=1;
        geometry.type=&geomType; geometry.params0=&geomParams; geometry.size=1;
    }
};
constexpr int n=24;
constexpr float dx=.004f;
const Vec3i dims{n,n,n};
const Vec3f lo{-.048f,-.048f,-.048f};
const Vec3f hi{.048f,.048f,.048f};
const ShapeData emptyShapes{};
const GeometryData emptyGeom{};
const GeometrySdfData emptySdf{};
}

int main() {
    std::cout << std::setprecision(10);
    Vec3f a(1,2,3),b(4,5,6);
    Mat3x3f expected;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) expected(i,j)=a[i]*b[j];
    check("outer_product",matrixError(a.outerProduct(b),expected),1e-6);

    auto M=Mat3x3f::Identity(); M(0,1)=.2f; M(2,0)=.1f;
    check("inverse_identity",matrixError(M*M.inverse(),Mat3x3f::Identity()),1e-6);
    Mat3x3f u,vt,sigma;
    svd3(M,u,vt,sigma);
    check("svd_reconstruction",matrixError(u*sigma*vt,M),2e-5);
    check("svd_orthogonality",std::max(matrixError(u.transpose()*u,Mat3x3f::Identity()),
          matrixError(vt*vt.transpose(),Mat3x3f::Identity())),2e-5);

    const float angle=.37f;
    auto R=Mat3x3f::Identity(); R(0,0)=std::cos(angle); R(0,1)=-std::sin(angle);
    R(1,0)=std::sin(angle); R(1,1)=std::cos(angle);
    Mat3x3f stress;
    computeCoRotationalStress(R,30000,20000,stress);
    check("rigid_rotation_zero_corotational_stress_Pa",matrixError(stress,Mat3x3f()),.1);
    auto F=Mat3x3f::Identity(); const float shear=.1f; F(0,1)=shear;
    auto polar=Mat3x3f::Identity(); const float den=std::sqrt(4+shear*shear);
    polar(0,0)=polar(1,1)=2/den; polar(0,1)=shear/den; polar(1,0)=-shear/den;
    computeCoRotationalStress(F,30000,20000,stress);
    const auto analyticalStress=40000*(F-polar)*F.transpose();
    check("simple_shear_corotational_stress_Pa",matrixError(stress,analyticalStress),.1);

    // Exact affine grid field must reproduce its velocity gradient, not its transpose.
    std::vector<float4> grid(n*n*n,make_float4(0,0,0,0));
    auto p=make_float4(.0012f,.0007f,.0031f,12e-6f);
    const Vec3f velocity0{.3f,-.2f,.1f};
    Mat3x3f A; A(0,1)=1.2f; A(1,0)=-.7f; A(2,0)=.4f; A(0,2)=-.3f;
    for(int k=0;k<n;++k) for(int j=0;j<n;++j) for(int i=0;i<n;++i) {
        const Vec3f node=lo+Vec3f(i*dx,j*dx,k*dx);
        const Vec3f v=velocity0+A*(node-Vec3f(p));
        grid[i+j*n+k*n*n]=make_float4(v.x(),v.y(),v.z(),1);
    }
    auto deformation=Mat3x3f::Identity(); Mat3x3f affine;
    auto v=velocity0;
    mlsMpmGridToParticle<false>(lo,hi,1/dx,dx,dims,0,grid.data(),0,nullptr,
        emptyShapes,emptyGeom,emptySdf,p,v,deformation,affine);
    check("affine_grid_to_particle_velocity_m_s",vectorError(v,velocity0),2e-6);
    check("affine_grid_to_particle_gradient_s_inv",matrixError(affine,A),5e-5);

    // A particle with zero elastic constants under gravity: analytical symplectic freefall.
    p=make_float4(.0012f,.0007f,.02f,12e-6f); const Vec3f initial{p};
    v=Vec3f(); affine=Mat3x3f(); deformation=Mat3x3f::Identity();
    const float dt=1e-4f; const int steps=100; const Vec3f gravity{0,0,-9.81f};
    float massError=0;
    for(int s=0;s<steps;++s) {
        std::fill(grid.begin(),grid.end(),make_float4(0,0,0,0));
        mlsMpmParticleToGrid<false>(dt,gravity,lo,1/dx,dx,dims,p,8e-9f,deformation,
            affine,make_float4(0,0,0,0),ParticleMaterialType::eCoRotational,
            0,nullptr,emptyShapes,emptyGeom,emptySdf,v,grid.data());
        double mass=0; for(auto& node:grid) mass+=node.w;
        massError=std::max(massError,static_cast<float>(std::abs(mass-p.w)/p.w));
        for(int idx=0;idx<n*n*n;++idx)
            mlsMpmUpdateGrid<false,false>(dt,idx,lo,dx,dims,0,nullptr,
                emptyShapes,emptyGeom,emptySdf,grid.data());
        mlsMpmGridToParticle<false>(lo,hi,1/dx,dx,dims,dt,grid.data(),0,nullptr,
            emptyShapes,emptyGeom,emptySdf,p,v,deformation,affine);
    }
    check("freefall_velocity_m_s",vectorError(v,gravity*(steps*dt)),2e-5);
    check("freefall_position_m",vectorError(Vec3f(p),initial+gravity*(dt*dt*steps*(steps+1)/2)),2e-7);
    check("p2g_relative_mass_error",massError,2e-6);

    Contact c;
    auto velocity=make_float4(1,-2,0,12e-6f);
    auto point=make_float4(0,-.001f,0,0);
    resolveRigidCollision<false,false,false>(velocity,point,1,&c.id,c.shapes,c.geometry,c.sdf);
    check("static_plane_coulomb_velocity_m_s",vectorError(Vec3f(velocity),Vec3f(.5,0,0)),2e-6);
    velocity=make_float4(0,-2,0,12e-6f);
    resolveRigidCollision<true,false,false>(velocity,point,1,&c.id,c.shapes,c.geometry,c.sdf);
    check("particle_projection_to_plane_m",std::abs(point.y),2e-7);
    // Particle collision only projects position; its velocity is resolved at grid level.

    c.linearVelocity=Vec3f(0,1,0); velocity=make_float4(0,-2,0,12e-6f);
    point=make_float4(0,-.001f,0,0);
    resolveRigidCollision<false,false,false>(velocity,point,1,&c.id,c.shapes,c.geometry,c.sdf);
    check("moving_plane_relative_contact_m_s",vectorError(Vec3f(velocity),Vec3f(0,1,0)),2e-6);

    // This is an executable test of memoryless current-slicer semantics, not cutting success.
    c.geomType=GeometryType::eQuadSlicer;
    const auto below=make_float4(0,-.001f,0,0),above=make_float4(0,.001f,0,0);
    const bool present=checkOnDifferentSides<false>(below,above,1,&c.id,c.shapes,c.geometry,c.sdf);
    c.position=Vec3f(.1f,0,0);
    const bool withdrawn=checkOnDifferentSides<false>(below,above,1,&c.id,c.shapes,c.geometry,c.sdf);
    check("current_slicer_crossing_detected",present?0:1,0);
    check("withdrawn_slicer_restores_transfer",withdrawn?1:0,0);
    std::cout << "{\"test\":\"persistent_cutting\",\"status\":\"UNSUPPORTED\","
                 "\"crossing_before_withdrawal\":" << (present?"true":"false")
              << ",\"crossing_after_withdrawal\":" << (withdrawn?"true":"false") << "}\n";
    std::cout << "{\"scope\":\"host_header_math_and_transfer_controls\",\"status\":\""
              << (allPass?"PASS":"FAIL") << "\"}\n";
    return allPass?0:1;
}
