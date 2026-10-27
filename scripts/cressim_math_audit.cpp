// Host-only diagnostic using actual CRESSim public/internal math functions.
// Expected failures are kept as evidence, not patched or converted into passes.
#include "constitutive_models.h"
#include <algorithm>
#include <cmath>
#include <iostream>
using namespace crmpm;
int main() {
    Vec3f a(1,2,3), b(4,5,6);
    auto outer=a.outerProduct(b);
    float outer_error=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        outer_error=std::max(outer_error,std::abs(outer(i,j)-a[i]*b[j]));
    auto M=Mat3x3f::Identity(); M(0,1)=.01;
    Mat3x3f u,vt,sigma;
    svd3(M,u,vt,sigma); auto reconstructed=u*sigma*vt;
    float svd_error=0;
    for(int i=0;i<3;++i) for(int j=0;j<3;++j)
        svd_error=std::max(svd_error,std::abs(reconstructed(i,j)-M(i,j)));
    std::cout << "{\"outer_product_max_error\":" << outer_error
              << ",\"svd_reconstruction_max_error\":" << svd_error
              << ",\"status\":\"" << ((outer_error<1e-6 && svd_error<1e-5)?"PASS":"FAIL") << "\"}\n";
    return (outer_error<1e-6 && svd_error<1e-5)?0:1;
}
