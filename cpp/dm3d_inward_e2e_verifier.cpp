// dm3d_inward_e2e_verifier.cpp
// -----------------------------------------------------------------------------
// End-to-end mathematical verifier for DM3D-MMC.
//
// It embeds the original engine and verifies the 3D inward loop:
//
//   packets -> fuse -> diffuse -> encode -> couple6 -> fold^K
//           -> decode -> compare -> reencode(error) -> latent add -> recur
//
// Verification targets:
//   1) T^3 periodic geometry
//   2) discrete Laplacian invariants
//   3) encoder/decoder cycle identity E(D(Z)) = Z
//   4) 6-neighbour coupling conservation
//   5) analytical and empirical contraction of F_R
//   6) fixed-point residual convergence
//   7) feedback descent of reconstruction loss
//   8) deterministic multimodal fusion
//   9) finite end-to-end recursive trace
//  10) symbolic hyperscale cardinality = 10^6,000,000
//
// Compile next to dm3d_multimodal_coding_environment.cpp:
//   g++ -O3 -march=native -std=c++17 -pthread \
//       dm3d_inward_e2e_verifier.cpp -o dm3d_verify
// -----------------------------------------------------------------------------

#define main dm3d_embedded_demo_main
#include "dm3d_multimodal_coding_environment.cpp"
#undef main

#include <array>
#include <cassert>
#include <sstream>

namespace verify3d {

using namespace dm3d;

struct Check {
    std::string name;
    bool pass = false;
    double value = 0.0;
    double limit = 0.0;
    std::string note;
};

static long double sum(const Volume3D& V) {
    long double s = 0.0L;
    for (float v : V.data) s += (long double)v;
    return s;
}

static long double mean(const Volume3D& V) {
    return V.data.empty() ? 0.0L : sum(V) / (long double)V.data.size();
}

static double l2_rms(const Volume3D& V) {
    long double s = 0.0L;
    for (float v : V.data) s += (long double)v * v;
    return std::sqrt((double)(s / std::max<size_t>(1, V.data.size())));
}

[[maybe_unused]] static bool finite_all(const Volume3D& V) {
    for (float v : V.data) if (!std::isfinite(v)) return false;
    return true;
}

static Volume3D random_volume(int N, int C, uint64_t seed, float scale=1.0f) {
    Volume3D V(N,C,0.0f);
    uint64_t s = seed;
    for (float& v : V.data) {
        s = mix64(s);
        const double u = (double)(s >> 11) * (1.0 / 9007199254740992.0); // [0,1)
        v = scale * (float)(2.0*u - 1.0);
    }
    return V;
}

[[maybe_unused]] static double relative_rms(const Volume3D& A, const Volume3D& B) {
    const double den = std::max(1e-30, l2_rms(B));
    return rms(A,B) / den;
}

static void print_check(const Check& c) {
    std::cout << (c.pass ? "[PASS] " : "[FAIL] ") << std::left << std::setw(36)
              << c.name << std::right << " value=" << std::scientific << c.value;
    if (c.limit != 0.0) std::cout << "  limit=" << c.limit;
    if (!c.note.empty()) std::cout << "  " << c.note;
    std::cout << "\n";
}

static std::vector<Check> verify_primitives() {
    std::vector<Check> out;

    // 1. T^3 periodic boundary identity.
    {
        Volume3D V(8,1,0.0f);
        V.at(7,3,4) = 3.25f;
        const double e = std::abs((double)V.at(-1,3,4) - 3.25);
        out.push_back({"T^3 periodic wrap", e == 0.0, e, 0.0,
                       "V(-1,y,z)=V(N-1,y,z)"});
    }

    // 2a. Laplacian annihilates constants.
    {
        Volume3D C(10,2,2.5f);
        double maxAbs = 0.0;
        for (int z=0; z<C.N; ++z)
        for (int y=0; y<C.N; ++y)
        for (int x=0; x<C.N; ++x)
        for (int c=0; c<C.C; ++c)
            maxAbs = std::max(maxAbs, std::abs((double)lap7(C,x,y,z,c)));
        out.push_back({"Laplacian(constant)=0", maxAbs < 1e-12, maxAbs, 1e-12,
                       "discrete differential invariant"});
    }

    // 2b. Integral of toroidal Laplacian is zero.
    {
        Volume3D V = random_volume(12,3,0xA11CE);
        long double s = 0.0L;
        for (int z=0; z<V.N; ++z)
        for (int y=0; y<V.N; ++y)
        for (int x=0; x<V.N; ++x)
        for (int c=0; c<V.C; ++c)
            s += (long double)lap7(V,x,y,z,c);
        const double e = std::abs((double)s) / std::max<size_t>(1,V.data.size());
        out.push_back({"sum(Delta V)=0 on T^3", e < 1e-6, e, 1e-6,
                       "diffusion preserves global mean to FP tolerance"});
    }

    // 3. Encoder is an exact left inverse of the replication decoder.
    //    E(D(Z)) = Z.
    {
        Volume3D Z = random_volume(11,4,0xE0D0);
        Volume3D cyc = encode2(decode2(Z));
        const double e = rms(cyc,Z);
        out.push_back({"latent cycle E(D(Z))=Z", e < 1e-7, e, 1e-7,
                       "exact block-average/replicate pair"});
    }

    // 4. Six-neighbour coupling preserves the global mean on a torus.
    {
        Volume3D Z = random_volume(13,2,0xC06A);
        const long double m0 = mean(Z);
        Volume3D C = couple6(Z,0.12f,1);
        const long double m1 = mean(C);
        const double e = std::abs((double)(m1-m0));
        out.push_back({"COUPLE_6N mean conservation", e < 1e-7, e, 1e-7,
                       "periodic 6-neighbour averaging"});
    }

    // 5. Analytical and empirical contraction of F_R.
    {
        constexpr float lambda = 0.70f;
        constexpr float mu = 0.02f;
        const double q = std::max(std::abs((double)lambda),
                                  std::abs((double)lambda - 12.0*(double)mu));
        Volume3D A = random_volume(16,3,0xAAAA);
        Volume3D B = random_volume(16,3,0xBBBB);
        Volume3D R = random_volume(16,3,0xCCCC);
        Volume3D FA = fold_in(A,R,lambda,mu,1);
        Volume3D FB = fold_in(B,R,lambda,mu,1);
        const double ratio = rms(FA,FB) / std::max(1e-30,(double)rms(A,B));
        out.push_back({"FOLD_IN Lipschitz contraction", ratio <= q + 2e-5,
                       ratio, q, "Fourier spectral bound q=max(|lambda|,|lambda-12mu|)"});
    }

    // 6. Fixed-point residual decreases under repeated inward folding.
    {
        constexpr float lambda = 0.70f;
        constexpr float mu = 0.02f;
        const double q = std::max(std::abs((double)lambda),
                                  std::abs((double)lambda - 12.0*(double)mu));
        Volume3D Z = random_volume(16,2,0xF1ED);
        Volume3D R = couple6(Z,0.12f,1);
        double prev = std::numeric_limits<double>::infinity();
        double worstRatio = 0.0;
        bool monotone = true;
        for (int k=0; k<30; ++k) {
            Volume3D N = fold_in(Z,R,lambda,mu,1);
            const double r = rms(N,Z);
            if (std::isfinite(prev)) {
                worstRatio = std::max(worstRatio, r/std::max(prev,1e-30));
                if (r > prev*(1.0+2e-5)) monotone = false;
            }
            Z = std::move(N);
            prev = r;
        }
        out.push_back({"fixed-point residual contracts", monotone && worstRatio <= q+2e-4,
                       worstRatio, q, "||Z_{k+1}-Z_k|| contracts geometrically"});
    }

    // 7. Residual re-encoding is a descent step for reconstruction MSE.
    //    With P=D E and E D = I, z+ = z + eta E(V-Dz) gives
    //    coarse error multiplier (1-eta).  For 0<eta<2, MSE cannot increase.
    {
        constexpr float eta = 0.25f;
        Volume3D V = random_volume(24,3,0xDA7A);
        Volume3D Z = random_volume(12,3,0x1A7E);
        Volume3D R0 = decode2(Z,1);
        const double L0 = mse(V,R0);
        Volume3D err = subtract(V,R0,1);
        Volume3D dZ = encode2(err,1);
        axpy(Z,dZ,eta,1);
        Volume3D R1 = decode2(Z,1);
        const double L1 = mse(V,R1);
        out.push_back({"REENCODE_ERR + LATADD descent", L1 <= L0 + 1e-12,
                       L1/L0, 1.0, "ratio L_after/L_before"});
    }

    // 8. Deterministic multimodal fusion.
    {
        std::vector<MediaPacket> P;
        P.push_back(make_text_packet("verify 3D recursive state",1.0f));
        P.push_back(make_synthetic_packet(Modality::Image,4096,0x11,1.0f));
        P.push_back(make_synthetic_packet(Modality::Audio,2048,0x22,0.8f));
        P.push_back(make_synthetic_packet(Modality::Video,8192,0x33,1.2f));
        Volume3D A = fuse_packets(P,24,4,1);
        Volume3D B = fuse_packets(P,24,4,1);
        const double e = rms(A,B);
        out.push_back({"multimodal fusion deterministic", e == 0.0, e, 0.0,
                       "same packets -> bit-identical field in serial mode"});
    }

    // 9. Symbolic hyperscale is represented correctly, never enumerated.
    {
        HyperOpSpace H;
        const bool ok = H.symbolic_cardinality() == "10^6000000";
        out.push_back({"virtual 10^(6,000,000) space", ok,
                       (double)H.log10_cardinality(), 6000000.0,
                       "logical cardinality only"});
    }

    return out;
}

struct TraceRow {
    int outer = 0;
    int folds = 0;
    double foldResidual0 = 0.0;
    double foldResidualN = 0.0;
    double reconBeforeFeedback = 0.0;
    double reconAfterFeedback = 0.0;
    double latentCycleError = 0.0;
    double contractionWorst = 0.0;
};

static std::vector<TraceRow> run_recursive_trace(int worldN=32, int channels=4) {
    std::vector<MediaPacket> packets;
    packets.push_back(make_text_packet(
        "WORLD -> ENCODE -> COUPLE -> FOLD -> DECODE -> COMPARE -> REENCODE -> LATADD -> RECUR",1.0f));
    packets.push_back(make_synthetic_packet(Modality::Image, 16*1024, 0x1111, 1.0f));
    packets.push_back(make_synthetic_packet(Modality::Audio,  8*1024, 0x2222, 0.8f));
    packets.push_back(make_synthetic_packet(Modality::Video, 32*1024, 0x3333, 1.2f));
    packets.push_back(make_synthetic_packet(Modality::Sensor, 4*1024, 0x4444, 0.6f));

    constexpr float alpha = 0.015f;
    constexpr float gamma = 0.12f;
    constexpr float lambda = 0.70f;
    constexpr float mu = 0.02f;
    constexpr float eta = 0.25f;
    constexpr float epsZ = 1e-6f;
    constexpr int maxFolds = 32;

    Volume3D world = fuse_packets(packets,worldN,channels,1);
    Volume3D field = diffuse(world,alpha,1);
    Volume3D Z = encode2(field,1);

    std::vector<TraceRow> rows;

    for (int outer=0; outer<8; ++outer) {
        TraceRow tr;
        tr.outer = outer;
        Volume3D ref = couple6(Z,gamma,1);

        double prevResidual = 0.0;
        for (int k=0; k<maxFolds; ++k) {
            Volume3D next = fold_in(Z,ref,lambda,mu,1);
            const double r = rms(next,Z);
            if (k==0) tr.foldResidual0 = r;
            if (k>0 && prevResidual>0.0)
                tr.contractionWorst = std::max(tr.contractionWorst,r/prevResidual);
            Z = std::move(next);
            prevResidual = r;
            tr.folds++;
            if (r < epsZ) break;
        }
        tr.foldResidualN = prevResidual;

        // Exact latent cycle invariant of the chosen codec pair.
        tr.latentCycleError = rms(encode2(decode2(Z,1),1),Z);

        Volume3D recon = decode2(Z,1);
        tr.reconBeforeFeedback = mse(field,recon);

        Volume3D err = subtract(field,recon,1);
        Volume3D dz = encode2(err,1);
        axpy(Z,dz,eta,1);

        Volume3D reconAfter = decode2(Z,1);
        tr.reconAfterFeedback = mse(field,reconAfter);

        rows.push_back(tr);
    }

    return rows;
}

static bool verify_trace(const std::vector<TraceRow>& rows) {
    bool ok = true;
    constexpr double q = 0.70;

    std::cout << "\n3D INWARD RECURSIVE TRACE\n";
    std::cout << "-----------------------------------------------------------------------------------------------------------\n";
    std::cout << "out  folds   r0(fold)      rN(fold)      worst q   cycle E(D)    MSE fold       MSE feedback   verdict\n";
    std::cout << "-----------------------------------------------------------------------------------------------------------\n";

    for (const auto& r : rows) {
        const bool contract = r.foldResidualN <= r.foldResidual0 + 1e-12 &&
                              r.contractionWorst <= q + 5e-3;
        const bool cycle = r.latentCycleError < 1e-7;
        const bool descent = r.reconAfterFeedback <= r.reconBeforeFeedback + 1e-12;
        const bool finite = std::isfinite(r.foldResidual0) && std::isfinite(r.foldResidualN) &&
                            std::isfinite(r.reconBeforeFeedback) && std::isfinite(r.reconAfterFeedback);
        const bool pass = contract && cycle && descent && finite;
        ok = ok && pass;

        std::cout << std::setw(3) << r.outer << "  "
                  << std::setw(5) << r.folds << "  "
                  << std::scientific << std::setprecision(3)
                  << std::setw(12) << r.foldResidual0 << "  "
                  << std::setw(12) << r.foldResidualN << "  "
                  << std::setw(8)  << r.contractionWorst << "  "
                  << std::setw(12) << r.latentCycleError << "  "
                  << std::setw(12) << r.reconBeforeFeedback << "  "
                  << std::setw(12) << r.reconAfterFeedback << "  "
                  << (pass ? "PASS" : "FAIL") << "\n";
    }
    return ok;
}

} // namespace verify3d

int main() {
    using namespace verify3d;

    std::cout << "DM3D-MMC — END-TO-END 3D INWARD VERIFICATION\n";
    std::cout << "============================================================\n";
    std::cout << "Analytical fold spectrum on T^3:\n";
    std::cout << "  Delta eigenvalue xi(k) = 2(cos kx + cos ky + cos kz) - 6 in [-12,0]\n";
    std::cout << "  J_F = lambda I + mu Delta\n";
    std::cout << "  q = max_{xi in [-12,0]} |lambda + mu xi|\n";
    std::cout << "  lambda=0.70, mu=0.02 -> q=max(0.70,0.46)=0.70 < 1\n";
    std::cout << "  therefore fixed-reference F_R is a contraction on finite L2(T^3).\n\n";

    const auto checks = verify_primitives();
    bool primitiveOK = true;
    for (const auto& c : checks) {
        print_check(c);
        primitiveOK = primitiveOK && c.pass;
    }

    const auto trace = run_recursive_trace();
    const bool traceOK = verify_trace(trace);

    std::cout << "\n============================================================\n";
    std::cout << "VERIFICATION SUMMARY\n";
    std::cout << "primitive invariants : " << (primitiveOK ? "PASS" : "FAIL") << "\n";
    std::cout << "recursive trace      : " << (traceOK ? "PASS" : "FAIL") << "\n";
    std::cout << "overall              : " << ((primitiveOK && traceOK) ? "PASS" : "FAIL") << "\n";
    std::cout << "\nInterpretation:\n";
    std::cout << "  * The inner FOLD_IN loop is mathematically contractive for the configured parameters.\n";
    std::cout << "  * E(D(Z))=Z verifies latent cycle consistency for this codec pair.\n";
    std::cout << "  * REENCODE_ERR + LATADD is verified as a local reconstruction-descent step.\n";
    std::cout << "  * The complete outer loop is alternating regularization + data correction;\n";
    std::cout << "    its pure reconstruction loss need not decrease across separate outer cycles\n";
    std::cout << "    unless a unified objective/acceptance rule is added.\n";
    std::cout << "  * 10^6,000,000 remains a symbolic logical operation space, not physical throughput.\n";

    return (primitiveOK && traceOK) ? 0 : 2;
}