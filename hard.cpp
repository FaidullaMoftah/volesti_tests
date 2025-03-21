#include "eigen-3.4.0/Eigen/Dense"
#include "utility"
#include <iostream>
#include <string>
#include <fstream>   
#include <cmath>

#define debug(x) ;
#ifdef DBG
#define debug(x) std::cerr << #x << " = " << x << std::endl
#endif

typedef double unit;
typedef Eigen::Matrix<unit, Eigen::Dynamic, Eigen::Dynamic> MT;
typedef Eigen::Matrix<unit, Eigen::Dynamic, 1> VT;

/*
Interior point solver, expects the problem to come as
minimize    c'x
st          Ax = b, x > 0
(Add slack for inequalities, and multiply by -1 if the program wants to maximize).
It also assumes A has full row rank.

Can get arbitrarily close to the optimizer for most problems; keep in mind that
exact solutions typically require more care.
*/
class IP {
public:
    MT A;
    Eigen::ColPivHouseholderQR<MT> gram;
    int n, m, iter;
    VT x, y, s, dxa, dya, dsa, dxc, dyc, dsc, b, c, rx, rs;
    double sigma, mu, apa, ada, zero_eps = 1e-15, ap, ad, eta = 0.99, mu_tol;

    IP(MT A_in, VT b_in, VT c_in, int iter_in, double mu)
        : A(A_in), b(b_in), c(c_in), iter(iter_in), mu_tol(mu),
          m(A_in.rows()), n(A_in.cols()) {}

    void init() {
        auto inv = (A * A.transpose()).ldlt();
        VT y_ = inv.solve(A * c);
        VT s_ = c - A.transpose() * y_;
        VT x_ = A.transpose() * inv.solve(b);
        y = y_;

        double delta_x = std::max(-1.5 * x_.minCoeff(), 0.0);
        double delta_s = std::max(-1.5 * s_.minCoeff(), 0.0);

        double a0 = ((x_ + delta_x * VT::Ones(n))
                    .transpose() * (s_ + delta_s * VT::Ones(n)))(0);
        double d_x = delta_x + 0.5 * a0 / (s_.sum() + n * delta_s);
        double d_s = delta_s + 0.5 * a0 / (x_.sum() + n * delta_x);

        x = x_ + d_x * VT::Ones(n);
        s = s_ + d_s * VT::Ones(n);
    }

    void get_errors() {
        rx = A * x - b;
        rs = A.transpose() * y + s - c;
    }

    void predict() {
        MT N = MT::Zero(2 * n + m, 2 * n + m);
        N.block(n, 0, m, n) = A;
        N.block(n + m, 0, n, n) = s.asDiagonal();
        N.block(0, n, n, m) = A.transpose();
        N.block(0, n + m, n, n) = VT::Ones(n).asDiagonal();
        N.block(n + m, n + m, n, n) = x.asDiagonal();

        gram = N.colPivHouseholderQr();

        VT error(n + n + m);
        error.segment(0, n)   = -rs;
        error.segment(n, m)   = -rx;
        error.segment(n + m, n) = -x.cwiseProduct(s);
        VT ans = gram.solve(error);
        dxa = ans.segment(0, n);
        dya = ans.segment(n, m);
        dsa = ans.segment(n + m, n);
    }

    void set_scalars() {
        double mdx = 1, mds = 1;
        for (int i = 0; i < n; i++) {
            if (dxa(i) < -zero_eps) {
                mdx = std::min(mdx, -x(i) / dxa(i));
            }
            if (dsa(i) < -zero_eps) {
                mds = std::min(mds, -s(i) / dsa(i));
            }
        }
        apa = mdx;
        ada = mds;

        double mu_aff = (x + apa * dxa).dot(s + ada * dsa) / n;
        mu = x.dot(s) / n; 
        sigma = std::max(0.0, std::max(1.0, std::pow(mu_aff / (x.dot(s)/n), 3.0)));
    }

    void correct() {
        VT error(n + n + m);
        error.segment(0, n) = -rs;
        error.segment(n, m) = -rx;
        error.segment(n + m, n) = -x.cwiseProduct(s) - x.dot(s)*VT::Ones(n) + sigma * (x.dot(s)) * VT::Ones(n);

        VT ans = gram.solve(error);
        dxc = ans.segment(0, n);
        dyc = ans.segment(n, m);
        dsc = ans.segment(n + m, n);
    }

    void alphas() {
        double ap_max = 1, ad_max = 1;
        for (int i = 0; i < n; i++) {
            if (dxc(i) < -zero_eps) {
                ap_max = std::min(ap_max, -x(i) / dxc(i));
            }
            if (dsc(i) < -zero_eps) {
                ad_max = std::min(ad_max, -s(i) / dsc(i));
            }
        }

        ap = std::max(zero_eps, 0.8 * ap_max);
        ad = std::max(zero_eps, 0.8 * ad_max);
    }

    void next() {
        x = x + ap * dxc;
        y = y + ad * dyc;
        s = s + ad * dsc;
    }

    std::pair<VT, double> solve() {
        init();
        for (int i = 0; i < iter; i++) {
            get_errors();
            predict();
            set_scalars();
            correct();
            alphas();
            next();

            double mu_val = x.dot(s) / n;
            if (mu_val < mu_tol) {
                break;
            }
        }
        return std::make_pair(x, c.dot(x));
    }

};
int main() {
    MT A(3, 5);
    A << 4.0,  7.0, 1, 0, 0,
         0.3,  1.0, 0, 1, 0,
         1.0, -1.0, 0, 0, 1;
    
    VT b(3);
    b << 100.0,
         8.0,
         9.0;
    
    VT c(5);
    c << -1.0,
         -1.0,
          0,
          0,
          0;
    
    IP ret(A, b, c, 15, 1e-12);
    ret.solve();
    debug(ret.c.dot(ret.x));
    debug(ret.x);;
    debug(ret.mu);
    return 0;
}
