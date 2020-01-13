#include <iostream>
#include <regex>
#include <unistd.h>
#include <sstream>
#include <iomanip>

#include <map>

#include "colormanip.h"

#include "geometry/geometry.hpp"
#include "loaders/loader.hpp"
#include "methods/hho"
#include "solvers/solver.hpp"
#include "core/loaders/loader.hpp"
#include "cfem/cfem.hpp"

//////////////////////////////////////////////////////////////////////////////////
/////////////////////////////   LAGRANGE BASES    ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

using namespace disk;

/* Generic template for Lagrange bases. */
template<typename MeshType, typename Element>
struct Lagrange_scalar_basis
{
    static_assert(sizeof(MeshType) == -1, "Lagrange_scalar_basis: not suitable for the requested kind of mesh");
    static_assert(sizeof(Element) == -1,
                  "Lagrange_scalar_basis: not suitable for the requested kind of element");
};

/* Basis 'factory'. */
template<typename MeshType, typename ElementType>
auto
make_scalar_Lagrange_basis(const MeshType& msh, const ElementType& elem, size_t degree)
{
    return Lagrange_scalar_basis<MeshType, ElementType>(msh, elem, degree);
}


/* Specialization for 2D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class Lagrange_scalar_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, Dynamic, 2>     gradient_type;
    typedef Matrix<scalar_type, Dynamic, 1>     function_type;

  private:
    std::vector<point_type>       ref_points;
    size_t                        basis_degree, basis_size;

#ifdef POWER_CACHE
    mutable std::vector<scalar_type> power_cache;
#endif

  public:
    Lagrange_scalar_basis(const mesh_type& msh, const cell_type& cl, size_t degree)
    {
        if( degree > 1 )
            throw std::invalid_argument("degree != 1 not yet supported");
        basis_degree = degree;

        if(degree == 0)
        {
            basis_size  = 1;
            ref_points.reserve(basis_size);
            ref_points.push_back( barycenter(msh, cl) );
        }
        else // degree == 1
        {
            basis_size   = 3;
            auto pts = points(msh, cl);
            ref_points.reserve(basis_size);
            ref_points.push_back( pts[0] );
            ref_points.push_back( pts[1] );
            ref_points.push_back( pts[2] );
        }
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        if(basis_degree == 0)
            ret(0) = 1.0;
        else
        {   // basis_degree == 1
            auto pts = ref_points;
            auto x0 = pts[0].x(); auto y0 = pts[0].y();
            auto x1 = pts[1].x(); auto y1 = pts[1].y();
            auto x2 = pts[2].x(); auto y2 = pts[2].y();

            auto m = (x1*y2 - y1*x2 - x0*(y2 - y1) + y0*(x2 - x1));

            ret(0) = (x1*y2 - y1*x2 - pt.x() * (y2 - y1) + pt.y() * (x2 - x1)) / m;
            ret(1) = (x2*y0 - y2*x0 + pt.x() * (y2 - y0) - pt.y() * (x2 - x0)) / m;
            ret(2) = (x0*y1 - y0*x1 - pt.x() * (y1 - y0) + pt.y() * (x1 - x0)) / m;
        }
        return ret;
    }

    gradient_type
    eval_gradients(const point_type& pt) const
    {
        gradient_type ret = gradient_type::Zero(basis_size, 2);

        if(basis_degree == 0)
        {
            ret(0,0) = 0.0;
            ret(0,1) = 0.0;
        }
        else
        {   // basis_degree == 1
            auto pts = ref_points;
            auto x0 = pts[0].x(); auto y0 = pts[0].y();
            auto x1 = pts[1].x(); auto y1 = pts[1].y();
            auto x2 = pts[2].x(); auto y2 = pts[2].y();

            auto m = (x1*y2 - y1*x2 - x0*(y2 - y1) + y0*(x2 - x1));

            ret(0,0) = (y1 - y2) / m;
            ret(1,0) = (y2 - y0) / m;
            ret(2,0) = (y0 - y1) / m;
            ret(0,1) = (x2 - x1) / m;
            ret(1,1) = (x0 - x2) / m;
            ret(2,1) = (x1 - x0) / m;
        }
        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};

/* Specialization for 2D meshes, faces */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class Lagrange_scalar_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::face>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type      point_type;
    typedef typename mesh_type::face            face_type;
    typedef Matrix<scalar_type, Dynamic, 1>     function_type;

  private:
    size_t      basis_degree, basis_size;

#ifdef POWER_CACHE
    mutable std::vector<scalar_type> power_cache;
#endif

  public:
    Lagrange_scalar_basis(const mesh_type& msh, const face_type& fc, size_t degree)
    {
        if( degree != 0 )
            throw std::invalid_argument("degree != 0 not yet supported");
        basis_degree = degree;
        basis_size   = degree + 1;
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        ret(0) = 1.0;
        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};


/////////////////////
/////////////////////  VECTOR BASES

/* Generic template for bases. */
template<typename MeshType, typename Element>
struct Lagrange_vector_basis
{
    static_assert(sizeof(MeshType) == -1, "Lagrange_vector_basis: not suitable for the requested kind of mesh");
    static_assert(sizeof(Element) == -1,
                  "Lagrange_vector_basis: not suitable for the requested kind of element");
};

/* Basis 'factory'. */
template<typename MeshType, typename ElementType>
auto
make_vector_Lagrange_basis(const MeshType& msh, const ElementType& elem, size_t degree)
{
    return Lagrange_vector_basis<MeshType, ElementType>(msh, elem, degree);
}


/* Specialization for 2D meshes, cells */
template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
class Lagrange_vector_basis<Mesh<T, 2, Storage>, typename Mesh<T, 2, Storage>::cell>
{

  public:
    typedef Mesh<T, 2, Storage>                 mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::cell            cell_type;
    typedef typename mesh_type::point_type      point_type;
    typedef Matrix<scalar_type, 2, 2>           gradient_type;
    typedef Matrix<scalar_type, Dynamic, 2>     function_type;
    typedef Matrix<scalar_type, Dynamic, 1>     divergence_type;

  private:
    size_t basis_degree, basis_size;

    typedef Lagrange_scalar_basis<mesh_type, cell_type>    scalar_basis_type;
    scalar_basis_type                                      scalar_basis;

  public:
    Lagrange_vector_basis(const mesh_type& msh, const cell_type& cl, size_t degree) :
      scalar_basis(msh, cl, degree)
    {
        basis_degree = degree;
        basis_size   = 2 * scalar_basis.size();
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size, 2);

        const auto phi = scalar_basis.eval_functions(pt);

        for (size_t i = 0; i < scalar_basis.size(); i++)
        {
            ret(2 * i, 0)     = phi(i);
            ret(2 * i + 1, 1) = phi(i);
        }
        return ret;
    }

    eigen_compatible_stdvector<gradient_type>
    eval_gradients(const point_type& pt) const
    {
        eigen_compatible_stdvector<gradient_type> ret;
        ret.reserve(basis_size);

        const function_type dphi = scalar_basis.eval_gradients(pt);

        for (size_t i = 0; i < scalar_basis.size(); i++)
        {
            const Matrix<scalar_type, 1, 2> dphi_i = dphi.row(i);
            gradient_type                   g;

            g        = gradient_type::Zero();
            g.row(0) = dphi_i;
            ret.push_back(g);

            g        = gradient_type::Zero();
            g.row(1) = dphi_i;
            ret.push_back(g);
        }
        assert(ret.size() == basis_size);
        return ret;
    }

    // eigen_compatible_stdvector<gradient_type>
    // eval_sgradients(const point_type& pt) const
    // {
    //     eigen_compatible_stdvector<gradient_type> ret;
    //     ret.reserve(basis_size);

    //     const function_type dphi = scalar_basis.eval_gradients(pt);

    //     for (size_t i = 0; i < scalar_basis.size(); i++)
    //     {
    //         const Matrix<scalar_type, 1, 2> dphi_i = dphi.row(i);
    //         gradient_type                   g;

    //         g        = gradient_type::Zero();
    //         g.row(0) = dphi_i;
    //         ret.push_back(0.5 * (g + g.transpose()));

    //         g        = gradient_type::Zero();
    //         g.row(1) = dphi_i;
    //         ret.push_back(0.5 * (g + g.transpose()));
    //     }
    //     assert(ret.size() == basis_size);

    //     return ret;
    // }

    // Matrix<scalar_type, Dynamic, 1>
    // eval_curls(const point_type& pt) const
    // {
    //     Matrix<scalar_type, Dynamic, 1> ret = Matrix<scalar_type, Dynamic, 1>::Zero(basis_size);

    //     const function_type dphi = scalar_basis.eval_gradients(pt);

    //     size_t j = 0;
    //     for (size_t i = 0; i < scalar_basis.size(); i++)
    //     {
    //         Matrix<scalar_type, 1, 2> dphi_i = dphi.row(i);

    //         ret(j++) = dphi_i(1);
    //         ret(j++) = -dphi_i(0);
    //     }
    //     return ret;
    // }

    divergence_type
    eval_divergences(const point_type& pt) const
    {
        divergence_type ret = divergence_type::Zero(basis_size);

        const function_type dphi = scalar_basis.eval_gradients(pt);

        for (size_t i = 0; i < scalar_basis.size(); i++)
        {
            ret(2 * i)     = dphi(i, 0);
            ret(2 * i + 1) = dphi(i, 1);
        }

        return ret;
    }

    size_t
    size() const
    {
        return basis_size;
    }

    size_t
    degree() const
    {
        return basis_degree;
    }
};

/////////////////////////////  ASSEMBLY ROUTINES  ////////////////////////////////
template<typename Mesh>
std::pair<Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>,
          Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>>
make_vector_hho_gradrec_Lag(const Mesh&                     msh,
                            const typename Mesh::cell_type& cl,
                            const hho_degree_info&          di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    // const auto cb = make_scalar_monomial_basis(msh, cl, celdeg);
    const auto cb = make_scalar_Lagrange_basis(msh, cl, celdeg);
    // const auto gb = make_vector_monomial_basis(msh, cl, graddeg);
    const auto gb = make_vector_Lagrange_basis(msh, cl, graddeg);

    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);
    const auto gbs = gb.size();

    const auto num_faces = howmany_faces(msh, cl);

    const matrix_type gr_lhs = make_mass_matrix(msh, cl, gb);
    matrix_type       gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    // (vT, div(tau))_T
    if (graddeg > 0)
    {
        const auto qps = integrate(msh, cl, celdeg + graddeg - 1);
        for (auto& qp : qps)
        {
            const auto c_phi = cb.eval_functions(qp.point());
            const auto g_dphi  = gb.eval_divergences(qp.point());
            const vector_type qp_g_dphi = qp.weight() * g_dphi;

            gr_rhs.block(0, 0, gbs, cbs) -= priv::outer_product(qp_g_dphi, c_phi);
        }
    }

    // (vF, tau.n)_F
    const auto fcs = faces(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = normal(msh, cl, fc);
        // const auto fb = make_scalar_monomial_basis(msh, fc, facdeg);
        const auto fb = make_scalar_Lagrange_basis(msh, fc, facdeg);

        const auto qps_f = integrate(msh, fc, graddeg + facdeg);
        for (auto& qp : qps_f)
        {
            const vector_type f_phi      = fb.eval_functions(qp.point());
            const auto        g_phi      = gb.eval_functions(qp.point());
            const vector_type qp_g_phi_n = g_phi * (qp.weight() * n);

            gr_rhs.block(0, cbs + i * fbs, gbs, fbs) += priv::outer_product(qp_g_phi_n, f_phi);
        }
    }

    matrix_type oper = gr_lhs.ldlt().solve(gr_rhs);
    matrix_type data = gr_rhs.transpose() * oper;

    return std::make_pair(oper, data);
}



// we compute the stabilisation 1/h_F(uF-pi^k_F(uT), vF-pi^k_F(vT))_F
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_scalar_hdg_stabilization_Lag(const Mesh& msh, const typename Mesh::cell_type& cl, const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();

    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);

    const auto num_faces = howmany_faces(msh, cl);
    const auto total_dofs = cbs + num_faces * fbs;

    matrix_type       data = matrix_type::Zero(total_dofs, total_dofs);
    const matrix_type If   = matrix_type::Identity(fbs, fbs);

    // auto cb = make_scalar_monomial_basis(msh, cl, celdeg);
    auto cb = make_scalar_Lagrange_basis(msh, cl, celdeg);
    const auto fcs = faces(msh, cl);

    for (size_t i = 0; i < num_faces; i++)
    {
        const auto fc = fcs[i];
        const auto h  = diameter(msh, fc);
        // auto fb = make_scalar_monomial_basis(msh, fc, facdeg);
        auto fb = make_scalar_Lagrange_basis(msh, fc, facdeg);

        matrix_type oper  = matrix_type::Zero(fbs, total_dofs);
        matrix_type tr    = matrix_type::Zero(fbs, total_dofs);
        matrix_type mass  = make_mass_matrix(msh, fc, fb);
        matrix_type trace = matrix_type::Zero(fbs, cbs);

        oper.block(0, cbs + i  * fbs, fbs, fbs) = -If;

        const auto qps = integrate(msh, fc, facdeg + celdeg);
        for (auto& qp : qps)
        {
            const auto c_phi = cb.eval_functions(qp.point());
            const auto f_phi = fb.eval_functions(qp.point());

            assert(c_phi.rows() == cbs);
            assert(f_phi.rows() == fbs);
            assert(c_phi.cols() == f_phi.cols());

            trace += priv::outer_product(priv::inner_product(qp.weight(), f_phi), c_phi);
        }

        tr.block(0, cbs + i * fbs, fbs, fbs) = -mass;
        tr.block(0, 0, fbs, cbs) = trace;

        oper.block(0, 0, fbs, cbs) = mass.ldlt().solve(trace);
        data += oper.transpose() * tr * (1./h);
    }

    return data;
}


/////////////////////////////      ASSEMBLER      ////////////////////////////////

template<typename Mesh>
class diffusion_assembler
{
    using T = typename Mesh::coordinate_type;
    typedef disk::BoundaryConditions<Mesh, true> boundary_type;

    std::vector<size_t>     compress_table;
    std::vector<size_t>     expand_table;
    hho_degree_info         di;
    std::vector<Triplet<T>> triplets;
    bool                    use_bnd;
    std::vector< Matrix<T, Dynamic, Dynamic> > loc_LHS;
    std::vector< Matrix<T, Dynamic, 1> >       loc_RHS;

    size_t num_all_faces, num_dirichlet_faces, num_other_faces, system_size;

    class assembly_index
    {
        size_t  idx;
        bool    assem;

    public:
        assembly_index(size_t i, bool as)
            : idx(i), assem(as)
        {}

        operator size_t() const
        {
            if (!assem)
                throw std::logic_error("Invalid assembly_index");

            return idx;
        }

        bool assemble() const
        {
            return assem;
        }

        friend std::ostream& operator<<(std::ostream& os, const assembly_index& as)
        {
            os << "(" << as.idx << "," << as.assem << ")";
            return os;
        }
    };

public:
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    SparseMatrix<T> LHS;
    vector_type     RHS;

    diffusion_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi), use_bnd(false)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return msh.is_boundary(fc);
        };

        num_all_faces       = msh.faces_size();
        num_dirichlet_faces = std::count_if(msh.faces_begin(), msh.faces_end(), is_dirichlet);
        num_other_faces     = num_all_faces - num_dirichlet_faces;

        compress_table.resize( num_all_faces );
        expand_table.resize( num_other_faces );

        size_t compressed_offset = 0;
        for (size_t i = 0; i < num_all_faces; i++)
        {
            const auto fc = *std::next(msh.faces_begin(), i);
            if (!is_dirichlet(fc))
            {
                compress_table.at(i)               = compressed_offset;
                expand_table.at(compressed_offset) = i;
                compressed_offset++;
            }
        }

        auto num_cells = msh.cells_size();
        loc_LHS.resize( num_cells );
        loc_RHS.resize( num_cells );

        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);
        system_size    = 2 * (cbs * num_cells + fbs * num_other_faces);

        LHS = SparseMatrix<T>(system_size, system_size);
        RHS = vector_type::Zero(system_size);
    }

    void
    set_loc_mat(const Mesh&                     msh,
                const typename Mesh::cell_type& cl,
                const matrix_type&              lhs,
                const vector_type&              rhs)
    {
        auto cell_offset = offset(msh, cl);
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;
    }

    template<typename Function>
    void
    assemble(const Mesh&                     msh,
             const typename Mesh::cell_type& cl,
             const matrix_type&              lhs,
             const vector_type&              rhs,
             const Function&                 dirichlet_bf)
    {
        if(use_bnd)
            throw std::invalid_argument("diffusion_assembler: you have to use boundary type");

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return msh.is_boundary(fc);
        };

        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        const auto fcs = faces(msh, cl);

        std::vector<assembly_index> asm_map;
        asm_map.reserve(cbs + fcs.size() * fbs);

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * cbs;

        for (size_t i = 0; i < cbs; i++)
                asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        vector_type dirichlet_data = vector_type::Zero(cbs + fcs.size()*fbs);

        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc              = fcs[face_i];
            const auto face_offset     = priv::offset(msh, fc);
            const auto face_LHS_offset = msh.cells_size() * cbs + compress_table.at(face_offset) * fbs;

            const bool dirichlet = is_dirichlet(fc);

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                auto fb = make_scalar_Lagrange_basis(msh, fc, di.face_degree());
                dirichlet_data.block(cbs + face_i * fbs, 0, fbs, 1) =
                  project_function(msh, fc, fb, dirichlet_bf, di.face_degree());
            }
        }

        for (size_t i = 0; i < lhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs(i,j) * dirichlet_data(j);
            }
        }

        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(asm_map[i]) += rhs(i);
        }

    } // assemble()


    // init : set no contact constraint and assemble matrix
    // (first iteration matrix)
    template<typename Function>
    void
    init(const Mesh&                     msh,
         const Function&                 dirichlet_bf)
    {
        // assemble all local contributions for Laplacian part
        for (auto& cl : msh)
        {
            auto cell_offset = offset(msh, cl);
            assemble(msh, cl, loc_LHS.at(cell_offset), loc_RHS.at(cell_offset), dirichlet_bf);
        }
        // assemble constraints (no contact)
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        auto mult_offset = cbs * msh.cells_size() + fbs * num_other_faces;
        for(size_t i = 0; i < mult_offset; i++)
        {
            triplets.push_back( Triplet<T>(mult_offset + i, mult_offset + i, 1.0) );
        }

        // end assembly
        finalize();
    }

    template<typename Function>
    vector_type
    take_u(const Mesh& msh, const typename Mesh::cell_type& cl,
    const vector_type& solution, const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto fcs = faces(msh, cl);

        auto num_faces = fcs.size();

        auto cell_offset        = offset(msh, cl);
        auto cell_SOL_offset    = cell_offset * cbs;

        vector_type ret = vector_type::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                return msh.is_boundary(fc);
            };

            bool dirichlet = is_dirichlet(fc);

            if (dirichlet)
            {
                // auto fb = make_scalar_monomial_basis(msh, fc, di.face_degree());
                auto fb = make_scalar_Lagrange_basis(msh, fc, di.face_degree());

                matrix_type mass = make_mass_matrix(msh, fc, fb, di.face_degree());
                vector_type rhs = make_rhs(msh, fc, fb, dirichlet_bf, di.face_degree());
                ret.block(cbs + face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
            }
            else
            {
                auto face_offset = priv::offset(msh, fc);
                auto face_SOL_offset = msh.cells_size() * cbs + compress_table.at(face_offset)*fbs;
                ret.block(cbs + face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }


    vector_type
    take_mult(const Mesh& msh, const typename Mesh::cell_type& cl,
              const vector_type& solution)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto fcs = faces(msh, cl);

        auto num_faces = fcs.size();

        auto mult_offset = cbs * msh.cells_size() + fbs * num_other_faces;
        auto cell_offset        = offset(msh, cl);
        auto cell_SOL_offset    = mult_offset + cell_offset * cbs;

        vector_type ret = vector_type::Zero(cbs + num_faces*fbs);
        ret.block(0, 0, cbs, 1) = solution.block(cell_SOL_offset, 0, cbs, 1);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                return msh.is_boundary(fc);
            };

            bool dirichlet = is_dirichlet(fc);

            if (dirichlet)
            {
                // no contact on the boundary
                for(size_t i = 0; i < fbs; i++)
                    ret(cbs + face_i*fbs + i) = 0.0;
            }
            else
            {
                auto face_offset = priv::offset(msh, fc);
                auto face_SOL_offset = mult_offset
                    + msh.cells_size() * cbs + compress_table.at(face_offset)*fbs;
                ret.block(cbs + face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    void finalize(void)
    {
        LHS.setFromTriplets( triplets.begin(), triplets.end() );
        triplets.clear();

        dump_sparse_matrix(LHS, "diff.dat");
    }

    size_t num_assembled_faces() const
    {
        return num_other_faces;
    }

};
template<typename Mesh>
auto make_assembler_Lag(const Mesh& msh, hho_degree_info hdi)
{
    return diffusion_assembler<Mesh>(msh, hdi);
}



/////////////////////////////   OUTPUT   OBJECTS  ////////////////////////////////
template<typename T>
class postprocess_output_object {

public:
    postprocess_output_object()
    {}

    virtual bool write() = 0;
};

template<typename T>
class gnuplot_output_object : public postprocess_output_object<T>
{
    std::string                                 output_filename;
    std::vector< std::pair< point<T,2>, T > >   data;

public:
    gnuplot_output_object(const std::string& filename)
        : output_filename(filename)
    {}

    void add_data(const point<T,2>& pt, const T& val)
    {
        data.push_back( std::make_pair(pt, val) );
    }

    bool write()
    {
        std::ofstream ofs(output_filename);

        for (auto& d : data)
            ofs << d.first.x() << " " << d.first.y() << " " << d.second << std::endl;

        ofs.close();

        return true;
    }
};

template<typename T>
class postprocess_output
{
    std::list< std::shared_ptr< postprocess_output_object<T>> >     postprocess_objects;

public:
    postprocess_output()
    {}

    void add_object( std::shared_ptr<postprocess_output_object<T>> obj )
    {
        postprocess_objects.push_back( obj );
    }

    bool write(void) const
    {
        for (auto& obj : postprocess_objects)
            obj->write();

        return true;
    }
};


////////////////////////////   MAIN  CODE  ////////////////////////////////

using namespace disk;

template<typename Mesh>
typename Mesh::coordinate_type
run_membranes_solver(const Mesh& msh, size_t degree)
{
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;

    hho_degree_info hdi(degree+1, degree);

    auto rhs_fun = [](const point_type& pt) -> T {
        auto sin_px = std::sin(M_PI * pt.x());
        auto sin_py = std::sin(M_PI * pt.y());
        return 2.0 * M_PI * M_PI * sin_px * sin_py;
    };
    auto sol_fun = [](const point_type& pt) -> T {
        auto sin_px = std::sin(M_PI * pt.x());
        auto sin_py = std::sin(M_PI * pt.y());
        return sin_px * sin_py;
    };

    auto assembler_sc = make_diffusion_assembler(msh, hdi);
    auto assembler = make_assembler_Lag(msh, hdi);

    bool scond = false; // static condensation

    for (auto& cl : msh)
    {
        // auto cb = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
        auto cb = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
        auto gr     = make_vector_hho_gradrec_Lag(msh, cl, hdi);
        auto stab   = make_scalar_hdg_stabilization_Lag(msh, cl, hdi);
        auto rhs    = make_rhs(msh, cl, cb, rhs_fun);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = gr.second + stab;
        if(scond) {
            auto sc     = make_scalar_static_condensation(msh, cl, hdi, A, rhs);
            assembler_sc.assemble(msh, cl, sc.first, sc.second, sol_fun);
        }
        else
        {
            assembler.set_loc_mat(msh, cl, A, rhs);
        }
    }

    size_t systsz, nnz;
    if(scond)
    {
        assembler_sc.finalize();
        systsz = assembler_sc.LHS.rows();
        // nnz = assembler_sc.LHS.nonZeros();
    }
    else
    {
        assembler.init(msh, sol_fun);
        systsz = assembler.LHS.rows();
        // nnz = assembler.LHS.nonZeros();
    }

    //std::cout << "Mesh elements: " << msh.cells_size() << std::endl;
    //std::cout << "Mesh faces: " << msh.faces_size() << std::endl;
    //std::cout << "Dofs: " << systsz << std::endl;

    dynamic_vector<T> sol = dynamic_vector<T>::Zero(systsz);

    disk::solvers::pardiso_params<T> pparams;
    pparams.report_factorization_Mflops = false;

    if(scond)
        mkl_pardiso(pparams, assembler_sc.LHS, assembler_sc.RHS, sol);
    else
        mkl_pardiso(pparams, assembler.LHS, assembler.RHS, sol);

    T error = 0.0;

    //std::ofstream ofs("sol.dat");

    postprocess_output<T>  postoutput;

    auto diff_uT_gp  = std::make_shared< gnuplot_output_object<T> >("diff_uT.dat");
    auto uT_gp  = std::make_shared< gnuplot_output_object<T> >("uT.dat");
    auto multT_gp  = std::make_shared< gnuplot_output_object<T> >("multT.dat");

    for (auto& cl : msh)
    {
        // auto cb     = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
        auto cb     = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();

        // Eigen::Matrix<T, Eigen::Dynamic, 1> realsol = project_function(msh, cl, hdi, sol_fun, 2);
        Eigen::Matrix<T, Eigen::Dynamic, 1> realsol = project_function(msh, cl, cb, sol_fun, 2);
        Eigen::Matrix<T, Eigen::Dynamic, 1> fullsol, mult_sol;
        auto gr     = make_vector_hho_gradrec_Lag(msh, cl, hdi);
        auto stab   = make_scalar_hdg_stabilization_Lag(msh, cl, hdi);
        auto rhs    = make_rhs(msh, cl, cb, rhs_fun);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = gr.second + stab;

        if(scond)
        {
            Eigen::Matrix<T, Eigen::Dynamic, 1> locsol =
                assembler_sc.take_local_data(msh, cl, sol, sol_fun);

            fullsol = make_scalar_static_decondensation(msh, cl, hdi, A, rhs, locsol);
        }
        else
            fullsol = assembler.take_u(msh, cl, sol, sol_fun);

        auto diff = realsol - fullsol.head( cb.size() );
        error += diff.dot(A.block(0,0,cbs,cbs) * diff);

        auto bar = barycenter(msh, cl);
        diff_uT_gp->add_data( bar, diff.dot(A.block(0,0,cbs,cbs) * diff) );


        auto test = fullsol.head( cb.size() );
        T sol_uT = test.dot( cb.eval_functions(bar) );
        // T sol = fullsol.head( cb.size() ).dot( cb );
        uT_gp->add_data( bar, sol_uT );


        if(scond) {}
        else
            mult_sol = assembler.take_mult(msh, cl, sol);

        auto test2 = mult_sol.head( cb.size() );
        T sol_multT = test2.dot( cb.eval_functions(bar) );
        multT_gp->add_data( bar, sol_multT );
    }

    //std::cout << std::sqrt(error) << std::endl;

    //ofs.close();

    postoutput.add_object(diff_uT_gp);
    postoutput.add_object(uT_gp);
    postoutput.add_object(multT_gp);
    postoutput.write();

    std::cout << "ended run : error is " << std::sqrt(error) << std::endl;
    
    return std::sqrt(error);
}

using namespace Eigen;

int main(void)
{
    using T = double;

    
    typedef disk::generic_mesh<T, 2>  mesh_type;
    
    
    if(1)
    {
        std::vector<std::string> meshfiles;
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_1.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_2.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_3.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_5.typ1");
    

        for(size_t i=0; i < meshfiles.size(); i++)
        {
            mesh_type msh;
            disk::fvca5_mesh_loader<T, 2> loader;
            if (!loader.read_mesh(meshfiles.at(i)) )
            {
                std::cout << "Problem loading mesh." << std::endl;
            }
            loader.populate_mesh(msh);
            run_membranes_solver(msh, 0);
        }

    }
    else
    {
        mesh_type msh;
        disk::fvca5_mesh_loader<T, 2> loader;
        std::string mesh_filename = "../../../diskpp/meshes/2D_triangles/fvca5/mesh1_1.typ1";
        if (!loader.read_mesh(mesh_filename) )
        {
            std::cout << "Problem loading mesh." << std::endl;
        }
        loader.populate_mesh(msh);
        run_membranes_solver(msh, 0);
    }
    return 0;
}
