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
    std::vector<point_type>       vertices;
    size_t                        basis_degree, basis_size;

#ifdef POWER_CACHE
    mutable std::vector<scalar_type> power_cache;
#endif

  public:
    Lagrange_scalar_basis(const mesh_type& msh, const cell_type& cl, size_t degree)
    {
        if( degree > 2 )
            throw std::invalid_argument("degree > 2 not yet supported");
        basis_degree = degree;

        // store the vertices
        vertices = points(msh, cl);

        if(degree == 0)
            basis_size  = 1;
        else if(degree == 1)
            basis_size   = 3;
        else // degree == 2
            basis_size   = 6;
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        if(basis_degree == 0)
            ret(0) = 1.0;
        else if(basis_degree == 1)
        {
            ret = bar_coord(pt);
        }
        else // degree == 2
        {
            auto bar_c = bar_coord(pt);
            // 0-2 : vertices
            for(size_t i = 0; i < 3; i++)
                ret[i] = bar_c[i] * (2.0*bar_c[i] - 1.0);

            // 3-5 : mid-points (12,02,01)
            ret[3] = 4.0*bar_c[1]*bar_c[2];
            ret[4] = 4.0*bar_c[0]*bar_c[2];
            ret[5] = 4.0*bar_c[0]*bar_c[1];
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
        else if(basis_degree == 1)
        {
            ret = bar_coord_grad(pt);
        }
        else // degree == 2
        {
            auto bar_c = bar_coord(pt);
            auto bar_c_g = bar_coord_grad(pt);

            // 0-2 : vertices
            for(size_t i = 0; i < 3; i++)
            {
                ret(i,0) = (4.0*bar_c[i]-1.0) * bar_c_g(i,0);
                ret(i,1) = (4.0*bar_c[i]-1.0) * bar_c_g(i,1);
            }
            // 3-5 : mid-points (12,02,01)
            for(size_t j = 0; j < 2; j++)
            {
                ret(3,j) = 4.0*( bar_c[1]*bar_c_g(2,j) + bar_c[2]*bar_c_g(1,j) );
                ret(4,j) = 4.0*( bar_c[0]*bar_c_g(2,j) + bar_c[2]*bar_c_g(0,j) );
                ret(5,j) = 4.0*( bar_c[0]*bar_c_g(1,j) + bar_c[1]*bar_c_g(0,j) );
            }
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


    // barycentric coordinates
    function_type
    bar_coord(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        auto pts = vertices;
        auto x0 = pts[0].x(); auto y0 = pts[0].y();
        auto x1 = pts[1].x(); auto y1 = pts[1].y();
        auto x2 = pts[2].x(); auto y2 = pts[2].y();

        auto m = (x1*y2 - y1*x2 - x0*(y2 - y1) + y0*(x2 - x1));

        ret(0) = (x1*y2 - y1*x2 - pt.x() * (y2 - y1) + pt.y() * (x2 - x1)) / m;
        ret(1) = (x2*y0 - y2*x0 + pt.x() * (y2 - y0) - pt.y() * (x2 - x0)) / m;
        ret(2) = (x0*y1 - y0*x1 - pt.x() * (y1 - y0) + pt.y() * (x1 - x0)) / m;

        return ret;
    }

    // gradients of the barycentric coordinates
    gradient_type
    bar_coord_grad(const point_type& pt) const
    {
        gradient_type ret = gradient_type::Zero(basis_size, 2);

        auto pts = vertices;
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

        return ret;
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
    point_type  face_ref, base;
    scalar_type face_h;
    size_t      basis_degree, basis_size;

#ifdef POWER_CACHE
    mutable std::vector<scalar_type> power_cache;
#endif

  public:
    Lagrange_scalar_basis(const mesh_type& msh, const face_type& fc, size_t degree)
    {
        if( degree > 1 )
            throw std::invalid_argument("degree > 1 not yet supported");
        basis_degree = degree;
        basis_size   = degree + 1;

        const auto pts = points(msh, fc);
        face_ref = pts[0];
        face_h       = diameter(msh, fc);
        base = pts[1] - pts[0];
    }

    function_type
    eval_functions(const point_type& pt) const
    {
        function_type ret = function_type::Zero(basis_size);

        // pos on the face
        const auto v   = base.to_vector();
        const auto t   = (pt - face_ref).to_vector();
        const auto dot = v.dot(t);
        const auto pos = dot/(face_h*face_h); // 0 -> pts[0], 1 -> pts[1]

        if(basis_degree == 0)
            ret(0) = 1.0;
        else // degree == 1
        {
            ret(0) = - pos + 1.0;
            ret(1) = pos;
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


//////////////////////////////////////////////////////////////////////////////////
/////////////////////////////     ASSEMBLERS      ////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////

template<typename Mesh>
class membrane_assembler
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

    membrane_assembler(const Mesh& msh, hho_degree_info hdi)
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
    assemble_mat(const Mesh&                     msh,
                 const typename Mesh::cell_type& cl,
                 const matrix_type&              lhs,
                 const Function&                 dirichlet_bf)
    {
        if(use_bnd)
            throw std::invalid_argument("membrane_assembler: you have to use boundary type");

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
            }
        }

    } // assemble_mat()


    template<typename Function>
    void
    assemble_rhs(const Mesh&                     msh,
                 const typename Mesh::cell_type& cl,
                 const matrix_type&              lhs,
                 const vector_type&              rhs,
                 const Function&                 dirichlet_bf)
    {
        if(use_bnd)
            throw std::invalid_argument("membrane_assembler: you have to use boundary type");

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
                if ( !asm_map[j].assemble() )
                    RHS(asm_map[i]) -= lhs(i,j) * dirichlet_data(j);
            }
        }

        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(asm_map[i]) += rhs(i);
        }

    } // assemble_rhs()

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
            assemble_mat(msh, cl, loc_LHS.at(cell_offset), dirichlet_bf);
            assemble_rhs(msh, cl, loc_LHS.at(cell_offset), loc_RHS.at(cell_offset), dirichlet_bf);
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

    // update_mat : assemble matrix according to the previous iteration solution
    template<typename Function>
    void
    update_mat(const Mesh&                     msh,
               const vector_type&              prev_sol,
               const Function&                 dirichlet_bf)
    {
        // assemble all local contributions for Laplacian part
        for (auto& cl : msh)
        {
            auto cell_offset = offset(msh, cl);
            assemble_mat(msh, cl, loc_LHS.at(cell_offset), dirichlet_bf);
        }
        // assemble constraints
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        auto mult_offset = cbs * msh.cells_size() + fbs * num_other_faces;

        for(size_t i = 0; i < mult_offset; i++)
        {
            auto sol_u    = prev_sol(i);
            auto sol_mult = prev_sol(mult_offset + i);

            if(sol_u <= 0.0 && sol_mult >= 0)
                triplets.push_back( Triplet<T>(mult_offset + i, i, 1.0) );
            else
                triplets.push_back( Triplet<T>(mult_offset + i, mult_offset + i, 1.0) );
        }

        // identity block
        for(size_t i = 0; i < mult_offset; i++)
        {
            triplets.push_back( Triplet<T>(i, mult_offset + i, -1.0) );
        }

        // end assembly
        finalize();
    }


    bool
    stop(const Mesh&         msh,
         const vector_type&  sol)
    {
        T TOL = 1e-16;

        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        auto mult_offset = cbs * msh.cells_size() + fbs * num_other_faces;

        bool ret = true;
        for(size_t i = 0; i < mult_offset; i++)
        {
            auto sol_u    = sol(i);
            auto sol_mult = sol(mult_offset + i);

            if(sol_u < -TOL || sol_mult < -TOL)
            {
                ret = false;
                break;
            }
        }

        return ret;
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
    return membrane_assembler<Mesh>(msh, hdi);
}

////////  STATIC CONDENSATION
template<typename Mesh, typename T>
auto
make_membrane_SC(const Mesh&                                                      msh,
                 const typename Mesh::cell_type&                                  cl,
                 const hho_degree_info&                                           hdi,
                 const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& lhs,
                 const typename Eigen::Matrix<T, Eigen::Dynamic, 1>&              rhs,
                 const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& D)
{
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_type = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    const auto facdeg         = hdi.face_degree();
    const auto celdeg         = hdi.cell_degree();
    const auto num_face_dofs  = scalar_basis_size(facdeg, Mesh::dimension - 1);
    const auto num_cell_dofs  = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fcs            = faces(msh, cl);
    const auto num_faces      = fcs.size();
    const auto num_faces_dofs = num_faces * num_face_dofs;
    const auto num_total_dofs = num_cell_dofs + num_faces_dofs;

    assert(lhs.rows() == lhs.cols());
    assert(lhs.cols() == num_total_dofs);
    assert(D.rows()   == num_cell_dofs);
    assert(D.cols()   == 2 * num_cell_dofs);
    if ((rhs.size() != num_cell_dofs) && (rhs.size() != num_total_dofs))
    {
        throw std::invalid_argument("static condensation: incorrect size of the rhs");
    }

    const matrix_type K_TT = lhs.topLeftCorner(num_cell_dofs, num_cell_dofs);
    const matrix_type K_TF = lhs.topRightCorner(num_cell_dofs, num_faces_dofs);
    const matrix_type K_FT = lhs.bottomLeftCorner(num_faces_dofs, num_cell_dofs);
    const matrix_type K_FF = lhs.bottomRightCorner(num_faces_dofs, num_faces_dofs);

    assert(K_TT.cols() == num_cell_dofs);
    assert(K_TT.cols() + K_TF.cols() == lhs.cols());
    assert(K_TT.rows() + K_FT.rows() == lhs.rows());
    assert(K_TF.rows() + K_FF.rows() == lhs.rows());
    assert(K_FT.cols() + K_FF.cols() == lhs.cols());

    const vector_type cell_rhs  = rhs.head(num_cell_dofs);
    vector_type       faces_rhs = vector_type::Zero(num_faces_dofs);

    if (rhs.size() == num_total_dofs)
    {
        faces_rhs = rhs.tail(num_faces_dofs);
    }

    const matrix_type C_TT = D.topLeftCorner(num_cell_dofs, num_cell_dofs);
    const matrix_type D_TT = D.topRightCorner(num_cell_dofs, num_cell_dofs);

    const auto K_TT_ldlt = K_TT.ldlt();
    if (K_TT_ldlt.info() != Eigen::Success)
    {
        throw std::invalid_argument("static condensation: K_TT is not positive definite");
    }

    const matrix_type AL = K_TT_ldlt.solve(K_TF);
    const vector_type bL = K_TT_ldlt.solve(cell_rhs);

    const auto ID = matrix_type::Identity(num_cell_dofs, num_cell_dofs);
    const auto K_TT_inv = K_TT_ldlt.solve(ID);

    const auto E = C_TT * K_TT_inv + D_TT;
    const auto E_inv = E.inverse();

    const matrix_type E2 = E_inv * C_TT;
    const matrix_type E3 = K_TT_ldlt.solve(E2);

    const matrix_type AC = K_FF - K_FT * AL + K_FT * E3 * AL;
    const vector_type bC = faces_rhs - K_FT * bL + K_FT * E3 * bL;

    return std::make_pair(AC, bC);
}


//////////// MEMBRANE_STATIC_DECONDENSATION
template<typename Mesh, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
membrane_static_decondensation(const Mesh&                                                      msh,
                               const typename Mesh::cell_type&                                  cl,
                               const hho_degree_info&                                           hdi,
                               const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& lhs,
                               const typename Eigen::Matrix<T, Eigen::Dynamic, 1>&              rhs,
                               const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& D,
                               const typename Eigen::Matrix<T, Eigen::Dynamic, 1>&              solF)
{
    using matrix_type = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using vector_type = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    const auto facdeg         = hdi.face_degree();
    const auto celdeg         = hdi.cell_degree();
    const auto num_face_dofs  = scalar_basis_size(facdeg, Mesh::dimension - 1);
    const auto num_cell_dofs  = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fcs            = faces(msh, cl);
    const auto num_faces      = fcs.size();
    const auto num_faces_dofs = num_faces * num_face_dofs;
    const auto num_total_dofs = num_cell_dofs + num_faces_dofs;

    assert(lhs.rows() == lhs.cols());
    assert(lhs.cols() == num_total_dofs);

    if ((rhs.size() < num_cell_dofs))
    {
        throw std::invalid_argument("static condensation: incorrect size of the rhs");
    }

    const matrix_type K_TT = lhs.topLeftCorner(num_cell_dofs, num_cell_dofs);
    const matrix_type K_TF = lhs.topRightCorner(num_cell_dofs, num_faces_dofs);

    const matrix_type C_TT = D.topLeftCorner(num_cell_dofs, num_cell_dofs);
    const matrix_type D_TT = D.topRightCorner(num_cell_dofs, num_cell_dofs);

    vector_type uF = solF.head(num_faces_dofs);
    vector_type multF = solF.tail(num_faces_dofs);

    const auto K_TT_ldlt = K_TT.ldlt();
    if (K_TT_ldlt.info() != Eigen::Success)
    {
        throw std::invalid_argument("static condensation: K_TT is not positive definite");
    }

    const auto ID = matrix_type::Identity(num_cell_dofs, num_cell_dofs);
    const auto K_TT_inv = K_TT_ldlt.solve(ID);

    const auto E_inv = (C_TT * K_TT_inv + D_TT).inverse();
    const matrix_type E2 = E_inv * C_TT;

    const vector_type solT = K_TT_ldlt.solve(rhs.head(num_cell_dofs) - K_TF * uF)
        - K_TT_inv * E2 * K_TT_ldlt.solve(rhs.head(num_cell_dofs) - K_TF * uF);

    const vector_type multT = - E2 * K_TT.ldlt().solve(rhs.head(num_cell_dofs) - K_TF * uF);

    vector_type ret          = vector_type::Zero(2 * num_total_dofs);
    ret.head(num_cell_dofs)  = solT;
    ret.block(num_cell_dofs, 0, num_faces_dofs, 1) = uF;
    ret.block(num_total_dofs, 0, num_cell_dofs, 1) = multT;
    ret.tail(num_faces_dofs) = multF;

    return ret;
}

////////////////////////////////////////

template<typename Mesh>
class membrane_condensed_assembler
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
    std::vector< bool >     active_constr;

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

    membrane_condensed_assembler(const Mesh& msh, hho_degree_info hdi)
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
        system_size = 2 * fbs * num_other_faces;

        active_constr.resize(num_cells * cbs);

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

        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        for(size_t i = cell_offset * cbs; i < cell_offset * cbs + cbs; i++)
            active_constr.at(i) = false;
    }

    template<typename Function>
    void
    assemble_contrib(const Mesh&                     msh,
                     const typename Mesh::cell_type& cl,
                     const matrix_type&              lhs,
                     const vector_type&              rhs,
                     const vector_type&              solF,
                     const Function&                 dirichlet_bf)
    {
        if(use_bnd)
            throw std::invalid_argument("membrane_condensed_assembler: you have to use boundary type");

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return msh.is_boundary(fc);
        };

        auto cell_offset = offset(msh, cl);

        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        const auto fcs = faces(msh, cl);

        matrix_type D = matrix_type::Zero(cbs, 2*cbs);

        for(size_t i = 0; i < cbs; i++)
        {
            auto OFF = cell_offset * cbs;
            if( active_constr.at(OFF + i) )
                D(i,i) = 1.0;
            else
                D(i,cbs + i) = 1.0;
        }

        auto loc_sol = membrane_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

        vector_type solT = loc_sol.head(cbs);
        vector_type multT = loc_sol.block(cbs + fcs.size() * fbs, 0, cbs, 1);


        D = matrix_type::Zero(cbs, 2*cbs);
        auto pts2 = points(msh, cl);
        for(size_t i = 0; i < cbs; i++)
        {
            auto sol_u    = solT(i);
            auto sol_mult = multT(i);

            if(sol_u <= 0.0 && sol_mult >= 0)
            {
                active_constr.at(cell_offset * cbs + i) = true;
                D(i,i) = 1.0;
            }
            else if(sol_u >= 0.0 && sol_mult <= 0)
            {
                active_constr.at(cell_offset * cbs + i) = false;
                D(i,cbs + i) = 1.0;
            }
            else if(sol_u * sol_mult > 0.0)
            {
                // this case is considered because of the error made during SC
                if(sol_u < sol_mult)
                {
                    active_constr.at(cell_offset * cbs + i) = true;
                    D(i,i) = 1.0;
                }
                else
                {
                    active_constr.at(cell_offset * cbs + i) = false;
                    D(i,cbs + i) = 1.0;
                }
            }
            else
                throw std::logic_error("we should not arrive here !!");
        }

        auto SC = make_membrane_SC(msh, cl, di, lhs, rhs, D);
        matrix_type lhs_sc = SC.first;
        vector_type rhs_sc = SC.second;

        std::vector<assembly_index> asm_map;
        asm_map.reserve(fcs.size() * fbs);

        vector_type dirichlet_data = vector_type::Zero(fcs.size()*fbs);

        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc              = fcs[face_i];
            const auto face_offset     = priv::offset(msh, fc);
            const auto face_LHS_offset = compress_table.at(face_offset) * fbs;

            const bool dirichlet = is_dirichlet(fc);

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                auto fb = make_scalar_Lagrange_basis(msh, fc, di.face_degree());
                dirichlet_data.block(face_i * fbs, 0, fbs, 1) =
                  project_function(msh, fc, fb, dirichlet_bf, di.face_degree());
            }
        }

        for (size_t i = 0; i < lhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs_sc(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs_sc(i,j) * dirichlet_data(j);
            }
        }

        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(asm_map[i]) += rhs_sc(i);
        }

    } // assemble_contrib()


    // init : set no contact constraint and assemble matrix
    // (matrix for the first iteration)
    template<typename Function>
    void
    init(const Mesh&                     msh,
         const Function&                 dirichlet_bf)
    {
        const auto facdeg  = di.face_degree();
        const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);

        // assemble all local contributions for Laplacian part
        for (auto& cl : msh)
        {
            // init solution with no contact
            auto num_faces = howmany_faces(msh, cl);
            auto num_dofs = num_faces * fbs;
            auto cell_offset = offset(msh, cl);
            vector_type solF = vector_type::Zero(2 * num_dofs);
            for(size_t i=0; i<num_dofs; i++)
                solF(i) = 1.0;

            assemble_contrib(msh, cl, loc_LHS.at(cell_offset), loc_RHS.at(cell_offset),solF, dirichlet_bf);
        }
        // assemble constraints (no contact)
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        auto mult_offset = fbs * num_other_faces;
        for(size_t i = 0; i < mult_offset; i++)
        {
            triplets.push_back( Triplet<T>(mult_offset + i, mult_offset + i, 1.0) );
        }

        // end assembly
        finalize();
    }


    template<typename Function>
    vector_type
    get_solF(const Mesh& msh, const typename Mesh::cell_type& cl,
             const vector_type& solution, const Function& dirichlet_bf)
    {
        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto fcs = faces(msh, cl);

        auto num_faces = fcs.size();

        vector_type ret = vector_type::Zero(2*num_faces*fbs);

        auto mult_offset = fbs * num_other_faces;

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                return msh.is_boundary(fc);
            };

            bool dirichlet = is_dirichlet(fc);

            if (dirichlet)
            {
                auto fb = make_scalar_Lagrange_basis(msh, fc, di.face_degree());

                matrix_type mass = make_mass_matrix(msh, fc, fb, di.face_degree());
                vector_type rhs = make_rhs(msh, fc, fb, dirichlet_bf, di.face_degree());
                ret.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
                ret.block( (num_faces + face_i)*fbs, 0, fbs, 1) = vector_type::Zero(fbs);
            }
            else
            {
                auto face_offset = priv::offset(msh, fc);
                auto face_SOL_offset = compress_table.at(face_offset)*fbs;
                ret.block(face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
                ret.block((num_faces + face_i)*fbs, 0, fbs, 1)
                    = solution.block(mult_offset + face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    // update_mat : assemble matrix according to the previous iteration solution
    template<typename Function>
    void
    update_mat(const Mesh&                     msh,
               const vector_type&              prev_sol,
               const Function&                 dirichlet_bf)
    {
        // clear RHS
        RHS = vector_type::Zero(system_size);

        // assemble all local contributions for Laplacian part
        for (auto& cl : msh)
        {
            auto solF = get_solF(msh, cl, prev_sol, dirichlet_bf);

            auto cell_offset = offset(msh, cl);
            assemble_contrib(msh, cl, loc_LHS.at(cell_offset), loc_RHS.at(cell_offset),solF, dirichlet_bf);
        }

        // assemble constraints
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        auto mult_offset = fbs * num_other_faces;

        for(size_t i = 0; i < mult_offset; i++)
        {
            auto sol_u    = prev_sol(i);
            auto sol_mult = prev_sol(mult_offset + i);

            if(sol_u <= 0.0 && sol_mult >= 0)
                triplets.push_back( Triplet<T>(mult_offset + i, i, 1.0) );
            else if(sol_u >= 0.0 && sol_mult <= 0)
                triplets.push_back( Triplet<T>(mult_offset + i, mult_offset + i, 1.0) );
            else if(sol_u * sol_mult > 0.0)
            {
                // this case is considered because of the error made during SC
                if(sol_u < sol_mult)
                    triplets.push_back( Triplet<T>(mult_offset + i, i, 1.0) );
                else
                    triplets.push_back( Triplet<T>(mult_offset + i, mult_offset + i, 1.0) );
            }
            else
                throw std::logic_error("we should not arrive here !!");
        }

        // identity block
        for(size_t i = 0; i < mult_offset; i++)
        {
            triplets.push_back( Triplet<T>(i, mult_offset + i, -1.0) );
        }

        // end assembly
        finalize();
    }

    template<typename Function>
    bool
    stop(const Mesh&         msh,
         const vector_type&  sol, const Function& dirichlet_bf)
    {
        T TOL = 1e-16;

        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        auto mult_offset = fbs * num_other_faces;

        // test the faces
        for(size_t i = 0; i < mult_offset; i++)
        {
            auto sol_u    = sol(i);
            auto sol_mult = sol(mult_offset + i);

            if(sol_u < -TOL || sol_mult < -TOL)
                return false;
        }

        // test the cells
        for (auto& cl : msh)
        {
            auto cell_offset = offset(msh, cl);
            matrix_type D = matrix_type::Zero(cbs, 2*cbs);
            for(size_t i = 0; i < cbs; i++)
            {
                auto OFF = cell_offset * cbs;
                if( active_constr.at(OFF + i) )
                    D(i,i) = 1.0;
                else
                    D(i,cbs + i) = 1.0;
            }

            auto solF = get_solF(msh, cl, sol, dirichlet_bf);

            auto loc_sol = membrane_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

            const auto fcs = faces(msh, cl);
            vector_type solT = loc_sol.head(cbs);
            vector_type multT = loc_sol.block(cbs + fcs.size() * fbs, 0, cbs, 1);

            for(size_t i = 0; i < cbs; i++)
            {
                auto sol_u    = solT(i);
                auto sol_mult = multT(i);

                if(sol_u < -TOL || sol_mult < -TOL)
                    return false;
            }
        }

        return true;
    }



    template<typename Function>
    vector_type
    take_u(const Mesh& msh, const typename Mesh::cell_type& cl,
    const vector_type& solution, const Function& dirichlet_bf)
    {
        auto solF = get_solF(msh, cl, solution, dirichlet_bf);

        auto cell_offset        = offset(msh, cl);
        auto celdeg = di.cell_degree();
        auto cbs = scalar_basis_size(celdeg, Mesh::dimension);

        matrix_type D = matrix_type::Zero(cbs, 2*cbs);

        for(size_t i = 0; i < cbs; i++)
        {
            auto OFF = cell_offset * cbs;
            if( active_constr.at(OFF + i) )
                D(i,i) = 1.0;
            else
                D(i,cbs + i) = 1.0;
        }

        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto num_faces = howmany_faces(msh, cl);

        vector_type full_sol = membrane_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

        return full_sol.head(cbs + num_faces * fbs);
    }

    template<typename Function>
    vector_type
    take_mult(const Mesh& msh, const typename Mesh::cell_type& cl,
              const vector_type& solution, const Function& dirichlet_bf)
    {
        auto solF = get_solF(msh, cl, solution, dirichlet_bf);

        auto cell_offset        = offset(msh, cl);
        auto celdeg = di.cell_degree();
        auto cbs = scalar_basis_size(celdeg, Mesh::dimension);

        matrix_type D = matrix_type::Zero(cbs, 2*cbs);

        for(size_t i = 0; i < cbs; i++)
        {
            auto OFF = cell_offset * cbs;
            if( active_constr.at(OFF + i) )
                D(i,i) = 1.0;
            else
                D(i,cbs + i) = 1.0;
        }

        auto full_sol = membrane_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto num_faces = howmany_faces(msh, cl);

        return full_sol.tail(cbs + num_faces * fbs);
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
auto make_condensed_assembler_Lag(const Mesh& msh, hho_degree_info hdi)
{
    return membrane_condensed_assembler<Mesh>(msh, hdi);
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
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 8.0 * R2 - 16.0 * r2;
        else
            return - 8.0 * R2;
    };
    auto sol_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return (r2 - R2) * (r2 - R2);
        else
            return 0.0;
    };

    auto assembler_sc = make_condensed_assembler_Lag(msh, hdi);
    auto assembler = make_assembler_Lag(msh, hdi);

    bool scond = true; // static condensation

    for (auto& cl : msh)
    {
        auto cb = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
        auto gr     = make_vector_hho_gradrec_Lag(msh, cl, hdi);
        auto stab   = make_scalar_hdg_stabilization_Lag(msh, cl, hdi);
        auto rhs    = make_rhs(msh, cl, cb, rhs_fun);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = gr.second + stab;
        if(scond)
            assembler_sc.set_loc_mat(msh, cl, A, rhs);
        else
        {
            assembler.set_loc_mat(msh, cl, A, rhs);
        }
    }

    size_t systsz, nnz;
    dynamic_vector<T> sol;
    size_t Newton_iter = 0;
    bool stop_loop = false;

    // Newton loop
    while(!stop_loop)
    {
        if(scond)
        {
            if(Newton_iter == 0)
                assembler_sc.init(msh, sol_fun);
            else
                assembler_sc.update_mat(msh, sol, sol_fun);
            systsz = assembler_sc.LHS.rows();
        }
        else
        {
            if(Newton_iter == 0)
                assembler.init(msh, sol_fun);
            else
                assembler.update_mat(msh, sol, sol_fun);
            systsz = assembler.LHS.rows();
        }
        sol = dynamic_vector<T>::Zero(systsz);

        disk::solvers::pardiso_params<T> pparams;
        pparams.report_factorization_Mflops = false;

        if(scond)
            mkl_pardiso(pparams, assembler_sc.LHS, assembler_sc.RHS, sol);
        else
            mkl_pardiso(pparams, assembler.LHS, assembler.RHS, sol);

        Newton_iter++;
        std::cout << blue << "end Newton iter nb " << Newton_iter << std::endl;

        if(scond)
            stop_loop = assembler_sc.stop(msh, sol, sol_fun);
        else
            stop_loop = assembler.stop(msh, sol);
    } // Newton loop

    std::cout << "Start post-process" << std::endl;

    T error = 0.0;

    postprocess_output<T>  postoutput;

    auto uT_gp  = std::make_shared< gnuplot_output_object<T> >("uT.dat");
    auto multT_gp  = std::make_shared< gnuplot_output_object<T> >("multT.dat");

    auto uF_gp  = std::make_shared< gnuplot_output_object<T> >("uF.dat");
    auto multF_gp  = std::make_shared< gnuplot_output_object<T> >("multF.dat");

    for (auto& cl : msh)
    {
        auto cb     = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();

        Eigen::Matrix<T, Eigen::Dynamic, 1> realsol = project_function(msh, cl, cb, sol_fun, 2);
        Eigen::Matrix<T, Eigen::Dynamic, 1> fullsol, mult_sol;
        auto gr     = make_vector_hho_gradrec_Lag(msh, cl, hdi);
        auto stab   = make_scalar_hdg_stabilization_Lag(msh, cl, hdi);
        auto rhs    = make_rhs(msh, cl, cb, rhs_fun);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = gr.second + stab;

        if(scond)
            fullsol = assembler_sc.take_u(msh, cl, sol, sol_fun);
        else
            fullsol = assembler.take_u(msh, cl, sol, sol_fun);

        auto diff = realsol - fullsol.head( cb.size() );
        error += diff.dot(A.block(0,0,cbs,cbs) * diff);

        auto uT = fullsol.head( cb.size() );

        if(scond)
            mult_sol = assembler_sc.take_mult(msh, cl, sol, sol_fun);
        else
            mult_sol = assembler.take_mult(msh, cl, sol);

        auto multT = mult_sol.head( cb.size() );

        // gnuplot output for cells
        auto pts = points(msh, cl);
        for(size_t i=0; i < pts.size(); i++)
        {
            T sol_uT = uT.dot( cb.eval_functions( pts[i] ) );
            uT_gp->add_data( pts[i], sol_uT );
            T sol_multT = multT.dot( cb.eval_functions(pts[i]) );
            multT_gp->add_data( pts[i], sol_multT );
        }

        // gnuplot output for faces
        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto fcs = faces(msh, cl);
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc = fcs[face_i];
            auto face_sol = fullsol.block(cbs+face_i*fbs, 0, fbs, 1);
            auto face_mult = mult_sol.block(cbs+face_i*fbs, 0, fbs, 1);

            const auto fb = make_scalar_Lagrange_basis(msh, fc, hdi.face_degree());
            auto barF = barycenter(msh, fc);

            T solbarF = fb.eval_functions(barF).dot(face_sol);
            uF_gp->add_data( barF, solbarF );

            T multbarF = fb.eval_functions(barF).dot(face_mult);
            multF_gp->add_data( barF, multbarF );
        }
    }

    postoutput.add_object(uT_gp);
    postoutput.add_object(multT_gp);
    postoutput.add_object(uF_gp);
    postoutput.add_object(multF_gp);
    postoutput.write();

    std::cout << yellow << "ended run : error is " << std::sqrt(error) << std::endl;
    
    return std::sqrt(error);
}

using namespace Eigen;

int main(void)
{
    using T = double;

    // degree of the polynomials on the faces
    size_t degree = 0;
    
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
            run_membranes_solver(msh, degree);
        }

    }
    else
    {
        mesh_type msh;
        disk::fvca5_mesh_loader<T, 2> loader;
        std::string mesh_filename = "../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1";
        if (!loader.read_mesh(mesh_filename) )
        {
            std::cout << "Problem loading mesh." << std::endl;
        }
        loader.populate_mesh(msh);
        run_membranes_solver(msh, degree);
    }
    return 0;
}
