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

#include "../tests/common.hpp"
#include "output/silo.hpp"

/***************************************************************************/
/* RHS definition */
template<typename Mesh>
struct rhs_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct rhs_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        auto sin_px = std::sin(M_PI * pt.x());
        auto sin_py = std::sin(M_PI * pt.y());
        return 2.0 * M_PI * M_PI * sin_px * sin_py;
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct rhs_functor< Mesh<T, 3, Storage> >
{
    typedef Mesh<T,3,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        auto sin_px = std::sin(M_PI * pt.x());
        auto sin_py = std::sin(M_PI * pt.y());
        auto sin_pz = std::sin(M_PI * pt.z());
        return 3.0 * M_PI * M_PI * sin_px * sin_py * sin_pz;
    }
};

template<typename Mesh>
auto make_rhs_function(const Mesh& msh)
{
    return rhs_functor<Mesh>();
}

/***************************************************************************/
/* Expected solution definition */
template<typename Mesh>
struct solution_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct solution_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        auto sin_px = std::sin(M_PI * pt.x());
        auto sin_py = std::sin(M_PI * pt.y());
        return sin_px * sin_py;
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct solution_functor< Mesh<T, 3, Storage> >
{
    typedef Mesh<T,3,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        auto sin_px = std::sin(M_PI * pt.x());
        auto sin_py = std::sin(M_PI * pt.y());
        auto sin_pz = std::sin(M_PI * pt.z());
        return sin_px * sin_py * sin_pz;
    }
};

template<typename Mesh>
auto make_solution_function(const Mesh& msh)
{
    return solution_functor<Mesh>();
}

/***************************************************************************/
/* gradients of the expected solution */
template<typename Mesh>
struct grad_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct grad_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    auto operator()(const point_type& pt) const
    {
        Matrix<T, 1, 2> ret;
        auto sin_px = std::sin(M_PI * pt.x());
        auto sin_py = std::sin(M_PI * pt.y());
        auto cos_px = std::cos(M_PI * pt.x());
        auto cos_py = std::cos(M_PI * pt.y());

        ret(0) = M_PI * cos_px * sin_py;
        ret(1) = M_PI * sin_px * cos_py;
        return ret;
    }
};

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct grad_functor< Mesh<T, 3, Storage> >
{
    typedef Mesh<T,3,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    auto operator()(const point_type& pt) const
    {
        Matrix<T, 1, 3> ret;
        auto sin_px = std::sin(M_PI * pt.x());
        auto sin_py = std::sin(M_PI * pt.y());
        auto sin_pz = std::sin(M_PI * pt.z());
        auto cos_px = std::cos(M_PI * pt.x());
        auto cos_py = std::cos(M_PI * pt.y());
        auto cos_pz = std::cos(M_PI * pt.z());

        ret(0) = M_PI * cos_px * sin_py * sin_pz;
        ret(1) = M_PI * sin_px * cos_py * sin_pz;
        ret(2) = M_PI * sin_px * sin_py * cos_pz;

        return ret;
    }
};

template<typename Mesh>
auto make_grad_function(const Mesh& msh)
{
    return grad_functor<Mesh>();
}

/***************************************************************************/
/* domain varpi function */

template<typename Mesh>
struct varpi_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct varpi_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        scalar_type ret;

        if( (pt.x() >= 0.0) && (pt.x() <= 0.875) && (pt.y() >= 0.125) && (pt.y() <= 0.875) )
            ret = 0.0;
        else
            ret = 1.0;

        return ret;
    }
};

// template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
// struct varpi_functor< Mesh<T, 3, Storage> >
// {
//     typedef Mesh<T,3,Storage>               mesh_type;
//     typedef typename mesh_type::coordinate_type scalar_type;
//     typedef typename mesh_type::point_type  point_type;

//     scalar_type operator()(const point_type& pt) const
//     {
//         scalar_type ret;
//         return ret;
//     }
// };

template<typename Mesh>
auto make_varpi_function(const Mesh& msh)
{
    return varpi_functor<Mesh>();
}


/***************************************************************************/
/* domain B function */

template<typename Mesh>
struct B_functor;

template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
struct B_functor< Mesh<T, 2, Storage> >
{
    typedef Mesh<T,2,Storage>               mesh_type;
    typedef typename mesh_type::coordinate_type scalar_type;
    typedef typename mesh_type::point_type  point_type;

    scalar_type operator()(const point_type& pt) const
    {
        scalar_type ret;

        if( (pt.x() >= 0.0) && (pt.x() <= 0.125) && (pt.y() >= 0.125) && (pt.y() <= 0.875) )
            ret = 0.0;
        else
            ret = 1.0;

        return ret;
    }
};

// template<template<typename, size_t, typename> class Mesh, typename T, typename Storage>
// struct B_functor< Mesh<T, 3, Storage> >
// {
//     typedef Mesh<T,3,Storage>               mesh_type;
//     typedef typename mesh_type::coordinate_type scalar_type;
//     typedef typename mesh_type::point_type  point_type;

//     scalar_type operator()(const point_type& pt) const
//     {
//         scalar_type ret;

//         if( (pt.x() >= 0.0) && (pt.x() <= 0.125) && (pt.y() >= 0.125) && (pt.y() <= 0.875) )
//             ret = 0.0;
//         else
//             ret = 1.0;

//         return ret;
//     }
// };

template<typename Mesh>
auto make_B_function(const Mesh& msh)
{
    return B_functor<Mesh>();
}

////////////////////////////////////////////////////////////////////////////
////////////////////////   ASSEMBLERS  /////////////////////////////////////
////////////////////////////////////////////////////////////////////////////

using namespace disk;


template<typename Mesh>
class helmholtz_assembler
{
public:

    using T = typename Mesh::coordinate_type;

    std::vector<size_t>     compress_table;
    std::vector<size_t>     expand_table;
    hho_degree_info         di;
    std::vector<Triplet<T>> triplets;

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

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    SparseMatrix<T> LHS;
    vector_type     RHS;

    helmholtz_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi)
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

        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);
        system_size    = cbs * num_cells + fbs * num_other_faces;

        LHS = SparseMatrix<T>(system_size, system_size);
        RHS = vector_type::Zero(system_size);
    }


    template<typename Function>
    void
    assemble(const Mesh&                     msh,
             const typename Mesh::cell_type& cl,
             const matrix_type&              lhs,
             const vector_type&              rhs,
             const Function&                 dirichlet_bf)
    {
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
                auto fb = make_scalar_monomial_basis(msh, fc, di.face_degree());
                dirichlet_data.block(cbs + face_i * fbs, 0, fbs, 1) =
                    project_function(msh, fc, fb, dirichlet_bf, di.face_degree());
            }
        }

        // assemble LHS
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

        // assemble RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(asm_map[i]) += rhs(i);
        }

    } // assemble()

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
                auto fb = make_scalar_monomial_basis(msh, fc, di.face_degree());

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
auto make_helmholtz_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return helmholtz_assembler<Mesh>(msh, hdi);
}

//////////////////  condensed assembler
template<typename Mesh>
class condensed_helmholtz_assembler : public helmholtz_assembler<Mesh>
{
    using T = typename Mesh::coordinate_type;

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

    std::vector< matrix_type > loc_LHS;
    std::vector< vector_type > loc_RHS;

    condensed_helmholtz_assembler(const Mesh& msh, hho_degree_info hdi)
        : helmholtz_assembler<Mesh>(msh, hdi)
    {
        auto num_cells = msh.cells_size();
        loc_LHS.resize(num_cells);
        loc_RHS.resize(num_cells);

        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);
        this->system_size    = fbs * this->num_other_faces;

        this->LHS = SparseMatrix<T>(this->system_size, this->system_size);
        this->RHS = vector_type::Zero(this->system_size);
    }


    template<typename Function>
    void
    assemble(const Mesh&                     msh,
             const typename Mesh::cell_type& cl,
             const matrix_type&              lhs,
             const vector_type&              rhs,
             const Function&                 dirichlet_bf)
    {
        // store local LHS and RHS
        auto cell_offset = offset(msh, cl);
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS.at( cell_offset ) = rhs;


        auto sc = make_scalar_static_condensation(msh, cl, this->di, lhs, rhs);
        auto lhs_sc = sc.first;
        auto rhs_sc = sc.second;

        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return msh.is_boundary(fc);
        };

        const auto fbs = scalar_basis_size(this->di.face_degree(), Mesh::dimension-1);
        const auto fcs = faces(msh, cl);

        std::vector<assembly_index> asm_map;
        asm_map.reserve(fcs.size() * fbs);

        vector_type dirichlet_data = vector_type::Zero(fcs.size()*fbs);

        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc              = fcs[face_i];
            const auto face_offset     = priv::offset(msh, fc);
            const auto face_LHS_offset = this->compress_table.at(face_offset) * fbs;

            const bool dirichlet = is_dirichlet(fc);

            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );

            if (dirichlet)
            {
                auto fb = make_scalar_monomial_basis(msh, fc, this->di.face_degree());
                dirichlet_data.block(face_i * fbs, 0, fbs, 1) =
                    project_function(msh, fc, fb, dirichlet_bf, this->di.face_degree());
            }
        }

        // assemble LHS
        for (size_t i = 0; i < lhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    this->triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs_sc(i,j)) );
                else
                    this->RHS(asm_map[i]) -= lhs_sc(i,j) * dirichlet_data(j);
            }
        }

        // assemble RHS
        for (size_t i = 0; i < rhs_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            this->RHS(asm_map[i]) += rhs_sc(i);
        }

    } // assemble()

    template<typename Function>
    vector_type
    take_u(const Mesh& msh, const typename Mesh::cell_type& cl,
    const vector_type& solution, const Function& dirichlet_bf)
    {
        auto facdeg = this->di.face_degree();
        auto fbs = scalar_basis_size(this->di.face_degree(), Mesh::dimension-1);
        auto fcs = faces(msh, cl);

        auto num_faces = fcs.size();

        vector_type face_sol = vector_type::Zero(num_faces*fbs);

        for (size_t face_i = 0; face_i < num_faces; face_i++)
        {
            auto fc = fcs[face_i];

            auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
                return msh.is_boundary(fc);
            };

            bool dirichlet = is_dirichlet(fc);

            if (dirichlet)
            {
                auto fb = make_scalar_monomial_basis(msh, fc, this->di.face_degree());

                matrix_type mass = make_mass_matrix(msh, fc, fb, this->di.face_degree());
                vector_type rhs = make_rhs(msh, fc, fb, dirichlet_bf, this->di.face_degree());
                face_sol.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs);
            }
            else
            {
                auto face_offset = priv::offset(msh, fc);
                auto face_SOL_offset = this->compress_table.at(face_offset)*fbs;
                face_sol.block(face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
            }
        }

        auto cell_offset = offset(msh, cl);

        return make_scalar_static_decondensation(msh, cl, this->di, loc_LHS[cell_offset], loc_RHS[cell_offset], face_sol);
    }

    void finalize(void)
    {
        this->LHS.setFromTriplets( this->triplets.begin(), this->triplets.end() );
        this->triplets.clear();

        dump_sparse_matrix(this->LHS, "diff.dat");
    }

    size_t num_assembled_faces() const
    {
        return this->num_other_faces;
    }

};


template<typename Mesh>
auto make_condensed_helmholtz_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return condensed_helmholtz_assembler<Mesh>(msh, hdi);
}




/**********   Helmholtz UC  assemblers   *************/

template<typename Mesh>
class helmholtz_UC_assembler
{
public:

    using T = typename Mesh::coordinate_type;

    std::vector<size_t>     compress_table;
    std::vector<size_t>     expand_table;
    hho_degree_info         di;
    std::vector<Triplet<T>> triplets;

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

    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    SparseMatrix<T> LHS;
    vector_type     RHS;

    helmholtz_UC_assembler(const Mesh& msh, hho_degree_info hdi)
        : di(hdi)
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

        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(hdi.cell_degree(), Mesh::dimension);

        system_size    = 2 * cbs * num_cells + fbs * (num_all_faces + num_other_faces) ;

        LHS = SparseMatrix<T>(system_size, system_size);
        RHS = vector_type::Zero(system_size);
    }


    template<typename Function>
    void
    assemble(const Mesh&                     msh,
             const typename Mesh::cell_type& cl,
             const matrix_type&              lhs,
             const vector_type&              rhs,
             const Function&                 dirichlet_bf)
    {
        auto is_dirichlet = [&](const typename Mesh::face_type& fc) -> bool {
            return msh.is_boundary(fc);
        };

        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        const auto fcs = faces(msh, cl);

        std::vector<assembly_index> asm_map;
        asm_map.reserve(2*(cbs + fcs.size() * fbs));

        auto cell_offset        = offset(msh, cl);
        auto cell_LHS_offset    = cell_offset * cbs;

        // dofs for uT
        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(cell_LHS_offset+i, true) );

        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc              = fcs[face_i];
            const auto face_offset     = priv::offset(msh, fc);
            const auto face_LHS_offset = msh.cells_size() * cbs + face_offset * fbs;

            // dofs for uF
            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, true) );
        }

        const auto z_offset = msh.cells_size() * cbs + num_all_faces * fbs;

        // dofs for zT
        for (size_t i = 0; i < cbs; i++)
            asm_map.push_back( assembly_index(z_offset + cell_LHS_offset+i, true) );


        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc              = fcs[face_i];
            const auto face_offset     = priv::offset(msh, fc);
            const auto face_LHS_offset = z_offset + msh.cells_size() * cbs
                + compress_table.at(face_offset) * fbs;

            const bool dirichlet = is_dirichlet(fc);

            // dofs for zF
            for (size_t i = 0; i < fbs; i++)
                asm_map.push_back( assembly_index(face_LHS_offset+i, !dirichlet) );
        }

        // assemble LHS
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

        // assemble RHS
        for (size_t i = 0; i < rhs.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(asm_map[i]) += rhs(i);
        }

    } // assemble()

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
            auto face_offset = priv::offset(msh, fc);
            auto face_SOL_offset = msh.cells_size() * cbs + face_offset*fbs;
            ret.block(cbs + face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
        }

        return ret;
    }


    template<typename Function>
    vector_type
    take_z(const Mesh& msh, const typename Mesh::cell_type& cl,
    const vector_type& solution, const Function& dirichlet_bf)
    {
        auto celdeg = di.cell_degree();
        auto facdeg = di.face_degree();
        auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto fcs = faces(msh, cl);

        auto num_faces = fcs.size();

        auto z_offset = msh.cells_size() * cbs + num_all_faces * fbs;

        auto cell_offset        = offset(msh, cl);
        auto cell_SOL_offset    = z_offset + cell_offset * cbs;

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
                ret.block(cbs + face_i*fbs, 0, fbs, 1) = vector_type::Zero(fbs);
            }
            else
            {
                auto face_offset = priv::offset(msh, fc);
                auto face_SOL_offset = z_offset + msh.cells_size() * cbs
                    + compress_table.at(face_offset)*fbs;
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
        return num_all_faces;
    }

};


template<typename Mesh>
auto make_helmholtz_UC_assembler(const Mesh& msh, hho_degree_info hdi)
{
    return helmholtz_UC_assembler<Mesh>(msh, hdi);
}

/////////////////////////////////////////////////////////////////
//////////////////  ASSEMBLY  ROUTINES //////////////////////////
/////////////////////////////////////////////////////////////////


/****************************************************************/
/************          Laplacian  terms          ****************/

template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_DGH_laplacian(const Mesh&                     msh,
                   const typename Mesh::cell_type& cl,
                   const hho_degree_info&          di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;
    typedef Matrix<T, Dynamic, 1>       vector_type;

    const auto celdeg  = di.cell_degree();
    const auto facdeg  = di.face_degree();
    const auto graddeg = di.grad_degree();

    const auto cb = make_scalar_monomial_basis(msh, cl, celdeg);
    const auto gb = make_vector_monomial_basis(msh, cl, graddeg);

    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);
    const auto gbs = gb.size();

    const auto num_faces = howmany_faces(msh, cl);

    const matrix_type gr_lhs = make_mass_matrix(msh, cl, gb);
    matrix_type       gr_rhs = matrix_type::Zero(gbs, cbs + num_faces * fbs);

    size_t loc_size = cbs + num_faces * fbs;
    matrix_type ret = matrix_type::Zero(loc_size, loc_size);

    int intdeg = 2*celdeg - 2;
    auto int_deg = std::max(intdeg , 0);
    const auto qps = integrate(msh, cl, int_deg);
    for (auto& qp : qps)
    {
        const auto grad_phi = cb.eval_gradients(qp.point());

        ret.block(0, 0, cbs, cbs) += qp.weight() * grad_phi * grad_phi.transpose();
    }

    const auto fcs = faces(msh, cl);
    for (size_t i = 0; i < fcs.size(); i++)
    {
        const auto fc = fcs[i];
        const auto n  = normal(msh, cl, fc);
        const auto fb = make_scalar_monomial_basis(msh, fc, facdeg);

        int intdeg = celdeg + facdeg - 1;
        auto int_deg = std::max(intdeg,0);
        const auto qps_f = integrate(msh, fc, int_deg);
        for (auto& qp : qps_f)
        {
            const vector_type f_phi  = fb.eval_functions(qp.point());
            const auto c_phi         = cb.eval_functions(qp.point());
            const auto grad_phi      = cb.eval_gradients(qp.point());
            const vector_type qp_g_phi_n = grad_phi * (qp.weight() * n);

            ret.block(0, cbs + i * fbs, cbs, fbs) += qp_g_phi_n *  f_phi.transpose();
            ret.block(0, 0, cbs, cbs) -= qp_g_phi_n * c_phi.transpose();

            ret.block(cbs + i * fbs, 0, fbs, cbs) += f_phi * qp_g_phi_n.transpose();
            ret.block(0, 0, cbs, cbs) -= c_phi * qp_g_phi_n.transpose();
        }
    }

    return ret;
}

/****************************************************************/
/**********         Stabilization  terms          ***************/

// we compute hT^{-1} (uT - uF , vT - vF)_F
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_scalar_fool_stabilization(const Mesh& msh, const typename Mesh::cell_type& cl,
                               const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();

    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);

    const auto num_faces = howmany_faces(msh, cl);
    const auto total_dofs = cbs + num_faces * fbs;

    matrix_type       ret = matrix_type::Zero(total_dofs, total_dofs);

    auto cb = make_scalar_monomial_basis(msh, cl, celdeg);
    const auto fcs = faces(msh, cl);


    for (size_t i = 0; i < num_faces; i++)
    {
        const auto fc = fcs[i];
        auto fb = make_scalar_monomial_basis(msh, fc, facdeg);

        // compute (uT - uF , vT - vF)_F
        const auto qps = integrate(msh, fc, 2*std::max(facdeg, celdeg));
        for (auto& qp : qps)
        {
            const auto c_phi = cb.eval_functions(qp.point());
            const auto f_phi = fb.eval_functions(qp.point());

            ret.block(0, 0, cbs, cbs) += qp.weight() * c_phi * c_phi.transpose();
            ret.block(0, cbs + i * fbs, cbs, fbs) -= qp.weight() * c_phi * f_phi.transpose();
            ret.block(cbs + i * fbs, 0, fbs, cbs) -= qp.weight() * f_phi * c_phi.transpose();
            ret.block(cbs + i * fbs, cbs + i * fbs, fbs, fbs)
                += qp.weight() * f_phi * f_phi.transpose();
        }
    }

    // scale with hT^{-1}
    const auto hT  = diameter(msh, cl);
    ret = (1.0/hT) * ret;

    return ret;
}



// we compute hT^{-1} (uT - uF , vT - vF)_F + hT^{2k} (uT , vT)_T
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_scalar_stabilization_UC(const Mesh& msh, const typename Mesh::cell_type& cl,
                             const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();

    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);

    const auto num_faces = howmany_faces(msh, cl);
    const auto total_dofs = cbs + num_faces * fbs;

    matrix_type       ret = matrix_type::Zero(total_dofs, total_dofs);

    auto cb = make_scalar_monomial_basis(msh, cl, celdeg);

    // compute (uT - uF , vT - vF)_F
    const auto fcs = faces(msh, cl);
    for (size_t i = 0; i < num_faces; i++)
    {
        const auto fc = fcs[i];
        auto fb = make_scalar_monomial_basis(msh, fc, facdeg);

        const auto qps_F = integrate(msh, fc, 2*std::max(facdeg, celdeg));
        for (auto& qp : qps_F)
        {
            const auto c_phi = cb.eval_functions(qp.point());
            const auto f_phi = fb.eval_functions(qp.point());

            ret.block(0, 0, cbs, cbs) += qp.weight() * c_phi * c_phi.transpose();
            ret.block(0, cbs + i * fbs, cbs, fbs) -= qp.weight() * c_phi * f_phi.transpose();
            ret.block(cbs + i * fbs, 0, fbs, cbs) -= qp.weight() * f_phi * c_phi.transpose();
            ret.block(cbs + i * fbs, cbs + i * fbs, fbs, fbs)
                += qp.weight() * f_phi * f_phi.transpose();
        }
    }

    // scale with hT^{-1}
    const auto hT  = diameter(msh, cl);
    ret = (1.0/hT) * ret;

    // compute hT^{2k}
    const auto hT2  = hT * hT;
    auto hT2k = 1.0;
    for(size_t i = 0; i < celdeg; i++)
    {
        hT2k = hT2k * hT2;
    }

    // compute hT^{2k} (uT,vT)_T
    const auto qps_T = integrate(msh, cl, 2*celdeg);
    for (auto& qp : qps_T)
    {
        const auto phi = cb.eval_functions(qp.point());

        ret.block(0, 0, cbs, cbs) += qp.weight() * hT2k * phi * phi.transpose();
    }

    return ret;
}


// we compute hT^{-1} (uT - uF , vT - vF)_F + hT^{-1} (uT , vT)_{\partial \Omega} 
//            + (\GRAD uT , \GRAD vT)_T
template<typename Mesh>
Matrix<typename Mesh::coordinate_type, Dynamic, Dynamic>
make_scalar_stabilization_UC_star(const Mesh& msh, const typename Mesh::cell_type& cl,
                                  const hho_degree_info& di)
{
    using T = typename Mesh::coordinate_type;
    typedef Matrix<T, Dynamic, Dynamic> matrix_type;

    const auto celdeg = di.cell_degree();
    const auto facdeg = di.face_degree();

    const auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
    const auto fbs = scalar_basis_size(facdeg, Mesh::dimension - 1);

    const auto num_faces = howmany_faces(msh, cl);
    const auto total_dofs = cbs + num_faces * fbs;

    matrix_type       ret = matrix_type::Zero(total_dofs, total_dofs);

    auto cb = make_scalar_monomial_basis(msh, cl, celdeg);

    const auto fcs = faces(msh, cl);
    for (size_t i = 0; i < num_faces; i++)
    {
        const auto fc = fcs[i];
        auto fb = make_scalar_monomial_basis(msh, fc, facdeg);

        // compute (uT - uF , vT - vF)_F
        const auto qps_F = integrate(msh, fc, 2*std::max(facdeg, celdeg));
        for (auto& qp : qps_F)
        {
            const auto c_phi = cb.eval_functions(qp.point());
            const auto f_phi = fb.eval_functions(qp.point());

            ret.block(0, 0, cbs, cbs) += qp.weight() * c_phi * c_phi.transpose();
            ret.block(0, cbs + i * fbs, cbs, fbs) -= qp.weight() * c_phi * f_phi.transpose();
            ret.block(cbs + i * fbs, 0, fbs, cbs) -= qp.weight() * f_phi * c_phi.transpose();
            ret.block(cbs + i * fbs, cbs + i * fbs, fbs, fbs)
                += qp.weight() * f_phi * f_phi.transpose();
        }

        // compute (uT,vT)_{\partial \Omega}
        if( msh.is_boundary(fc) )
        {
            for (auto& qp : qps_F)
            {
                const auto c_phi = cb.eval_functions(qp.point());
                const auto f_phi = fb.eval_functions(qp.point());

                ret.block(0, 0, cbs, cbs) += qp.weight() * c_phi * c_phi.transpose();
            }
        }
    }

    // scale with hT^{-1}
    const auto hT  = diameter(msh, cl);
    ret = (1.0/hT) * ret;

    // compute (\GRAD uT , \GRAD vT)_T
    const auto qps_T = integrate(msh, cl, 2*celdeg);
    for (auto& qp : qps_T)
    {
        const auto grad_phi = cb.eval_gradients(qp.point());

        ret.block(0, 0, cbs, cbs) += qp.weight() * grad_phi * grad_phi.transpose();
    }

    return ret;
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


/////////////////////////////////////////////////////////////////////////
/////////////////////////////   MAIN PART   /////////////////////////////
/////////////////////////////////////////////////////////////////////////


template<typename Mesh>
typename Mesh::coordinate_type
run_Helmholtz(const Mesh& msh, size_t degree)
{
    using T = typename Mesh::coordinate_type;

    hho_degree_info hdi(degree, degree);

    auto rhs_fun = make_rhs_function(msh);
    auto sol_fun = make_solution_function(msh);
    auto sol_grad = make_grad_function(msh);


    auto varpi_fun = make_varpi_function(msh);
    auto B_fun = make_B_function(msh);

    auto assembler = make_helmholtz_assembler(msh, hdi);
    auto assembler_sc = make_condensed_helmholtz_assembler(msh, hdi);


    bool scond = true;

    for (auto& cl : msh)
    {
        auto cb = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
        auto lap    = make_DGH_laplacian(msh, cl, hdi);
        auto stab   = make_scalar_fool_stabilization(msh, cl, hdi);
        auto rhs    = make_rhs(msh, cl, cb, rhs_fun);
        T gamma = 5.0 * (degree+1) * (degree+2);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = lap + gamma * stab;
        if(scond)
            assembler_sc.assemble(msh, cl, A, rhs, sol_fun);
        else
            assembler.assemble(msh, cl, A, rhs, sol_fun);
    }

    if(scond)
        assembler_sc.finalize();
    else
        assembler.finalize();

    size_t systsz, nnz;

    if(scond)
    {
        systsz = assembler_sc.LHS.rows();
        nnz = assembler_sc.LHS.nonZeros();
    }
    else
    {
        systsz = assembler.LHS.rows();
        nnz = assembler.LHS.nonZeros();
    }

    // std::cout << "system size = " << systsz << std::endl;

    dynamic_vector<T> sol = dynamic_vector<T>::Zero(systsz);

    disk::solvers::pardiso_params<T> pparams;
    pparams.report_factorization_Mflops = false;
    if(scond)
        mkl_pardiso(pparams, assembler_sc.LHS, assembler_sc.RHS, sol);
    else
        mkl_pardiso(pparams, assembler.LHS, assembler.RHS, sol);

    ///////////////////////  POST PROCESS ///////////////////
    // std::cout << "Start post-process" << std::endl;
    // std::cout << "h = " << disk::average_diameter(msh) << std::endl;

    T u_H1_error = 0.0;
    T u_L2_error = 0.0;

    // postprocess_output<T>  postoutput;

    // auto uT_gp  = std::make_shared< gnuplot_output_object<T> >("uT.dat");
    // auto sol_gp  = std::make_shared< gnuplot_output_object<T> >("sol.dat");

    for (auto& cl : msh)
    {
        auto cb  = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();

        Eigen::Matrix<T, Eigen::Dynamic, 1> fullsol;

        if(scond)
        {
            fullsol = assembler_sc.take_u(msh, cl, sol, sol_fun);
        }
        else
        {
            fullsol = assembler.take_u(msh, cl, sol, sol_fun);
        }

        auto cell_dofs = fullsol.head( cbs );

        // errors
        const auto celdeg = hdi.cell_degree();
        const auto qps = integrate(msh, cl, 2*celdeg);
        for (auto& qp : qps)
        {
            const auto dim = Mesh::dimension;
            auto grad_ref = sol_grad( qp.point() );
            auto t_dphi = cb.eval_gradients( qp.point() );
            Matrix<T, 1, dim> grad = Matrix<T, 1, dim>::Zero();

            for (size_t i = 0; i < cbs; i++ )
                grad += cell_dofs(i) * t_dphi.block(i, 0, 1, dim);

            // H1-error
            u_H1_error += qp.weight() * (grad_ref - grad).dot(grad_ref - grad);

            // L2-error
            auto t_phi = cb.eval_functions( qp.point() );
            T v = cell_dofs.dot( t_phi );
            u_L2_error += qp.weight() * (sol_fun(qp.point()) - v) * (sol_fun(qp.point()) - v);
        }

        // // gnuplot output for cells
        // auto pts = points(msh, cl);
        // for(size_t i=0; i < pts.size(); i++)
        // {
        //     T sol_uT = cell_dofs.dot( cb.eval_functions( pts[i] ) );
        //     uT_gp->add_data( pts[i], sol_uT );
        //     sol_gp->add_data( pts[i], sol_fun(pts[i]) );
        // }
    }

    // postoutput.add_object(uT_gp);
    // postoutput.add_object(sol_gp);
    // postoutput.write();

    // std::cout << yellow << "ended run : H1-error is " << std::sqrt(u_H1_error) << std::endl;
    // std::cout << yellow << "            L2-error is " << std::sqrt(u_L2_error) << std::endl;
    // std::cout << nocolor;



    silo_database silo;
    silo.create("helmholtz.silo");
    silo.add_mesh(msh, "mesh");
    std::vector<T> cell_varpi, cell_B;

    for(auto& cl : msh)
    {
        cell_varpi.push_back( varpi_fun( barycenter(msh, cl) ) );
        cell_B.push_back( B_fun( barycenter(msh, cl) ) );
    }
    silo_zonal_variable<T> silo_varpi("varpi", cell_varpi);
    silo_zonal_variable<T> silo_B("B", cell_B);
    silo.add_variable("mesh", silo_varpi);
    silo.add_variable("mesh", silo_B);
    silo.close();

    return std::sqrt(u_H1_error);
}


template<typename Mesh>
struct test_functor
{
    /* Expect k+1 convergence (hho stabilization, energy norm) */
    typename Mesh::coordinate_type
    operator()(const Mesh& msh, size_t degree) const
    {
        return run_Helmholtz(msh, degree);
    }

    size_t
    expected_rate(size_t k)
    {
        return k;
    }
};


template<typename Mesh>
void
run_diffusion_solver(const Mesh& msh)
{
    run_Helmholtz(msh, 0);
}


#if 0
int main(void)
{
    tester<test_functor> tstr;
    tstr.run();
    return 0;
}
#endif


#if 1
int main(void)
{
    using T = double;

    // degree of the polynomials on the faces
    size_t degree = 1;
    
    typedef disk::generic_mesh<T, 2>  mesh_type;
    // typedef disk::simplicial_mesh<T, 3>  mesh_type;
    
    
    if(0)
    {
        std::vector<std::string> meshfiles;
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_1.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_2.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_3.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_5.typ1");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_6.typ1");
    

        for(size_t i=0; i < meshfiles.size(); i++)
        {
            mesh_type msh;
            disk::fvca5_mesh_loader<T, 2> loader;
            // disk::netgen_mesh_loader<T, 3> loader;
            if (!loader.read_mesh(meshfiles.at(i)) )
            {
                std::cout << "Problem loading mesh." << std::endl;
            }
            loader.populate_mesh(msh);
            run_Helmholtz(msh, degree);
        }

    }
    else
    {
        mesh_type msh;
        disk::fvca5_mesh_loader<T, 2> loader;
        // disk::netgen_mesh_loader<T, 3> loader;
        std::string mesh_filename = "../../../diskpp/meshes/2D_triangles/fvca5/mesh1_3.typ1";
        // std::string mesh_filename = "../../../diskpp/meshes/3D_tetras/netgen/cube4.mesh";
        if (!loader.read_mesh(mesh_filename) )
        {
            std::cout << "Problem loading mesh." << std::endl;
        }
        loader.populate_mesh(msh);
        run_Helmholtz(msh, degree);
    }

    std::cout << "\a" << std::endl;
    return 0;
}
#endif
