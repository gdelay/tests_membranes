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

#include "output/silo.hpp"

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
        if( degree > 4 )
            throw std::invalid_argument("degree > 4 not yet supported");
        basis_degree = degree;

        // store the vertices
        auto pts = points(msh, cl);
        assert( pts.size() == 3);
        vertices.push_back( pts[0] );
        vertices.push_back( pts[1] );
        vertices.push_back( pts[2] );
        // vertices = points(msh, cl);

        basis_size = scalar_basis_size(degree, 2);
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
        else if(basis_degree == 2)
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
        else if(basis_degree == 3)
        {
            auto bar_c = bar_coord(pt);
            // 0-2 : vertices
            for(size_t i = 0; i < 3; i++)
                ret[i] = 0.5*bar_c[i] * (3.0*bar_c[i] - 1.0) * (3.0*bar_c[i] - 2.0);

            // 3-8 : face-points (112,122,002,022,001,011)
            ret[3] = 4.5*bar_c[1]*bar_c[2]*(3.0*bar_c[1]-1.0);
            ret[4] = 4.5*bar_c[1]*bar_c[2]*(3.0*bar_c[2]-1.0);
            ret[5] = 4.5*bar_c[0]*bar_c[2]*(3.0*bar_c[0]-1.0);
            ret[6] = 4.5*bar_c[0]*bar_c[2]*(3.0*bar_c[2]-1.0);
            ret[7] = 4.5*bar_c[0]*bar_c[1]*(3.0*bar_c[0]-1.0);
            ret[8] = 4.5*bar_c[0]*bar_c[1]*(3.0*bar_c[1]-1.0);

            // 9 : center of mass
            ret[9] = 27.0*bar_c[0]*bar_c[1]*bar_c[2];
        }
        else // degree == 4
        {
            auto bar_c = bar_coord(pt);
            // 0-2 : vertices
            for(size_t i = 0; i < 3; i++)
                ret[i] = (1.0/6.0)*bar_c[i] * (4.0*bar_c[i] - 1.0)
                    * (4.0*bar_c[i] - 2.0) * (4.0*bar_c[i] - 3.0);

            // 3-5 : mid-points (12,02,01)
            ret[3] = 4.0*bar_c[1]*bar_c[2]*(4.0*bar_c[1]-1.0)*(4.0*bar_c[2]-1.0);
            ret[4] = 4.0*bar_c[0]*bar_c[2]*(4.0*bar_c[0]-1.0)*(4.0*bar_c[2]-1.0);
            ret[5] = 4.0*bar_c[0]*bar_c[1]*(4.0*bar_c[0]-1.0)*(4.0*bar_c[1]-1.0);

            // 6-11 : face-points (1112,1222,0002,0222,0001,0111)
            ret[6] = (8.0/3.0)*bar_c[1]*bar_c[2]*(4.0*bar_c[1]-1.0)*(4.0*bar_c[1]-2.0);
            ret[7] = (8.0/3.0)*bar_c[2]*bar_c[1]*(4.0*bar_c[2]-1.0)*(4.0*bar_c[2]-2.0);
            ret[8] = (8.0/3.0)*bar_c[0]*bar_c[2]*(4.0*bar_c[0]-1.0)*(4.0*bar_c[0]-2.0);
            ret[9] = (8.0/3.0)*bar_c[2]*bar_c[0]*(4.0*bar_c[2]-1.0)*(4.0*bar_c[2]-2.0);
            ret[10] = (8.0/3.0)*bar_c[0]*bar_c[1]*(4.0*bar_c[0]-1.0)*(4.0*bar_c[0]-2.0);
            ret[11] = (8.0/3.0)*bar_c[1]*bar_c[0]*(4.0*bar_c[1]-1.0)*(4.0*bar_c[1]-2.0);

            // 12-14 : others (0012,0112,0122)
            ret[12] = 32.0 * bar_c[0]*bar_c[1]*bar_c[2]*(4.0*bar_c[0]-1.0);
            ret[13] = 32.0 * bar_c[0]*bar_c[1]*bar_c[2]*(4.0*bar_c[1]-1.0);
            ret[14] = 32.0 * bar_c[0]*bar_c[1]*bar_c[2]*(4.0*bar_c[2]-1.0);
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
        else if(basis_degree == 2)
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
        else if(basis_degree == 3)
        {
            auto bar_c = bar_coord(pt);
            auto bar_c_g = bar_coord_grad(pt);

            // 0-2 : vertices
            for(size_t i = 0; i < 3; i++)
            {
                T coeff = 0.5 * (27.0*bar_c[i]*bar_c[i] - 18.0 * bar_c[i] + 2.0);
                ret(i,0) = coeff * bar_c_g(i,0);
                ret(i,1) = coeff * bar_c_g(i,1);
            }
            // 3-8 : face-points (112,122,002,022,001,011)
            for(size_t j = 0; j < 2; j++)
            {
                ret(3,j) = 4.5*( (6.0*bar_c[1]-1.0)*bar_c[2]*bar_c_g(1,j)
                                 + (3.0*bar_c[1]-1.0)*bar_c[1]*bar_c_g(2,j) );
                ret(4,j) = 4.5*( (6.0*bar_c[2]-1.0)*bar_c[1]*bar_c_g(2,j)
                                 + (3.0*bar_c[2]-1.0)*bar_c[2]*bar_c_g(1,j) );
                ret(5,j) = 4.5*( (6.0*bar_c[0]-1.0)*bar_c[2]*bar_c_g(0,j)
                                 + (3.0*bar_c[0]-1.0)*bar_c[0]*bar_c_g(2,j) );
                ret(6,j) = 4.5*( (6.0*bar_c[2]-1.0)*bar_c[0]*bar_c_g(2,j)
                                 + (3.0*bar_c[2]-1.0)*bar_c[2]*bar_c_g(0,j) );
                ret(7,j) = 4.5*( (6.0*bar_c[0]-1.0)*bar_c[1]*bar_c_g(0,j)
                                 + (3.0*bar_c[0]-1.0)*bar_c[0]*bar_c_g(1,j) );
                ret(8,j) = 4.5*( (6.0*bar_c[1]-1.0)*bar_c[0]*bar_c_g(1,j)
                                 + (3.0*bar_c[1]-1.0)*bar_c[1]*bar_c_g(0,j) );
            }
            // 9 : center of mass
            for(size_t j = 0; j < 2; j++)
            {
                ret(9,j) = 27.0 * (bar_c[1] * bar_c[2] * bar_c_g(0,j)
                                   + bar_c[0] * bar_c[1] * bar_c_g(2,j)
                                   + bar_c[2] * bar_c[0] * bar_c_g(1,j) );
            }
        }
        else // degree == 4
        {
            auto bar_c = bar_coord(pt);
            auto bar_c_g = bar_coord_grad(pt);
            // 0-2 : vertices
            for(size_t i = 0; i < 3; i++)
            {
                T coeff = (1.0/6.0) * (256.0*bar_c[i]*bar_c[i]*bar_c[i] - 288.0*bar_c[i]*bar_c[i]
                                       + 88.0*bar_c[i] - 6.0);
                ret(i,0) = coeff * bar_c_g(i,0);
                ret(i,1) = coeff * bar_c_g(i,1);
            }
            // 3-5 : mid-points (12,02,01)
            for(size_t j = 0; j < 2; j++)
            {
                ret(3,j) =
                    4.0*bar_c[1]*(32.0*bar_c[1]*bar_c[2] - 4.0*(2.0*bar_c[2]+bar_c[1]) + 1.0)*bar_c_g(2,j)
                    + 4.0*bar_c[2]*(32.0*bar_c[2]*bar_c[1] - 4.0*(2.0*bar_c[1]+bar_c[2])+1.0)*bar_c_g(1,j);

                ret(4,j) =
                    4.0*bar_c[0]*(32.0*bar_c[0]*bar_c[2] - 4.0*(2.0*bar_c[2]+bar_c[0]) + 1.0)*bar_c_g(2,j)
                    + 4.0*bar_c[2]*(32.0*bar_c[2]*bar_c[0] - 4.0*(2.0*bar_c[0]+bar_c[2])+1.0)*bar_c_g(0,j);

                ret(5,j) =
                    4.0*bar_c[0]*(32.0*bar_c[0]*bar_c[1] - 4.0*(2.0*bar_c[1]+bar_c[0]) + 1.0)*bar_c_g(1,j)
                    + 4.0*bar_c[1]*(32.0*bar_c[1]*bar_c[0] - 4.0*(2.0*bar_c[0]+bar_c[1])+1.0)*bar_c_g(0,j);
            }
            // 6-11 : face-points (1112,1222,0002,0222,0001,0111)
            for(size_t j = 0; j < 2; j++)
            {
                ret(6,j) =
                    (16.0/3.0) * bar_c[2] * (24.0*bar_c[1]*bar_c[1] -12.0*bar_c[1] + 1.0)*bar_c_g(1,j)
                    +(16.0/3.0) * bar_c[1] * (8.0*bar_c[1]*bar_c[1] -6.0*bar_c[1] + 1.0)*bar_c_g(2,j);

                ret(7,j) =
                    (16.0/3.0) * bar_c[1] * (24.0*bar_c[2]*bar_c[2] -12.0*bar_c[2] + 1.0)*bar_c_g(2,j)
                    +(16.0/3.0) * bar_c[2] * (8.0*bar_c[2]*bar_c[2] -6.0*bar_c[2] + 1.0)*bar_c_g(1,j);

                ret(8,j) =
                    (16.0/3.0) * bar_c[2] * (24.0*bar_c[0]*bar_c[0] -12.0*bar_c[0] + 1.0)*bar_c_g(0,j)
                    +(16.0/3.0) * bar_c[0] * (8.0*bar_c[0]*bar_c[0] -6.0*bar_c[0] + 1.0)*bar_c_g(2,j);

                ret(9,j) =
                    (16.0/3.0) * bar_c[0] * (24.0*bar_c[2]*bar_c[2] -12.0*bar_c[2] + 1.0)*bar_c_g(2,j)
                    +(16.0/3.0) * bar_c[2] * (8.0*bar_c[2]*bar_c[2] -6.0*bar_c[2] + 1.0)*bar_c_g(0,j);

                ret(10,j) =
                    (16.0/3.0) * bar_c[1] * (24.0*bar_c[0]*bar_c[0] -12.0*bar_c[0] + 1.0)*bar_c_g(0,j)
                    +(16.0/3.0) * bar_c[0] * (8.0*bar_c[0]*bar_c[0] -6.0*bar_c[0] + 1.0)*bar_c_g(1,j);

                ret(11,j) =
                    (16.0/3.0) * bar_c[0] * (24.0*bar_c[1]*bar_c[1] -12.0*bar_c[1] + 1.0)*bar_c_g(1,j)
                    +(16.0/3.0) * bar_c[1] * (8.0*bar_c[1]*bar_c[1] -6.0*bar_c[1] + 1.0)*bar_c_g(0,j);
            }
            // 12-14 : others (0012,0112,0122)
            for(size_t j = 0; j < 2; j++)
            {
                ret(12,j) = 32.0*(4.0*bar_c[0]-1.0)
                    *(bar_c[0]*bar_c[1]*bar_c_g(2,j)
                      + bar_c[1]*bar_c[2]*bar_c_g(0,j) + bar_c[2]*bar_c[0]*bar_c_g(1,j))
                    + 128.0 * bar_c[0]*bar_c[1]*bar_c[2]*bar_c_g(0,j);

                ret(13,j) = 32.0*(4.0*bar_c[1]-1.0)
                    *(bar_c[0]*bar_c[1]*bar_c_g(2,j)
                      + bar_c[1]*bar_c[2]*bar_c_g(0,j) + bar_c[2]*bar_c[0]*bar_c_g(1,j))
                    + 128.0 * bar_c[0]*bar_c[1]*bar_c[2]*bar_c_g(1,j);

                ret(14,j) = 32.0*(4.0*bar_c[2]-1.0)
                    *(bar_c[0]*bar_c[1]*bar_c_g(2,j)
                      + bar_c[1]*bar_c[2]*bar_c_g(0,j) + bar_c[2]*bar_c[0]*bar_c_g(1,j))
                    + 128.0 * bar_c[0]*bar_c[1]*bar_c[2]*bar_c_g(2,j);
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
        if( degree > 3 )
            throw std::invalid_argument("degree > 3 not yet supported");
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
        else if(basis_degree == 1)
        {
            ret(0) = - pos + 1.0; // 0
            ret(1) = pos;         // 1
        }
        else if(basis_degree == 2)
        {
            ret(0) = 2.0*pos*pos - 3.0*pos + 1.0; // 0
            ret(1) = -4.0*pos*pos + 4.0*pos;      // 0.5
            ret(2) = 2.0*pos*pos - pos;           // 1
        }
        else  // degree == 3
        {
            ret(0) = -0.5*(3*pos-1)*(3*pos-2)*(pos-1); // 0
            ret(1) = 4.5*pos*(3*pos-2)*(pos-1);        // 1/3
            ret(2) = -4.5*pos*(3*pos-1)*(pos-1);       // 2/3
            ret(3) = 0.5*pos*(3*pos-1)*(3*pos-2);      // 1
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
class contact_assembler
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

    contact_assembler(const Mesh& msh, hho_degree_info hdi)
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
            throw std::invalid_argument("contact_assembler: you have to use boundary type");

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
            throw std::invalid_argument("contact_assembler: you have to use boundary type");

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
    return contact_assembler<Mesh>(msh, hdi);
}

////////  STATIC CONDENSATION
template<typename Mesh, typename T>
auto
make_contact_SC(const Mesh&                                                      msh,
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


/////// static condensation for the membrane problem
template<typename Mesh, typename T>
auto
make_membrane_SC(const Mesh&                                                      msh,
                 const typename Mesh::cell_type&                                  cl,
                 const hho_degree_info&                                           hdi,
                 const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& lhs,
                 const typename Eigen::Matrix<T, Eigen::Dynamic, 1>&              rhs1,
                 const typename Eigen::Matrix<T, Eigen::Dynamic, 1>&              rhs2,
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
    if ((rhs1.size() != num_cell_dofs) && (rhs1.size() != num_total_dofs))
    {
        throw std::invalid_argument("static condensation: incorrect size of the rhs1");
    }
    if ((rhs2.size() != num_cell_dofs) && (rhs2.size() != num_total_dofs))
    {
        throw std::invalid_argument("static condensation: incorrect size of the rhs2");
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

    const vector_type cell_rhs1  = rhs1.head(num_cell_dofs);
    const vector_type cell_rhs2  = rhs2.head(num_cell_dofs);
    vector_type       faces_rhs1 = vector_type::Zero(num_faces_dofs);
    vector_type       faces_rhs2 = vector_type::Zero(num_faces_dofs);

    if (rhs1.size() == num_total_dofs)
    {
        faces_rhs1 = rhs1.tail(num_faces_dofs);
    }
    if (rhs2.size() == num_total_dofs)
    {
        faces_rhs2 = rhs2.tail(num_faces_dofs);
    }

    const matrix_type C_TT = D.topLeftCorner(num_cell_dofs, num_cell_dofs);
    const matrix_type D_TT = D.topRightCorner(num_cell_dofs, num_cell_dofs);

    const auto K_TT_ldlt = K_TT.ldlt();
    if (K_TT_ldlt.info() != Eigen::Success)
    {
        throw std::invalid_argument("static condensation: K_TT is not positive definite");
    }

    const matrix_type AL = K_TT_ldlt.solve(K_TF);
    const vector_type bL1 = K_TT_ldlt.solve(cell_rhs1);
    const vector_type bL2 = K_TT_ldlt.solve(cell_rhs2);

    const auto ID = matrix_type::Identity(num_cell_dofs, num_cell_dofs);
    const auto K_TT_inv = K_TT_ldlt.solve(ID);

    const auto E = 2.0*C_TT * K_TT_inv + D_TT;
    const auto E_inv = E.inverse();

    const matrix_type E2 = E_inv * C_TT;
    const matrix_type E3 = K_TT_ldlt.solve(E2);

    const matrix_type A12 = - K_FT * E3 * AL;
    const matrix_type A11 = K_FF - K_FT * AL - A12;

    const vector_type b1 = faces_rhs1 - K_FT * bL1 + K_FT * E3 * (bL1 - bL2);
    const vector_type b2 = faces_rhs2 - K_FT * bL2 + K_FT * E3 * (bL2 - bL1);

    // return two matrices and two RHS
    return std::make_pair(std::make_pair(A11, A12) , std::make_pair(b1, b2) );
}

//////////// MEMBRANE_STATIC_DECONDENSATION
template<typename Mesh, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
contact_static_decondensation_multF(const Mesh&                                                      msh,
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


///////////////////
template<typename Mesh, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
contact_static_decondensation(const Mesh&                                                      msh,
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

    vector_type ret          = vector_type::Zero(num_total_dofs + num_cell_dofs);
    ret.head(num_cell_dofs)  = solT;
    ret.block(num_cell_dofs, 0, num_faces_dofs, 1) = uF;
    ret.block(num_total_dofs, 0, num_cell_dofs, 1) = multT;

    return ret;
}

//////////////////////////////////
template<typename Mesh, typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1>
membrane_static_decondensation(const Mesh&                                                     msh,
                              const typename Mesh::cell_type&                                  cl,
                              const hho_degree_info&                                           hdi,
                              const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& lhs,
                              const typename Eigen::Matrix<T, Eigen::Dynamic, 1>&             rhs1,
                              const typename Eigen::Matrix<T, Eigen::Dynamic, 1>&             rhs2,
                              const typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& D,
                              const typename Eigen::Matrix<T, Eigen::Dynamic, 1>&             solF)
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

    if ( (rhs1.size() < num_cell_dofs) || (rhs2.size() < num_cell_dofs) )
    {
        throw std::invalid_argument("static condensation: incorrect size of the rhs");
    }
    const vector_type fT1 = rhs1.head(num_cell_dofs);
    const vector_type fT2 = rhs2.head(num_cell_dofs);

    const matrix_type K_TT = lhs.topLeftCorner(num_cell_dofs, num_cell_dofs);
    const matrix_type K_TF = lhs.topRightCorner(num_cell_dofs, num_faces_dofs);

    const matrix_type C_TT = D.topLeftCorner(num_cell_dofs, num_cell_dofs);
    const matrix_type D_TT = D.topRightCorner(num_cell_dofs, num_cell_dofs);

    const vector_type uF1 = solF.head(num_faces_dofs);
    const vector_type uF2 = solF.tail(num_faces_dofs);

    const auto K_TT_ldlt = K_TT.ldlt();
    if (K_TT_ldlt.info() != Eigen::Success)
    {
        throw std::invalid_argument("static condensation: K_TT is not positive definite");
    }

    const auto ID = matrix_type::Identity(num_cell_dofs, num_cell_dofs);
    const auto K_TT_inv = K_TT_ldlt.solve(ID);

    const auto E_inv = (2.0 * C_TT * K_TT_inv + D_TT).inverse();
    const matrix_type E2 = E_inv * C_TT;

    const vector_type solT1 = K_TT_ldlt.solve(fT1 - K_TF * uF1)
        - K_TT_inv * E2 * K_TT_ldlt.solve(fT1 - fT2 - K_TF * (uF1 - uF2) );

    const vector_type solT2 = K_TT_ldlt.solve(fT2 - K_TF * uF2)
        - K_TT_inv * E2 * K_TT_ldlt.solve(fT2 - fT1 - K_TF * (uF2 - uF1) );

    const vector_type multT = E2 * K_TT.ldlt().solve(fT2 - fT1 - K_TF * (uF2 - uF1) );

    vector_type ret          = vector_type::Zero(2 * num_total_dofs + num_cell_dofs);
    ret.head(num_cell_dofs)  = solT1;
    ret.block(num_cell_dofs, 0, num_faces_dofs, 1) = uF1;
    ret.block(num_total_dofs, 0, num_cell_dofs, 1) = solT2;
    ret.block(num_total_dofs + num_cell_dofs, 0, num_faces_dofs, 1) = uF2;
    ret.block(2*num_total_dofs, 0, num_cell_dofs, 1) = multT;

    return ret;
}


////////////////////////////////////////

template<typename Mesh>
class contact_condensed_assembler_multF
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

    contact_condensed_assembler_multF(const Mesh& msh, hho_degree_info hdi)
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
            throw std::invalid_argument("contact_condensed_assembler_multF: you have to use boundary type");

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

        auto loc_sol = contact_static_decondensation_multF(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

        vector_type solT = loc_sol.head(cbs);
        vector_type multT = loc_sol.block(cbs + fcs.size() * fbs, 0, cbs, 1);


        D = matrix_type::Zero(cbs, 2*cbs);
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

        auto SC = make_contact_SC(msh, cl, di, lhs, rhs, D);
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

        for (size_t i = 0; i < rhs_sc.rows(); i++)
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

            auto loc_sol = contact_static_decondensation_multF(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

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

        vector_type full_sol = contact_static_decondensation_multF(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

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
        const auto cb = make_scalar_Lagrange_basis(msh, cl, celdeg);

        matrix_type D = matrix_type::Zero(cbs, 2*cbs);

        for(size_t i = 0; i < cbs; i++)
        {
            auto OFF = cell_offset * cbs;
            if( active_constr.at(OFF + i) )
                D(i,i) = 1.0;
            else
                D(i,cbs + i) = 1.0;
        }

        auto full_sol = contact_static_decondensation_multF(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto num_faces = howmany_faces(msh, cl);

        // multT in the primal base
        auto multT_dual = full_sol.tail(cbs + num_faces * fbs).head(cbs);
        auto mass_matrixT = make_mass_matrix(msh, cl, cb);
        vector_type multT_primal = mass_matrixT.ldlt().solve(multT_dual);

        // multF in the primal base
        vector_type multF_primal = vector_type::Zero(num_faces*fbs);
        const auto fcs = faces(msh, cl);
        for(size_t face_i = 0; face_i < num_faces; face_i++)
        {
            const auto fc = fcs[face_i];
            const auto fb = make_scalar_Lagrange_basis(msh, fc, facdeg);

            auto multF_dual = full_sol.tail(num_faces * fbs).block(face_i*fbs,0,fbs,1);
            auto mass_matrixF = make_mass_matrix(msh, fc, fb);
            multF_primal.block(face_i*fbs,0,fbs,1) = mass_matrixF.ldlt().solve(multF_dual);
        }

        vector_type ret = vector_type::Zero(cbs + num_faces*fbs);
        ret.head(cbs) = multT_primal;
        ret.tail(num_faces * fbs) = multF_primal;
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
auto make_condensed_assembler_Lag_multF(const Mesh& msh, hho_degree_info hdi)
{
    return contact_condensed_assembler_multF<Mesh>(msh, hdi);
}


///////////////////////  ASSEMBLER WITHOUT MULT ON FACES  ////////////////////////

template<typename Mesh>
class contact_condensed_assembler
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

    contact_condensed_assembler(const Mesh& msh, hho_degree_info hdi)
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
        system_size = fbs * num_other_faces;

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
            throw std::invalid_argument("contact_condensed_assembler: you have to use boundary type");

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

        auto loc_sol = contact_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

        vector_type solT = loc_sol.head(cbs);
        vector_type multT = loc_sol.block(cbs + fcs.size() * fbs, 0, cbs, 1);


        D = matrix_type::Zero(cbs, 2*cbs);

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

        auto SC = make_contact_SC(msh, cl, di, lhs, rhs, D);
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

        for (size_t i = 0; i < rhs_sc.rows(); i++)
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

            vector_type solF = vector_type::Zero(num_dofs);
            for(size_t i=0; i<num_dofs; i++)
                solF(i) = 1.0;

            assemble_contrib(msh, cl, loc_LHS.at(cell_offset), loc_RHS.at(cell_offset),solF, dirichlet_bf);
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

        vector_type ret = vector_type::Zero(num_faces*fbs);

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
            }
            else
            {
                auto face_offset = priv::offset(msh, fc);
                auto face_SOL_offset = compress_table.at(face_offset)*fbs;
                ret.block(face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
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

            auto loc_sol = contact_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

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

        vector_type full_sol = contact_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

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
        const auto cb = make_scalar_Lagrange_basis(msh, cl, celdeg);

        matrix_type D = matrix_type::Zero(cbs, 2*cbs);

        for(size_t i = 0; i < cbs; i++)
        {
            auto OFF = cell_offset * cbs;
            if( active_constr.at(OFF + i) )
                D(i,i) = 1.0;
            else
                D(i,cbs + i) = 1.0;
        }

        auto full_sol = contact_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS.at( cell_offset ), D, solF);

        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto num_faces = howmany_faces(msh, cl);

        auto multT_dual = full_sol.tail(cbs);
        auto mass_matrixT = make_mass_matrix(msh, cl, cb);
        vector_type multT_primal = mass_matrixT.ldlt().solve(multT_dual);

        return multT_primal;
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
    return contact_condensed_assembler<Mesh>(msh, hdi);
}


//////////////////////////////////////////////

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
    std::vector< Matrix<T, Dynamic, 1> >       loc_RHS1, loc_RHS2;
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
        loc_RHS1.resize( num_cells );
        loc_RHS2.resize( num_cells );

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
                const vector_type&              rhs1,
                const vector_type&              rhs2)
    {
        auto cell_offset = offset(msh, cl);
        loc_LHS.at( cell_offset ) = lhs;
        loc_RHS1.at( cell_offset ) = rhs1;
        loc_RHS2.at( cell_offset ) = rhs2;

        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);
        for(size_t i = cell_offset * cbs; i < cell_offset * cbs + cbs; i++)
            active_constr.at(i) = false;
    }

    template<typename Function1, typename Function2>
    void
    assemble_contrib(const Mesh&                     msh,
                     const typename Mesh::cell_type& cl,
                     const matrix_type&              lhs,
                     const vector_type&              rhs1,
                     const vector_type&              rhs2,
                     const vector_type&              solF,
                     const Function1&                 dirichlet_bf1,
                     const Function2&                 dirichlet_bf2)
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

        auto loc_sol = membrane_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS1.at( cell_offset ), loc_RHS2.at( cell_offset ), D, solF);

        vector_type solT1 = loc_sol.head(cbs);
        vector_type solT2 = loc_sol.block(cbs + fcs.size() * fbs, 0, cbs, 1);
        vector_type multT = loc_sol.block(2*(cbs + fcs.size() * fbs), 0, cbs, 1);


        D = matrix_type::Zero(cbs, 2*cbs);

        for(size_t i = 0; i < cbs; i++)
        {
            auto delta_u    = solT1(i) - solT2(i);
            auto sol_mult = multT(i);

            if(delta_u <= 0.0 && sol_mult >= 0)
            {
                active_constr.at(cell_offset * cbs + i) = true;
                D(i,i) = 1.0;
            }
            else if(delta_u >= 0.0 && sol_mult <= 0)
            {
                active_constr.at(cell_offset * cbs + i) = false;
                D(i,cbs + i) = 1.0;
            }
            else if(delta_u * sol_mult > 0.0)
            {
                // this case is considered because of the error made during SC
                if(delta_u < sol_mult)
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

        auto SC = make_membrane_SC(msh, cl, di, lhs, rhs1, rhs2, D);
        matrix_type lhs11_sc = SC.first.first;
        matrix_type lhs12_sc = SC.first.second;
        vector_type rhs1_sc = SC.second.first;
        vector_type rhs2_sc = SC.second.second;

        std::vector<assembly_index> asm_map;
        asm_map.reserve(fcs.size() * fbs);

        vector_type dirichlet_data1 = vector_type::Zero(fcs.size()*fbs);
        vector_type dirichlet_data2 = vector_type::Zero(fcs.size()*fbs);

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
                dirichlet_data1.block(face_i * fbs, 0, fbs, 1) =
                  project_function(msh, fc, fb, dirichlet_bf1, di.face_degree());
                dirichlet_data2.block(face_i * fbs, 0, fbs, 1) =
                    project_function(msh, fc, fb, dirichlet_bf2, di.face_degree());
            }
        }


        size_t offset_2 = fbs * num_other_faces;

        // matrix assembly
        // A_11
        for (size_t i = 0; i < lhs11_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs11_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], asm_map[j], lhs11_sc(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs11_sc(i,j) * dirichlet_data1(j);
            }
        }

        // A_12
        for (size_t i = 0; i < lhs12_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs12_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(asm_map[i], offset_2 + asm_map[j], lhs12_sc(i,j)) );
                else
                    RHS(asm_map[i]) -= lhs12_sc(i,j) * dirichlet_data2(j);
            }
        }

        // A_21
        for (size_t i = 0; i < lhs12_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs12_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(offset_2 + asm_map[i], asm_map[j], lhs12_sc(i,j)) );
                else
                    RHS(offset_2 + asm_map[i]) -= lhs12_sc(i,j) * dirichlet_data1(j);
            }
        }

        // A_22
        for (size_t i = 0; i < lhs11_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;

            for (size_t j = 0; j < lhs11_sc.cols(); j++)
            {
                if ( asm_map[j].assemble() )
                    triplets.push_back( Triplet<T>(offset_2 + asm_map[i], offset_2 + asm_map[j], lhs11_sc(i,j)) );
                else
                    RHS(offset_2 + asm_map[i]) -= lhs11_sc(i,j) * dirichlet_data2(j);
            }
        }

        // RHS assembly
        for (size_t i = 0; i < rhs1_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(asm_map[i]) += rhs1_sc(i);
        }
        for (size_t i = 0; i < rhs2_sc.rows(); i++)
        {
            if (!asm_map[i].assemble())
                continue;
            RHS(offset_2 + asm_map[i]) += rhs2_sc(i);
        }

    } // assemble_contrib()


    // init : set no contact constraint and assemble matrix
    // (matrix for the first iteration)
    template<typename Function1, typename Function2>
    void
    init(const Mesh&                     msh,
         const Function1&                 dirichlet_bf1,
         const Function2&                 dirichlet_bf2)
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

            assemble_contrib(msh, cl, loc_LHS.at(cell_offset), loc_RHS1.at(cell_offset), loc_RHS2.at(cell_offset),solF, dirichlet_bf1,dirichlet_bf2);
        }

        // end assembly
        finalize();
    }


    template<typename Function1, typename Function2>
    vector_type
    get_solF(const Mesh& msh, const typename Mesh::cell_type& cl,
             const vector_type& solution, const Function1& dirichlet_bf1,
             const Function2& dirichlet_bf2)
    {
        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto fcs = faces(msh, cl);

        auto num_faces = fcs.size();
        size_t offset_2 = fbs * num_other_faces;

        vector_type ret = vector_type::Zero(2*num_faces*fbs);

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
                vector_type rhs1 = make_rhs(msh, fc, fb, dirichlet_bf1, di.face_degree());
                ret.block(face_i*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs1);

                vector_type rhs2 = make_rhs(msh, fc, fb, dirichlet_bf2, di.face_degree());
                ret.block( (num_faces + face_i)*fbs, 0, fbs, 1) = mass.ldlt().solve(rhs2);
            }
            else
            {
                auto face_offset = priv::offset(msh, fc);
                auto face_SOL_offset = compress_table.at(face_offset)*fbs;
                ret.block(face_i*fbs, 0, fbs, 1) = solution.block(face_SOL_offset, 0, fbs, 1);
                ret.block((num_faces+face_i)*fbs, 0, fbs, 1) =
                    solution.block(offset_2 + face_SOL_offset, 0, fbs, 1);
            }
        }

        return ret;
    }

    // update_mat : assemble matrix according to the previous iteration solution
    template<typename Function1, typename Function2>
    void
    update_mat(const Mesh&                     msh,
               const vector_type&              prev_sol,
               const Function1&                 dirichlet_bf1,
               const Function2&                 dirichlet_bf2)
    {
        // clear RHS
        RHS = vector_type::Zero(system_size);

        // assemble all local contributions for Laplacian part
        for (auto& cl : msh)
        {
            auto solF = get_solF(msh, cl, prev_sol, dirichlet_bf1, dirichlet_bf2);

            auto cell_offset = offset(msh, cl);
            assemble_contrib(msh, cl, loc_LHS.at(cell_offset), loc_RHS1.at(cell_offset), loc_RHS2.at(cell_offset),solF, dirichlet_bf1, dirichlet_bf2);
        }

        finalize();
    }

    template<typename Function1, typename Function2>
    bool
    stop(const Mesh&         msh,
         const vector_type&  sol, const Function1& dirichlet_bf1, const Function2& dirichlet_bf2)
    {
        T TOL = 1e-16;

        const auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension - 1);
        const auto cbs = scalar_basis_size(di.cell_degree(), Mesh::dimension);

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

            auto solF = get_solF(msh, cl, sol, dirichlet_bf1, dirichlet_bf2);

            auto loc_sol = membrane_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS1.at( cell_offset ), loc_RHS2.at( cell_offset ), D, solF);

            const auto fcs = faces(msh, cl);
            vector_type solT1 = loc_sol.head(cbs);
            vector_type solT2 = loc_sol.block(cbs + fcs.size() * fbs, 0, cbs, 1);
            vector_type multT = loc_sol.block(2*(cbs + fcs.size() * fbs), 0, cbs, 1);

            for(size_t i = 0; i < cbs; i++)
            {
                auto delta_u    = solT1(i) - solT2(i);
                auto sol_mult = multT(i);

                if(delta_u < -TOL || sol_mult < -TOL)
                    return false;
            }
        }

        return true;
    }



    template<typename Function1, typename Function2>
    vector_type
    take_u(const Mesh& msh, const typename Mesh::cell_type& cl,
           const vector_type& solution, const Function1& dirichlet_bf1,
           const Function2& dirichlet_bf2)
    {
        auto solF = get_solF(msh, cl, solution, dirichlet_bf1, dirichlet_bf2);

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

        vector_type full_sol = membrane_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS1.at( cell_offset ), loc_RHS2.at( cell_offset ), D, solF);

        return full_sol.head(2*(cbs + num_faces * fbs));
    }

    template<typename Function1, typename Function2>
    vector_type
    take_mult(const Mesh& msh, const typename Mesh::cell_type& cl,
              const vector_type& solution, const Function1& dirichlet_bf1,
              const Function2& dirichlet_bf2)
    {
        auto solF = get_solF(msh, cl, solution, dirichlet_bf1, dirichlet_bf2);

        auto cell_offset        = offset(msh, cl);
        auto celdeg = di.cell_degree();
        auto cbs = scalar_basis_size(celdeg, Mesh::dimension);
        const auto cb = make_scalar_Lagrange_basis(msh, cl, celdeg);

        matrix_type D = matrix_type::Zero(cbs, 2*cbs);

        for(size_t i = 0; i < cbs; i++)
        {
            auto OFF = cell_offset * cbs;
            if( active_constr.at(OFF + i) )
                D(i,i) = 1.0;
            else
                D(i,cbs + i) = 1.0;
        }

        auto full_sol = membrane_static_decondensation(msh, cl, di, loc_LHS.at( cell_offset ), loc_RHS1.at( cell_offset ), loc_RHS2.at( cell_offset ), D, solF);

        auto facdeg = di.face_degree();
        auto fbs = scalar_basis_size(di.face_degree(), Mesh::dimension-1);
        auto num_faces = howmany_faces(msh, cl);

        auto multT_dual = full_sol.tail(cbs);
        auto mass_matrixT = make_mass_matrix(msh, cl, cb);
        vector_type multT_primal = mass_matrixT.ldlt().solve(multT_dual);

        return multT_primal;
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
auto make_membrane_condensed_assembler_Lag(const Mesh& msh, hho_degree_info hdi)
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
run_contact_solver(const Mesh& msh, size_t degree)
{
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;

    hho_degree_info hdi(degree+1, degree);

#if 0
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
    auto sol_grad = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
        {
            T coeff = 2.0*2.0*(r2 - R2);
            ret(0) =  coeff * x1;
            ret(1) =  coeff * y1;
        }
        else
        {
            ret(0) = 0.0;
            ret(1) = 0.0;
        }
        return ret;
    };
    auto mult_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 8.0 * R2;
    };
#else
    auto rhs_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return -24.0*(r2-R2)*(r2-R2)*(r2-R2)*(r2-R2)*(6.0*r2-R2);
        else
            return -r2*(R2-r2)*(R2-r2)*(R2-r2);
    };
    auto sol_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
        else
            return 0.0;
    };
    auto sol_grad = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
        {
            T coeff = 2.0*6.0*(r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
            ret(0) =  coeff * x1;
            ret(1) =  coeff * y1;
        }
        else
        {
            ret(0) = 0.0;
            ret(1) = 0.0;
        }

        return ret;
    };
    auto mult_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return r2*(R2-r2)*(R2-r2)*(R2-r2);
    };
#endif

    // auto assembler_sc = make_condensed_assembler_Lag_multF(msh, hdi);
    auto assembler_sc = make_condensed_assembler_Lag(msh, hdi);
    auto assembler = make_assembler_Lag(msh, hdi);

    bool scond = true; // static condensation

    for (auto& cl : msh)
    {
        auto cb     = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
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


    if(scond)
        std::cout << green << "end assembly : nb dof = " << assembler_sc.RHS.size() << std::endl;
    else
        std::cout << green << "end assembly : nb dof = " << assembler.RHS.size() << std::endl;


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

    T u_H1_error = 0.0;
    T u_L2_error = 0.0;
    T mult_L2_error = 0.0;

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
        {
            fullsol = assembler_sc.take_u(msh, cl, sol, sol_fun);
            mult_sol = assembler_sc.take_mult(msh, cl, sol, sol_fun);
        }
        else
        {
            fullsol = assembler.take_u(msh, cl, sol, sol_fun);
            mult_sol = assembler.take_mult(msh, cl, sol);
        }

        auto cell_dofs = fullsol.head( cb.size() );
        auto mult_cell_dofs = mult_sol.head( cb.size() );

        // errors
        const auto celdeg = hdi.cell_degree();
        const auto qps = integrate(msh, cl, 2*celdeg);
        for (auto& qp : qps)
        {
            auto grad_ref = sol_grad( qp.point() );
            auto t_dphi = cb.eval_gradients( qp.point() );
            Matrix<T, 1, 2> grad = Matrix<T, 1, 2>::Zero();

            for (size_t i = 0; i < cbs; i++ )
                grad += cell_dofs(i) * t_dphi.block(i, 0, 1, 2);

            // H1-error
            u_H1_error += qp.weight() * (grad_ref - grad).dot(grad_ref - grad);

            // L2-error
            auto t_phi = cb.eval_functions( qp.point() );
            T v = cell_dofs.dot( t_phi );
            u_L2_error += qp.weight() * (sol_fun(qp.point()) - v) * (sol_fun(qp.point()) - v);

            // mult-L2-error
            T mult = mult_cell_dofs.dot( t_phi );
            T mult_sol = mult_fun(qp.point());
            mult_L2_error += qp.weight() * (mult_sol - mult) * (mult_sol - mult);
        }

        // gnuplot output for cells
        auto pts = points(msh, cl);
        for(size_t i=0; i < pts.size(); i++)
        {
            T sol_uT = cell_dofs.dot( cb.eval_functions( pts[i] ) );
            uT_gp->add_data( pts[i], sol_uT );
            T sol_multT = mult_cell_dofs.dot( cb.eval_functions(pts[i]) );
            multT_gp->add_data( pts[i], sol_multT );
        }

        // gnuplot output for faces
        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto fcs = faces(msh, cl);
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc = fcs[face_i];
            auto face_sol = fullsol.block(cbs+face_i*fbs, 0, fbs, 1);

            const auto fb = make_scalar_Lagrange_basis(msh, fc, hdi.face_degree());
            auto barF = barycenter(msh, fc);

            T solbarF = fb.eval_functions(barF).dot(face_sol);
            uF_gp->add_data( barF, solbarF );

            if( mult_sol.size() > cbs )
            {
                auto face_mult = mult_sol.block(cbs+face_i*fbs, 0, fbs, 1);
                T multbarF = fb.eval_functions(barF).dot(face_mult);
                multF_gp->add_data( barF, multbarF );
            }
        }
    }

    postoutput.add_object(uT_gp);
    postoutput.add_object(multT_gp);
    postoutput.add_object(uF_gp);
    postoutput.add_object(multF_gp);
    postoutput.write();

    std::cout << yellow << "ended run : H1-error is " << std::sqrt(u_H1_error) << std::endl;
    std::cout << yellow << "            L2-error is " << std::sqrt(u_L2_error) << std::endl;
    std::cout << yellow << "            mult-L2-error is " << std::sqrt(mult_L2_error) << std::endl;

    return std::sqrt(u_H1_error);
}




////////////////////////////////

template<typename Mesh>
typename Mesh::coordinate_type
run_membranes_solver(const Mesh& msh, size_t degree)
{
    using T = typename Mesh::coordinate_type;
    using point_type = typename Mesh::point_type;

    hho_degree_info hdi(degree+1, degree);

#if 1
    auto rhs_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 8.0 * R2 - 16.0 * r2;
        else
            return - 8.0 * R2;
    };
    auto rhs_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 8.0 * R2;
    };
    auto sol_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return (r2 - R2) * (r2 - R2);
        else
            return 0.0;
    };
    auto sol_fun2 = [](const point_type& pt) -> T {
        return 0.0;
    };
    auto sol_grad1 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
        {
            T coeff = 2.0*2.0*(r2 - R2);
            ret(0) =  coeff * x1;
            ret(1) =  coeff * y1;
        }
        else
        {
            ret(0) = 0.0;
            ret(1) = 0.0;
        }
        return ret;
    };
    auto sol_grad2 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;

        ret(0) = 0.0;
        ret(1) = 0.0;

        return ret;
    };
    auto mult_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 8.0 * R2;
    };
#elif 1
    auto rhs_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return -4.0;
        else
            return -6.0;
    };
    auto rhs_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        auto r = std::sqrt(r2);
        if(r2 > R2)
            return (9.0*r2 - 4.0*r - R2) / r;
        else
            return -2.0;
    };
    auto sol_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;

        return r2 - R2;
    };
    auto sol_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        auto r = std::sqrt(r2);
        if(r2 > R2)
            return (1.0-r) * (r2 - R2);
        else
            return r2 - R2;
    };
    auto sol_grad1 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        ret(0) = 2*x1;
        ret(1) = 2*y1;
        return ret;
    };
    auto sol_grad2 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        auto r = std::sqrt(r2);
        if(r2 > R2)
        {
            T coeff = (2.*r - 3.*r2 - R2) / r;
            ret(0) = coeff * x1;
            ret(1) = coeff * y1;
        }
        else
        {
            ret(0) = 2*x1;
            ret(1) = 2*y1;
        }
        return ret;

    };
    auto mult_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 2.0;
    };
#else
    auto rhs_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return -24.0*(r2-R2)*(r2-R2)*(r2-R2)*(r2-R2)*(6.0*r2-R2);
        else
            return -1000.0*r2*std::sqrt(r2)*(R2-r2)*(R2-r2)*(R2-r2);
    };
    auto rhs_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 24.0*(r2-R2)*(r2-R2)*(r2-R2)*(r2-R2)*(6.0*r2-R2);
        else
            return 1000.0*r2*std::sqrt(r2)*(R2-r2)*(R2-r2)*(R2-r2);
    };
    auto sol_fun1 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
        else
            return 0.0;
    };
    auto sol_fun2 = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return -(r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
        else
            return 0.0;
    };
    auto sol_grad1 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
        {
            T coeff = 2.0*6.0*(r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
            ret(0) =  coeff * x1;
            ret(1) =  coeff * y1;
        }
        else
        {
            ret(0) = 0.0;
            ret(1) = 0.0;
        }

        return ret;
    };
    auto sol_grad2 = [](const point_type& pt) -> auto {
        Matrix<T, 1, 2> ret;
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
        {
            T coeff = 2.0*6.0*(r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2) * (r2 - R2);
            ret(0) =  -coeff * x1;
            ret(1) =  -coeff * y1;
        }
        else
        {
            ret(0) = 0.0;
            ret(1) = 0.0;
        }

        return ret;
    };
    auto mult_fun = [](const point_type& pt) -> T {
        auto x1 = pt.x() - 0.5;
        auto y1 = pt.y() - 0.5;
        auto r2 = x1*x1 + y1*y1;
        auto R2 = 1.0 / 9.0;
        if(r2 > R2)
            return 0.0;
        else
            return 1000.0*r2*std::sqrt(r2)*(R2-r2)*(R2-r2)*(R2-r2);
    };
#endif


    auto assembler_sc = make_membrane_condensed_assembler_Lag(msh, hdi);

    bool scond = true; // static condensation

    for (auto& cl : msh)
    {
        auto cb     = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
        auto gr     = make_vector_hho_gradrec_Lag(msh, cl, hdi);
        auto stab   = make_scalar_hdg_stabilization_Lag(msh, cl, hdi);
        auto rhs1    = make_rhs(msh, cl, cb, rhs_fun1);
        auto rhs2    = make_rhs(msh, cl, cb, rhs_fun2);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = gr.second + stab;
        if(scond)
            assembler_sc.set_loc_mat(msh, cl, A, rhs1, rhs2);
        else
        {
            // assembler.set_loc_mat(msh, cl, A, rhs);
        }
    }


    if(scond)
        std::cout << green << "end assembly : nb dof = " << assembler_sc.RHS.size() << std::endl;
    // else
        // std::cout << green << "end assembly : nb dof = " << assembler.RHS.size() << std::endl;


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
                assembler_sc.init(msh, sol_fun1, sol_fun2);
            else
                assembler_sc.update_mat(msh, sol, sol_fun1, sol_fun2);
            systsz = assembler_sc.LHS.rows();
        }
        else
        {
            // if(Newton_iter == 0)
            //     assembler.init(msh, sol_fun);
            // else
            //     assembler.update_mat(msh, sol, sol_fun);
            // systsz = assembler.LHS.rows();
        }
        sol = dynamic_vector<T>::Zero(systsz);

        disk::solvers::pardiso_params<T> pparams;
        pparams.report_factorization_Mflops = false;

        if(scond)
            mkl_pardiso(pparams, assembler_sc.LHS, assembler_sc.RHS, sol);
        // else
            // mkl_pardiso(pparams, assembler.LHS, assembler.RHS, sol);

        Newton_iter++;
        std::cout << blue << "end Newton iter nb " << Newton_iter << std::endl;

        if(scond)
            stop_loop = assembler_sc.stop(msh, sol, sol_fun1, sol_fun2);
        // else
        //     stop_loop = assembler.stop(msh, sol);
    } // Newton loop

    std::cout << "Start post-process" << std::endl;

    T u_H1_error = 0.0;
    T u_L2_error = 0.0;
    T mult_L2_error = 0.0;

    postprocess_output<T>  postoutput;

    auto uT1_gp  = std::make_shared< gnuplot_output_object<T> >("uT1.dat");
    auto uT2_gp  = std::make_shared< gnuplot_output_object<T> >("uT2.dat");
    auto multT_gp  = std::make_shared< gnuplot_output_object<T> >("multT.dat");

    auto uF1_gp  = std::make_shared< gnuplot_output_object<T> >("uF1.dat");
    auto uF2_gp  = std::make_shared< gnuplot_output_object<T> >("uF2.dat");
    auto multF_gp  = std::make_shared< gnuplot_output_object<T> >("multF.dat");

    for (auto& cl : msh)
    {
        auto cb     = make_scalar_Lagrange_basis(msh, cl, hdi.cell_degree());
        auto cbs = cb.size();
        const auto fbs = scalar_basis_size(hdi.face_degree(), Mesh::dimension - 1);
        const auto num_faces = howmany_faces(msh, cl);

        Eigen::Matrix<T, Eigen::Dynamic, 1> fullsol, mult_sol;

        if(scond)
        {
            fullsol = assembler_sc.take_u(msh, cl, sol, sol_fun1, sol_fun2);
            mult_sol = assembler_sc.take_mult(msh, cl, sol, sol_fun1, sol_fun2);
        }
        else
        {
            // fullsol = assembler.take_u(msh, cl, sol, sol_fun);
            // mult_sol = assembler.take_mult(msh, cl, sol);
        }

        auto cell_dofs1 = fullsol.head( cbs );
        Matrix<T, Dynamic, 1> cell_dofs2 = fullsol.block( cbs + num_faces * fbs, 0, cbs, 1);
        auto mult_cell_dofs = mult_sol.head( cbs );

        // errors
        const auto celdeg = hdi.cell_degree();
        const auto qps = integrate(msh, cl, 2*celdeg);
        for (auto& qp : qps)
        {
            auto grad_ref1 = sol_grad1( qp.point() );
            auto grad_ref2 = sol_grad2( qp.point() );
            auto t_dphi = cb.eval_gradients( qp.point() );
            Matrix<T, 1, 2> grad1 = Matrix<T, 1, 2>::Zero();
            Matrix<T, 1, 2> grad2 = Matrix<T, 1, 2>::Zero();

            for (size_t i = 0; i < cbs; i++ )
            {
                grad1 += cell_dofs1(i) * t_dphi.block(i, 0, 1, 2);
                grad2 += cell_dofs2(i) * t_dphi.block(i, 0, 1, 2);
            }

            // H1-error
            u_H1_error += qp.weight() * (grad_ref1 - grad1).dot(grad_ref1 - grad1);
            u_H1_error += qp.weight() * (grad_ref2 - grad2).dot(grad_ref2 - grad2);

            // L2-error
            auto t_phi = cb.eval_functions( qp.point() );
            T v1 = cell_dofs1.dot( t_phi );
            T v2 = cell_dofs2.dot( t_phi );
            u_L2_error += qp.weight() * (sol_fun1(qp.point()) - v1) * (sol_fun1(qp.point()) - v1);
            u_L2_error += qp.weight() * (sol_fun2(qp.point()) - v2) * (sol_fun2(qp.point()) - v2);

            // mult-L2-error
            T mult = mult_cell_dofs.dot( t_phi );
            T mult_sol = mult_fun(qp.point());
            mult_L2_error += qp.weight() * (mult_sol - mult) * (mult_sol - mult);
        }

        // gnuplot output for cells
        auto pts = points(msh, cl);
        for(size_t i=0; i < pts.size(); i++)
        {
            T sol_uT1 = cell_dofs1.dot( cb.eval_functions( pts[i] ) );
            uT1_gp->add_data( pts[i], sol_uT1 );
            T sol_uT2 = cell_dofs2.dot( cb.eval_functions( pts[i] ) );
            uT2_gp->add_data( pts[i], sol_uT2 );
            T sol_multT = mult_cell_dofs.dot( cb.eval_functions(pts[i]) );
            multT_gp->add_data( pts[i], sol_multT );
        }

        // gnuplot output for faces
        const auto fcs = faces(msh, cl);
        for (size_t face_i = 0; face_i < fcs.size(); face_i++)
        {
            const auto fc = fcs[face_i];
            auto face_sol1 = fullsol.block(cbs+face_i*fbs, 0, fbs, 1);
            auto face_sol2 = fullsol.block(2*cbs+(fcs.size()+face_i)*fbs, 0, fbs, 1);

            const auto fb = make_scalar_Lagrange_basis(msh, fc, hdi.face_degree());
            auto barF = barycenter(msh, fc);

            T solbarF1 = fb.eval_functions(barF).dot(face_sol1);
            uF1_gp->add_data( barF, solbarF1 );

            T solbarF2 = fb.eval_functions(barF).dot(face_sol2);
            uF2_gp->add_data( barF, solbarF2 );

            if( mult_sol.size() > cbs )
            {
                auto face_mult = mult_sol.block(cbs+face_i*fbs, 0, fbs, 1);
                T multbarF = fb.eval_functions(barF).dot(face_mult);
                multF_gp->add_data( barF, multbarF );
            }
        }
    }


#if 1
    // silo outputs for domains B and varpi
    silo_database silo;
    silo.create("membrane.silo");
    silo.add_mesh(msh, "mesh");
    // std::vector<T> cell_varpi, cell_B;

    // for(auto& cl : msh)
    // {
    //     cell_varpi.push_back( varpi_fun( barycenter(msh, cl) ) );
    //     cell_B.push_back( B_fun( barycenter(msh, cl) ) );
    // }
    // silo_zonal_variable<T> silo_varpi("varpi", cell_varpi);
    // silo_zonal_variable<T> silo_B("B", cell_B);
    // silo.add_variable("mesh", silo_varpi);
    // silo.add_variable("mesh", silo_B);
    silo.close();
#endif


    postoutput.add_object(uT1_gp);
    postoutput.add_object(uT2_gp);
    postoutput.add_object(multT_gp);
    postoutput.add_object(uF1_gp);
    postoutput.add_object(uF2_gp);
    postoutput.add_object(multF_gp);
    postoutput.write();

    std::cout << yellow << "ended run : H1-error is " << std::sqrt(u_H1_error) << std::endl;
    std::cout << yellow << "            L2-error is " << std::sqrt(u_L2_error) << std::endl;
    std::cout << yellow << "            mult-L2-error is " << std::sqrt(mult_L2_error) << std::endl;
    
    return std::sqrt(u_H1_error);
}

///////////////////////////////

using namespace Eigen;

int main(void)
{
    using T = double;

    // degree of the polynomials on the faces
    size_t degree = 3;
    
    typedef disk::generic_mesh<T, 2>  mesh_type;
    typedef disk::simplicial_mesh<T, 2>  mesh_type2;
    
    if(1)
    {
        std::vector<std::string> meshfiles;
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_1.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_2.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_3.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_5.typ1");
        // meshfiles.push_back("../../../diskpp/meshes/2D_triangles/fvca5/mesh1_5.typ1");

        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j3.mesh2d");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j4.mesh2d");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j5.mesh2d");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j6.mesh2d");
        meshfiles.push_back("../../../diskpp/meshes/2D_triangles/netgen/mesh_j7.mesh2d");

        for(size_t i=0; i < meshfiles.size(); i++)
        {
            // mesh_type msh;
            mesh_type2 msh;
            // disk::fvca5_mesh_loader<T, 2> loader;
            disk::netgen_mesh_loader<T, 2> loader;
            if (!loader.read_mesh(meshfiles.at(i)) )
            {
                std::cout << "Problem loading mesh." << std::endl;
            }
            loader.populate_mesh(msh);
            // run_contact_solver(msh, degree);
            run_membranes_solver(msh, degree);
        }

    }
    else
    {
        mesh_type2 msh;
        // disk::fvca5_mesh_loader<T, 2> loader;
        disk::netgen_mesh_loader<T, 2> loader;
        // std::string mesh_filename = "../../../diskpp/meshes/2D_triangles/fvca5/mesh1_4.typ1";
        std::string mesh_filename = "../../../diskpp/meshes/2D_triangles/netgen/mesh_j7.mesh2d";
        if (!loader.read_mesh(mesh_filename) )
        {
            std::cout << "Problem loading mesh." << std::endl;
        }
        loader.populate_mesh(msh);
        // run_contact_solver(msh, degree);
        run_membranes_solver(msh, degree);
    }

    std::cout << "\a" << std::endl;
    return 0;
}
