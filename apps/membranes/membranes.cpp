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

    //size_t degree = 0;

    hho_degree_info hdi(degree);

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
    // auto rhs_fun = make_rhs_function(msh);
    // auto sol_fun = make_solution_function(msh);

    auto assembler = make_diffusion_assembler(msh, hdi);

    for (auto& cl : msh)
    {
        auto cb = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
        auto gr     = make_scalar_hho_laplacian(msh, cl, hdi);
        auto stab   = make_scalar_hho_stabilization(msh, cl, gr.first, hdi);
        auto rhs    = make_rhs(msh, cl, cb, rhs_fun);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = gr.second + stab;
        auto sc     = make_scalar_static_condensation(msh, cl, hdi, A, rhs);
        assembler.assemble(msh, cl, sc.first, sc.second, sol_fun);
    }

    assembler.finalize();

    size_t systsz = assembler.LHS.rows();
    size_t nnz = assembler.LHS.nonZeros();

    //std::cout << "Mesh elements: " << msh.cells_size() << std::endl;
    //std::cout << "Mesh faces: " << msh.faces_size() << std::endl;
    //std::cout << "Dofs: " << systsz << std::endl;

    dynamic_vector<T> sol = dynamic_vector<T>::Zero(systsz);

    disk::solvers::pardiso_params<T> pparams;
    pparams.report_factorization_Mflops = false;
    mkl_pardiso(pparams, assembler.LHS, assembler.RHS, sol);

    T error = 0.0;

    //std::ofstream ofs("sol.dat");

    postprocess_output<T>  postoutput;

    auto diff_uT_gp  = std::make_shared< gnuplot_output_object<T> >("diff_uT.dat");
    auto uT_gp  = std::make_shared< gnuplot_output_object<T> >("uT.dat");

    for (auto& cl : msh)
    {
        auto cb     = make_scalar_monomial_basis(msh, cl, hdi.cell_degree());
        auto gr     = make_scalar_hho_laplacian(msh, cl, hdi);
        auto stab   = make_scalar_hho_stabilization(msh, cl, gr.first, hdi);
        auto rhs    = make_rhs(msh, cl, cb, rhs_fun);
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> A = gr.second + stab;

        Eigen::Matrix<T, Eigen::Dynamic, 1> locsol =
            assembler.take_local_data(msh, cl, sol, sol_fun);

        Eigen::Matrix<T, Eigen::Dynamic, 1> fullsol = make_scalar_static_decondensation(msh, cl, hdi, A, rhs, locsol);

        Eigen::Matrix<T, Eigen::Dynamic, 1> realsol = project_function(msh, cl, hdi, sol_fun, 2);


        auto diff = realsol - fullsol;
        error += diff.dot(A*diff);

        auto bar = barycenter(msh, cl);
        diff_uT_gp->add_data( bar, diff.dot(A*diff) );


        auto test = fullsol.head( cb.size() );
        T sol_uT = test.dot( cb.eval_functions(bar) );
        // T sol = fullsol.head( cb.size() ).dot( cb );
        uT_gp->add_data( bar, sol_uT );

        //for (size_t i = 0; i < Mesh::dimension; i++)
        //    ofs << bar[i] << " ";
        //ofs << fullsol(0) << std::endl;

    }

    //std::cout << std::sqrt(error) << std::endl;

    //ofs.close();

    postoutput.add_object(diff_uT_gp);
    postoutput.add_object(uT_gp);
    postoutput.write();

    std::cout << "ended run : error is " << std::sqrt(error) << std::endl;
    
    return std::sqrt(error);
}

using namespace Eigen;

int main(void)
{
    using T = double;

    
    typedef disk::generic_mesh<T, 2>  mesh_type;
    
    
    if(0)
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
