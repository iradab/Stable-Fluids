#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <iostream>
#include <igl/unproject_onto_mesh.h>
#include <ostream>
#include <chrono>
#include <thread>
#include "simulator.hpp"


using namespace Eigen;
using namespace std;
MatrixXd V1;
MatrixXi F1;

MatrixXd draw_bounding_box(igl::opengl::glfw::Viewer& viewer, int k)
{
    MatrixXd V_box;
    int knew = k * k;
    V_box = MatrixXd::Zero(knew, 3);

    for (int i = 0; i < k; i++) {
        for (int j = 0; j < k; j++) {
            V_box.row(i + k * j) = (Vector3d(i, j, 0)).transpose();
        }
    }
    viewer.append_mesh();
    return V_box;
}

int main(int argc, char* argv[]) {
    Simulator sim;
    igl::opengl::glfw::Viewer viewer;
    int starti = -1; int startj = -1;
    viewer.core().is_animating = true;
    float Ss[64][64] = { 0 };
    float Force0[64][64] = { 0 };
    float Force1[64][64] = { 0 };

    MatrixXd V;
    int n = 64;
    V = draw_bounding_box(viewer, n + 1);
    MatrixXd V2 = MatrixXd::Zero((n) * (n) * 4, 3);
    MatrixXi F = MatrixXi::Zero(n * n * 2, 3);
    MatrixXd C = MatrixXd::Zero(n * n * 2, 3);
    for (int index = 0; index < n * n; index++) {
        int x = index % n;
        int y = index / n;
        V2.row(index * 4) = (Vector3d(x, y, 0)).transpose();
        V2.row(index * 4 + 1) = (Vector3d(x + 1, y, 0)).transpose();
        V2.row(index * 4 + 2) = (Vector3d(x + 1, y + 1, 0)).transpose();
        V2.row(index * 4 + 3) = (Vector3d(x, y + 1, 0)).transpose();
        F(index * 2, 0) = index * 4; F(index * 2, 1) = index * 4 + 1; F(index * 2, 2) = index * 4 + 2;
        F(index * 2 + 1, 0) = index * 4; F(index * 2 + 1, 1) = index * 4 + 2; F(index * 2 + 1, 2) = index * 4 + 3;
    }
    int new_i = 0;
    viewer.core().lighting_factor = 0;
    using RT = igl::opengl::ViewerCore::RotationType;
    viewer.core().rotation_type = RT::ROTATION_TYPE_NO_ROTATION;
    viewer.callback_mouse_up =
        [&starti, &startj, &V2, &F, &Force0, &Force1, &Ss](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        int fid;
        Eigen::Vector3f bc;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view,
            viewer.core().proj, viewer.core().viewport, V2, F, fid, bc))
        {

            int i = int(fid / 128);
            int j = int((fid % 128) / 2);

            for (int a = starti - 2; a <= starti + 2; a++) {
                for (int b = startj - 2; b <= startj + 2; b++)
                    Ss[a][b] = 1;
            }
            for (int a = starti - 2; a <= starti + 2; a++) {
                for (int b = startj - 2; b <= startj + 2; b++) {
                    Force0[a][b] = (starti - i) / 10;
                    Force1[a][b] = (j - startj) / 10;
                }
            }
            return true;
        }
        return false;
    };
    viewer.callback_mouse_down =
        [&starti, &startj, &V2, &F, &Ss](igl::opengl::glfw::Viewer& viewer, int, int)->bool
    {
        int fid;
        Eigen::Vector3f bc;
        double x = viewer.current_mouse_x;
        double y = viewer.core().viewport(3) - viewer.current_mouse_y;
        if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view,
            viewer.core().proj, viewer.core().viewport, V2, F, fid, bc))
        {
            int i = int(fid / 128);
            int j = int((fid % 128) / 2);
            starti = i;
            startj = j;
            return true;
        }
        return false;
    };
    viewer.data().set_mesh(V2, F);
    viewer.core().align_camera_center(V2, F1);
    viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer&)->bool
    {
        viewer.data().set_colors(C);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sim.Ssource[n - 1 - i][j] = Ss[i][j];

            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                sim.F[0][n - 1 - i][j] = Force0[i][j];
                sim.F[1][n - 1 - i][j] = Force1[i][j];
            }
        }
        sim.SwapV(sim.U, sim.Uprev);
        sim.SwapS(sim.S, sim.Sprev);
        sim.Vstep(sim.U, sim.Uprev);
        sim.Sstep(sim.S, sim.Sprev, sim.U);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                C.row((i * 64 + j) * 2) = RowVector3d(sim.S[n - 1 - i][j], 0, 0);
                C.row((i * 64 + j) * 2 + 1) = RowVector3d(sim.S[n - 1 - i][j], 0, 0);
            }
        }

        new_i++;
        return false;
    };
    viewer.launch();
}

