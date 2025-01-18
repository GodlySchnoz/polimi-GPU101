#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <climits>
#include <cuda_runtime.h>

#define MAX_FRONTIER_SIZE 128

#define CHECK(call)                                                                     \
{                                                                                       \
    const cudaError_t err = call;                                                       \
    if (err != cudaSuccess) {                                                           \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}

#define CHECK_KERNELCALL()                                                              \
{                                                                                       \
    const cudaError_t err = cudaGetLastError();                                         \
    if (err != cudaSuccess) {                                                           \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);   \
        exit(EXIT_FAILURE);                                                             \
    }                                                                                   \
}

__global__ void bfs_ker(int *Va, int *Ea, char *Fa, char *Xa, int *Ca, int num_vertex) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vertex || !Fa[tid]) return; // Escape condition: Out of bounds or inactive frontier

    Fa[tid] = 0; // Mark this node as processed
    Xa[tid] = 1; // Mark this node as explored

    for (int i = Va[tid]; i < Va[tid + 1]; ++i) {
        int neighbor = Ea[i];
        if (!Xa[neighbor]) { // If neighbor not explored
            Fa[neighbor] = 1;
            atomicMin(&Ca[neighbor], Ca[tid] + 1);
        }
    }
}

void bfs_cuda(std::vector<int> &Va, std::vector<int> &Ea, int source, int num_vertex) {
    int *Va_d, *Ea_d, *Ca_d;
    char *Fa_d, *Xa_d;

    // Host-side initialization
    std::vector<char> Fa(num_vertex, 0);
    std::vector<char> Xa(num_vertex, 0);
    std::vector<int> Ca(num_vertex, INT_MAX);

    Fa[source] = 1;
    Ca[source] = 0;

    // Device memory allocation
    CHECK(cudaMalloc((void**)&Va_d, Va.size() * sizeof(int)));
    CHECK(cudaMalloc((void**)&Ea_d, Ea.size() * sizeof(int)));
    CHECK(cudaMalloc((void**)&Ca_d, num_vertex * sizeof(int)));
    CHECK(cudaMemcpy(Ca_d, Ca.data(), num_vertex * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc((void**)&Fa_d, num_vertex * sizeof(char)));
    CHECK(cudaMemset(Fa_d, 0, num_vertex * sizeof(char)));
    CHECK(cudaMalloc((void**)&Xa_d, num_vertex * sizeof(char)));
    CHECK(cudaMemset(Xa_d, 0, num_vertex * sizeof(char)));

    // Copy data to device
    CHECK(cudaMemcpy(Va_d, Va.data(), Va.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ea_d, Ea.data(), Ea.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Ca_d, Ca.data(), num_vertex * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Fa_d, Fa.data(), num_vertex * sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(Xa_d, Xa.data(), num_vertex * sizeof(char), cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int tpb = 256; // Threads per block
    int bpg = (num_vertex + tpb - 1) / tpb; // Blocks per grid

    bool continue_bfs;
    do {
        continue_bfs = false;
        bfs_ker<<<bpg, tpb>>>(Va_d, Ea_d, Fa_d, Xa_d, Ca_d, num_vertex);
        CHECK_KERNELCALL();
        CHECK(cudaDeviceSynchronize());

        // Copy the Fa array back to the host to check if more processing is needed
        CHECK(cudaMemcpy(Fa.data(), Fa_d, num_vertex * sizeof(char), cudaMemcpyDeviceToHost));

        for (int i = 0; i < num_vertex; ++i) {
            if (Fa[i]) {
                continue_bfs = true;
                break;
            }
        }
    } while (continue_bfs);

    // Copy final distances back to the host
    CHECK(cudaMemcpy(Ca.data(), Ca_d, num_vertex * sizeof(int), cudaMemcpyDeviceToHost));

    // Print results
    for (int i = 0; i < num_vertex; ++i) {
        std::cout << "Vertex " << i + 1 << " Distance: " << Ca[i] << std::endl;
    }

    // Free device memory
    cudaFree(Va_d);
    cudaFree(Ea_d);
    cudaFree(Ca_d);
    cudaFree(Fa_d);
    cudaFree(Xa_d);
}

void read_matrix(std::vector<int> &Va, std::vector<int> &Ea, const std::string &filename, int &num_vertex, int &num_cols, int &num_edges) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("File cannot be opened!");
    }

    // Get matrix dimensions
    file >> num_vertex >> num_cols >> num_edges;
    if (num_vertex <= 0 || num_cols <= 0 || num_edges <= 0) {
        throw std::runtime_error("Invalid matrix dimensions in file!");
    }

    Va.resize(num_vertex + 1, 0);
    Ea.resize(num_edges);

    // Count row occurrences
    std::vector<int> row_counts(num_vertex, 0);
    int row, col;
    float val;

    while (file >> row >> col >> val) {
        row--; // Adjust to 0-based indexing
        row_counts[row]++;
    }

    // Construct Va
    int edge_index = 0;
    for (int i = 0; i < num_vertex; ++i) {
        Va[i] = edge_index;
        edge_index += row_counts[i];
    }
    Va[num_vertex] = num_edges;

    file.clear();
    file.seekg(0);

    file >> num_vertex >> num_cols >> num_edges; // Skip header line

    // Fill Ea
    std::vector<int> row_offsets(num_vertex, 0);
    while (file >> row >> col >> val) {
        row--; col--;
        int position = Va[row] + row_offsets[row];
        Ea[position] = col;
        row_offsets[row]++;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: ./exec matrix_file source\n";
        return EXIT_FAILURE;
    }

    std::vector<int> Va, Ea;
    int num_vertex, num_cols, num_edges;

    const std::string filename(argv[1]);
    int source = atoi(argv[2]) - 1;

    try {
        read_matrix(Va, Ea, filename, num_vertex, num_cols, num_edges);
        bfs_cuda(Va, Ea, source, num_vertex);
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
