#include <algorithm>
#include <cmath>
#include <vector>


static const int GS = 32;
struct Config{
    int dim,hidden,n_layers,n_heads,n_kv_heads,head_dim,vocab,max_seq,group;
    float eps;
};
struct QTensor{
     std::vector<uint8_t> q;
     std::vector<float> sk;

};

struct Layer{
    std::vector<float> atttn_norn,ffn_norm;
    QTensor  wq,qk,wv,wg,wu,wd;
};

struct Model{
    Config c;
    std::vector<float> inv_freq;
    QTensor tok;
    std::vector<Layer> layers;
    std::vector<float> final_norm;

};

