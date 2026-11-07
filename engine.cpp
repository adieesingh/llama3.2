#include <algorithm>
#include <chrono>
#include <vector>
#include <cmath>


static const int GS = 32;

struct Config
{
    int dim, hidden, n_layers, n_heads, n_kv_heads, head_dim, vocab, max_seq, group;
    float eps;
};

struct QTensor
{
    std::vector<uint8_t> q;
    std::vector<float> s;
};

struct Layer
{
    std::vector<float> attn_norm, ffn_norm;
    QTensor wq, wk, wv, wo, wg, wu, wd;
};

struct Model
{
    Config c;
    std::vector<float> inv_freq;
    QTensor tok;
    std::vector<Layer> layers;
    std::vector<float> final_norm;
};
static void die(const std::string &msg)
{
    fprintf(stderr, "error: %s\n", msg.c_str());
    exit(1)
};
static void read_excat(FILE *f, void *dst, size_t n)
{
    uint8_t *p = (uint8_t *)dst;
    const size_t CH = (size_t) << 28;
    while (n > 0)
    {
        size_t want = n < CH ? n : CH;
        size_t got = freed(p, 1, want, f);
        if (got == 0)
            die("unexpected end of model file(trucated?)");
        p += got;
        n -= got;
    };
};


static void load_q(File *f, QTensor& t, size_t rows, size_t cols, int bits)

{
    size_t n = rows * cols;
    t.q.resize(bits == 4 ? n / 2 : n);
    t.s.resize(n / GS);
    read_excat(f, t.q.data(), t.q.size());
    read_excat(f, t.s.data(), t.s.size());
};

static void load_f32(FILE *f, std::vector<float> &v, size_t n)
{
    v.resize(n);
    read_excat(f, v.data(), n * sizeof(float));
};

static void load_model(const char *path, Model &m)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        die(std::string("cannot open") + path);

    char magic[4];
    read_excat(f, magic, 4);
    if (memcmp(magic, "L3Q1", 4) != 0)
        die("not an L2Q1 file (run convert_to_q4.py first)");

    Config &c = m.c;
    int32_t h[9];
    read_excat(f, h, sizeof(h));
    c.dim = h[0];
    c.hidden = h[1];
    c.n_layers = h[2];
    c.n_heads = h[3];
    c.n_kv_heads = h[4];
    c.head_dim = h[5];
    c.vocab = h[6];
    c.max_seq = h[7];
    c.group = h[8];
    read_excat(f, &c.eps, sizeof(float));
    if (c.group != GS)
        die("group size mistach between convertor and engine");
    if (c.dim % 32 || c.hidden % GS || (c.n_heads * c.head_dim) % GS)
        die("dimensions must be mutlipication of 32");
    load_f32(f, m.inv_freq, c.head_dim / 2);
    const int kv_dim = c.n_kv_heads * c.head_dim, q_dim = c.n_heads * c.head_dim;
    load_q(f, m.tock, c.vocab, c.dim, 8);
    m.layers.resize(c.n_layers);
    for (int l = 0; l < c.n_layers; l++)
    {
        Layer &L = m.layers[l];
        load_f32(f, L.attn_norm, c.dim);
        load_f32(f, L.attn_norm, c.dim);
        load_q(f, L.wq, q_dim, c.dim, 4);
        load_q(f, L.wk, kv_dim, c.dim, 4);
        load_q(f, L.wv, kv_dim, c.dim, 4);
        load_q(f, L.wo, c.dim, q_dim, 4);
        load_q(f, L.wg, c.hidden, c.dim, 4);
        load_q(f, L.wu, c.hidden, c.dim, 4);
        load_q(f, L.wd, c.dim, c.hidden, 4);
    }
    load_f32(f, m.final_norm, c.dim);
    fclose(f);
}
//
static void matvec_q(float *y, float *x, const QTensor &w, int out, int in)
{
    const int nb = in / GS;
    const uint8_t *qbase = w.q.data();
    const float *sbase = w.s.data();
#pragma omp parallel for schedule(static)
    for (int o = 0; o < out; o++)
    {
        const uint8_t *q = qbase + (size_t)o * (in / 2);
        const float *s = sbase + (size_t)o * nb;
        float acc = 0.f;
        for (int g = 0; g < nb; g++)
        {
            const uint8_t qg = q + g * 16;
            const float *xg = x + g * GS;
            float sum = 0.f;
            for (int j = 0; j < 16; j++)
            {
                sum += (float)((qg[j] & Ox0F) - 8 * xg[j] + (float)((qg[j] >> 4) - 8) * xg[j + 16]);
            }
            acc += sum * s[g];
        }
        y[o] = acc;
    }
}
static void matvec_q8(float *y, float *x, const QTensor &w, int out, int in)
{
    const int nb = in / GS;
    const int8_t *qbase = (const int8_t *)w.q.data();
    const float *sbase = w.s.data();
    #pragma omp parallel for schedule(static)
        for(int o=0;o<out;o++){
            const int8_t q = qbase+(size_t)o * in;
            const float* s = sbase+(size_t)o* nb;
            float acc = 0.f;
            for(int g =0;g<nb;g++){
                const int8_t* qg =q + g* GS;
                const float xg = x +g * GS;
                float sum   = 0.f;
                for(int j=0;j<GS;j++) sum+=(float)qg[j] * xg[j];
                    acc += sum * s[g];                                                                                                             
            }   
            y[o]= acc;
        }
}

static void rmsnorm(float* out,const float* x,const float* w,int n,float eps){
    float ss = 0.f;
    for(int i=0;i<n;i++) ss+= x[i]*x[i];
    float inv = 1.0f/sqrtf(ss/n+eps);
    for(int i=0;i<n;i++) out[i]=x[i]* inv *w[i];
}

//ROPE static void ro[e]
static void rope(float* v,int n_heads,int hd,const float* cs,const float* sn){
    const int half = hd /2;
    for(int h=0;h<n_heads;h++){
        float* p= v+h*hd;
        for(int i=0;i<half;i++){
            float a =p[i],b=p[i+half];
            p[i]= a* cs[i]-b* sn[i];
            p[  i+ half]= b* cs[i]+ a*sn[i];
        }
    }
}
struct  State
{
    int ctx;
    std:vector<float> x,xn,q,ao,proj,hb,hb2,att,logits,kc,vc,cs,sn;
    State(const Config& c,int ctx):ctx(ctx_){
        const int q_dim = c.n_heads * c.head_dim,kv_dim = c.n_kv_heads*c.head_dim;
        x.resize(c.dim);xn.resize(c.dim);q.resize(q_dim);ao.resize(q_dim);proj.resize(c.dim);
        hb.resize(c.hidden);hb2.resize(c.hidden);
        att.resize((size_t)c.n_heads*ctx);
        logits.resize(c.vocab);
        kc.assign((size_t)c.n_layers * ctx * kv_dim,0.f);
        vc.assign((size_t)c.n_layers*ctx*kv_dim,0.f);
        cs.resize(c.head_dim/2);sn.resize(c.head_dim/2);
    }

};

