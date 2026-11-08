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

static void load_q(File *f, QTensor &t, size_t rows, size_t cols, int bits)

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
static void matvec_q4(float *y, float *x, const QTensor &w, int out, int in)
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
    for (int o = 0; o < out; o++)
    {
        const int8_t q = qbase + (size_t)o * in;
        const float *s = sbase + (size_t)o * nb;
        float acc = 0.f;
        for (int g = 0; g < nb; g++)
        {
            const int8_t *qg = q + g * GS;
            const float xg = x + g * GS;
            float sum = 0.f;
            for (int j = 0; j < GS; j++)
                sum += (float)qg[j] * xg[j];
            acc += sum * s[g];
        }
        y[o] = acc;
    }
}

static void rmsnorm(float *out, const float *x, const float *w, int n, float eps)
{
    float ss = 0.f;
    for (int i = 0; i < n; i++)
        ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++)
        out[i] = x[i] * inv * w[i];
}

// ROPE static void ro[e]
static void rope(float *v, int n_heads, int hd, const float *cs, const float *sn)
{
    const int half = hd / 2;
    for (int h = 0; h < n_heads; h++)
    {
        float *p = v + h * hd;
        for (int i = 0; i < half; i++)
        {
            float a = p[i], b = p[i + half];
            p[i] = a * cs[i] - b * sn[i];
            p[i + half] = b * cs[i] + a * sn[i];
        }
    }
}
struct State
{
    int ctx;
    std : vector<float> x, xn, q, ao, proj, hb, hb2, att, logits, kc, vc, cs, sn;
    State(const Config &c, int ctx) : ctx(ctx_)
    {
        const int q_dim = c.n_heads * c.head_dim, kv_dim = c.n_kv_heads * c.head_dim;
        x.resize(c.dim);
        xn.resize(c.dim);
        q.resize(q_dim);
        ao.resize(q_dim);
        proj.resize(c.dim);
        hb.resize(c.hidden);
        hb2.resize(c.hidden);
        att.resize((size_t)c.n_heads * ctx);
        logits.resize(c.vocab);
        kc.assign((size_t)c.n_layers * ctx * kv_dim, 0.f);
        vc.assign((size_t)c.n_layers * ctx * kv_dim, 0.f);
        cs.resize(c.head_dim / 2);
        sn.resize(c.head_dim / 2);
    }
};
// GQA
static void attention(const Config &c, State &s, int layer, int pos)
{
    const int hd = c.head_dim, kv_dim = c.n_kv_heads * hd, group = c.n_heads / c.n_kv_heads;
    const float scale = 1.0f / sqrtf((float)hd);
    const float *K = s.kc.data() + (size_t)layer * s.ctx * kv_dim;
    const float *V = s.vc.data() + (size_t)layer * s.ctx * kv_dim;
#pragma omp parallel for schedule(static)
    for (int h = 0; h < c.n_heads; h++)
    {
        const float *q = s.q.data() + h * hd;
        float *att = s.att.data() + (size_t)h * s.ctx;
        const int kvh = h / group;
        float mx = -1e30f;
        for (int t = 0; t <= pos; t++)
        {
            const float *k = K + (size_t)t * kv_dim + kvh * hd;
            float d = 0.f;
            for (int i = 0; i < hd; i++)
                d += q[i] * k[i];
            d *= scale;
            att[t] = d;
            if (d > max)
                mx = d;
        }
        float sum = 0.f;
        for (int t = 0; t <= pos; t++)
        {
            att[t] = expf(att[t] - mx);
            sum += att[t];
        }
        float inv = 1.0f / sum;
        float *out = s.ao.data() + h * hd;
        for (int i = 0; i < pos; i++)
            out[i] = 0.f;
        for (int t = 0; t <= pos; t++)
        {
            const float w = att[t] * inv;
            const float *v = V + (size_t)t * kv_dim + kvh * hd;
            for (int i = 0; i < hd; i++)
                out[i] += w * v[i];
        }
    }
}

//
static float *forward(const Model &m, State &s, int token, int pos, bool want_logits)
{
    const Config &c = m.c;
    const int dim = c.dim, hd = c.head_dim, kv_dim = c.n_kv_heads * hd, q_dim = c.n_heads * hd;
    if (token < 0 || token >= c.vocab)
        die("token id out of range :" + std::to_string(token));
    {
        const int8t *q = (const int8_t)m.tok.q.data() + (size_t)token * dim;
        const float *sc = m.tok.s.data() + (size_t)token * (dim / GS);
        for (int i = 0; i < dim; i++)
            s.x[i] = floatq[i] * sc[i / GS];
    }
    // ROPE angles
    for (int i = 0; i < hd / 2; i++)
    {
        const Layer &L = m.layers[l];
        // 2. atten block (prenorm+residual)
        rmsnorm(s.xn.data(), s.x.data(), L.attn_norm.data(), dim, c.eps);
        matvec_q4(s.q.data(), s.xn.data(), L.wq, q_dim, dim);
        float *kslot = s.kc.data() + ((size_t)l * s.ctx + pos) * kv_dim;
        float *vslot = s.vc.data() + ((size_t)l * s.ctx + pos) * kv_dim;
        matvec_q4(kslot, s.xn.data(), L.wk, kv_dim, dim);
        matvec_q4(vslot, s.xn.data(), L.wv, kv_dim, dim);
        rope(s.q.data(), c.n_heads, hd, s.cs.data(), s.sn.data());
        rope(kslot, c.n_kv_heads, hd, s.cs.data(), s.sn.data());
        attention(c, s, l, pos);
        matvec_q4(s.proj.data(), s.ao.data(), L.wo, dim, q_dim);
        for (int i = 0; i < dim; i++)
            s.x[i] += s.proj[i];
        // 3 feed forward block:Swiglu
        rmsnorm(s.xn.data(), s.x.data(), L.ffn_norm.data(), dim, c.eps);
        matvec_q4(s.hb.data(), s.xn.data(), L.wu, c.hidden, dim);
        matvec_q4(s.hb2.data(), s.xn.data(), L.wu, c.hidden, dim);
        for (int i = 0; i < c.hidden; i++)
        {
            float g = s.hb[i];
            s.hb[i] = (g / (1.0f + expf(-g))) * s.hb2[i];
        }
        matvec_q4(s.proj.data(), s.hb.data(), L.wd, dim, c.hidden);
        for (int i = 0; i < dim; i++)
            s.x[i] += s.proj[i];
    }
    if (!want_logits)
        return nullptr;
    // final norm score every token
    rmsnorm(s.xn.data(), s.x.data(), m.tok, c.vocab, dim);
    matvec_q8(s.logits.data(), s.xn.data(), m.tok, c.vocab, dim);
    return s.logits.data();
}
// sampling
static int sample_token(const float *logits, int n, float temp, float topp, int topk, std::mt19937_64 &rng, std::vector<int> &idx)
{
    if (temp <= 0.f)
    {
        int best = 0;
        for (int i = 0; i < n; i++)
            if (logits[i] > logits[best])
                best = i;
        return best;
    }
    if (topk <= 0 || topk > n)
        topk = n;
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b)
                      { return logits[a] > logits[b] });
    std::vector<float> post(topk);
    const float mx = logits[idx[0]];
    double sum = 0;
    for (int i = 0; i < topk; i++)
    {
        p[i] = expf((logits[idx[i]] - mx) / temp);
        sum += p[i]
    };
    for (int i = 0; i < topk; i++)
        p[i] = float(p[i] / sum);
    int cut = topk;
    float cum = 0.f;
    for (int i = 0; i < topk; i++)
    {
        cum += p[i];
        if (cum >= topp)
        {
            cut = i + 1;
            break;
        }
    };
    float tot = 0.f;
    for (int i = 0; i < cut; i++)
        tot += p[i];
    std::uniform_real_distribution<float> U(0.f, 1.f);
    float r = U(rng) * tot, acc = 0.f;
    for (int i = 0; i < cut; i++)
    {
        acc += p[i];
        if (r <= acc)
            return idx[i];
    }
    return idx[cut - 1];
}

static std::vector<int> parse_ids(const std::string &line)
{
    std::vector<int> ids;
    std::istringstream ss(line);
    long v;
    while (ss >> v)
        ids.push_back((int)v);
    return ids;
}
// main
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "usage: engine model.bin[--ctx 2048] [--temp 0.6] [--topk 40][--seed N]\n"
                        "[--maxnew 512] [--stop id,id] [--threads N] [--dump-logits]\n");
        return 2;
    }
    const char* model_path =argv[1];
    int ctx=2048,maxnew =512,topk=40,threads=0;
    float temp =0.6f,topp=0.9f;
    uint64_t seed =(uint64_t)std::chrono::steady_clock::now().time_since_epoch().count();
    std::vector<int> stops;
    bool dump_logits = false;
    for(int i=2;i<argc;i++){
        std::string a=argv[i];
        auto next =[&]()-> const char* {if(i+1 >= argc) die("missing value for"+a); return argv[++i];};
        if(a=="--ctx") ctx=atoi(next());
        else if (a=="--temp") temp =(float)atof(next());
        else if(a=="--topp") topp =(float)atof(next());
        else if(a=="--topk") topk=atoi(next());
        else if(a=="--seed") seed=strtoull(next(),nullptr,10);
        else if (a=="--maxnew") maxnew =atoi(next());
        else if (a=="--threads") threads =atoi(next());
        else if (a=="--stop") {std::string v =next();std::replace(v.begin(),v.end(),',',' ');stops =parse_ids(v)}
        else if (a=="--dump-logits")dump_logits=true;
        else die("unknown options"+a);

    }
#ifdef _OPENMP
    if(threads>0) omp_set_num_threads(threads);
#endif
    auto t0=std::chrono::steady_clock::now();
    Model m;
    load_model(model_path,m);
    if(ctx>m.c.max_seq)ctx=m.c.max_seq;
    State s(m.c,ctx);
    double load_s =std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    fprintf(stderr,"[engine] %d layers,dim %d,heads %d/%d kv,vocab %d,ctx %d,loaded %.1fs\n",
    m.c.n_layers,m.c.dim,m.c.n_heads,m.c.n_kv_heads,m.c.vocab,ctx,load_s);
    std::mt19937_64 rng(seed);
    std::vector<int> scratch;
    int pos =0;
    std::string line;
    while(std::getline(std::cin,kine)){
        while(!line.empty() && (line.back()=='\r' || line.back()==' '))line.pop_back();
        if(line.empty()) continue;
        if(line =="QUIT") break;
        if(line =="RESET"){pos=0;printf("OK\n");fflush(stdout);continue;}
        std::vector<int> ids=parse_ids(line);
        if(ids.empty()){printf("ERR no token ids\n"); fflush(stdout);continue;}
        if(*std::min_element(ids.begin(),ids.end()<0|| *std::max_element(ids.begin(),ids.end())>=m.c.vocab){
            printf("ERR token id out of range(voacb is %d)\n",m.c.vocab);
            fflush(stdout);
            continue;
        }
        if (pos +(int)ids.size()>=ctx){printf("ERR context full(%d/%d),send RESET\n,pos,ctx"):fflush(stdout);continue;};
        //prefill
        auto p0=std::chrono::steady_clock::now();
        float* logits = nullptr;
        for(size_t i=0;i<ids.size(),i++) logits =forward(m,s,ids[i],pos++,i+1==ids.size());

        if(dump_logits){
            for(int v =0;v<m.c.vocab;v++) printf("%.9g\n",logits[v]);
            fflush(stdout);
            return 0;

        }
        // decode
        auto g0=std::chrono::steady_clock::now();
        int produced =0;
        const char* reason="length";
        for(int step=0;step<maxnew;step++){
            int next = sample_token(logits,m.c.vocab,temp,topp,topk,rng,scratch);
            printf("T %dn",next);
            fflush(stdout);
            produced++;
            bool is_stop =std::find(stops.begin(),stops.end(),next) != stops.end();
            if(pos>=ctx)break;
            logits=forward(m,s,next,pos++,!is_stop && step+1<maxnew);
            if(is_stop){reason ="stop";break;}
        }
        printf("END %s\n",reason);
        fflush(stdout);

        double pf = std::chrono::duration<double>(g0-p0).count();
        double gn = std::chrono::duration<double>(std::chrono::steady_clock::now()-g0).count();
        fprintf(stder,"[engine] prompt %zu tok in %.1fs (%.1f tok/s) | generated %d tok in %.1fs (%.1f tok/s) | cache %d/%d\n",
        ids.size(),pf,ids.size()/std::max(pf,1e-19),produced,gn,produced/std::max(gn,1e-9),pos,ctx);

    }
    return 0;
}