
static inline size_t naive_conv_out_size(size_t in_size, size_t pad,size_t dilation, size_t ksize,size_t stride) {
    return (in_size + 2 * pad - dilation * (ksize - 1) - 1) / stride + 1;
}

void fwd_naive(const float *input_x, const float *filter,float *output_y, size_t n, size_t w, size_t h,
                size_t c, size_t k, size_t fx, size_t fy,size_t px, size_t py, size_t sx,size_t sy, 
                size_t dx, size_t dy, size_t group){
        
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
    size_t ig, in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;
    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ik = 0; ik < k_per_group; ik++) {
                for (ioh = 0; ioh < oh; ioh++) {
                    for (iow = 0; iow < ow; iow++) {
                        // sliding window for this filter
                        double value = .0f;
                        o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow + ioh * ow + iow;
                        for (ic = 0; ic < c_per_group; ic++) {
                            for (ir = 0; ir < fy; ir++) {
                                cur_h = sy * ioh - py + dy * ir;
                                if (cur_h < 0 || cur_h >= h)
                                    continue;
                                for (is = 0; is < fx; is++) {
                                    cur_w = sx * iow - px + dx * is;
                                    if (cur_w < 0 || cur_w >= w)
                                        continue;
                                    i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w +
                                            cur_h * w + cur_w;
                                    f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx +
                                            ir * fx + is;
                                    value += static_cast<double>(input_x[i_idx]) * static_cast<double>(filter[f_idx]);
                                }
                            }
                        }
                        output_y[o_idx] = static_cast<float>(value);
                    }
                }
            }
        }
    }
}

// implictGEMM
/*
#define OFFSET(n,c,h,w,C,H,W) (((n*C+c)*H+h)*W+w)
void yshun(const float *input_x , const float *filter, float *output_y, int N ,int C ,int H ,
                            int W ,int k ,int h ,int w,int oh, int ow){
    int K =k;
    int P = oh;
    int Q = ow;
    int stride_h=2;
    int stride_w=2;
    int R = h;
    int S = w;
    for(int n=0;n<N;++n){
        for(int k=0;k<K;++k){
            for(int p=0;p<P;++p){
                for(int q=0;q<Q;++q){
                    float temp=0;
                    int h=p*stride_h;
                    int w=q*stride_w;
                    for(int c=0;c<C;++c){
                        for(int r=0;r<R;++r){
                            for(int s=0;s<S;++s){
                                temp+=filter[OFFSET(k,c,r,s,C,R,S)]*input_x[OFFSET(n,c,h+r,w+s,C,H,W)];
                            }
                        }
                    }
                    output_y[OFFSET(n,k,p,q,K,P,Q)]=temp;
                }
            }
        }
    }
}
*/

void bwd_naive(const float *output_grad, const float *filter,float *input_grad, size_t n, size_t w, size_t h,
                size_t c, size_t k, size_t fx, size_t fy,size_t px, size_t py, size_t sx,size_t sy, 
                size_t dx, size_t dy, size_t group){
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;

    size_t ig, in, ik, ih, iw, ic, is, ir;
    size_t cur_oh, cur_ow, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (in = 0; in < n; in++) {
            for (ic = 0; ic < c_per_group; ic++) {
                for (ih = 0; ih < h; ih++) {
                    for (iw = 0; iw < w; iw++) {
                        double value = .0f;
                        i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w + ih * w + iw;
                        for (ik = 0; ik < k_per_group; ik++) {
                            for (ir = 0; ir < fy; ir++) {
                                cur_oh = ih + py - dy * ir; 
                                if (cur_oh < 0 || cur_oh % sy)
                                    continue;
                                cur_oh /= sy;
                                if (cur_oh >= oh)
                                    continue;
                                for (is = 0; is < fx; is++) {
                                    cur_ow = iw + px - dx * is; 
                                    if (cur_ow < 0 || cur_ow % sx)
                                        continue;
                                    cur_ow /= sx;
                                    if (cur_ow >= ow)
                                        continue;
                                    o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow +cur_oh * ow + cur_ow;
                                    f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx +ir * fx + is;
                                    value += static_cast<double>(output_grad[o_idx]) * static_cast<double>(filter[f_idx]);
                                }
                            }
                        }
                        input_grad[i_idx] = static_cast<float>(value);
                    }
                }
            }
        }
    }
}

void wrw_naive(const float *output_grad, const float *input_x,float *filter_grad, size_t n, size_t w, size_t h,
                size_t c, size_t k, size_t fx, size_t fy,size_t px, size_t py, size_t sx,size_t sy, 
                size_t dx, size_t dy, size_t group){
    size_t oh = naive_conv_out_size(h, py, dy, fy, sy);
    size_t ow = naive_conv_out_size(w, px, dx, fx, sx);
    size_t k_per_group = k / group;
    size_t c_per_group = c / group;
    size_t ig, in, ik, ioh, iow, ic, is, ir;
    size_t cur_h, cur_w, o_idx, i_idx, f_idx;

    for (ig = 0; ig < group; ig++) {
        for (ik = 0; ik < k_per_group; ik++) {
            for (ic = 0; ic < c_per_group; ic++) {
                for (ir = 0; ir < fy; ir++) {
                    for (is = 0; is < fx; is++) {
                        double value = .0f;
                        f_idx = ig * k_per_group * c_per_group * fy * fx + ik * c_per_group * fy * fx + ic * fy * fx + ir * fx + is;
                        for (in = 0; in < n; in++) {
                            for (ioh = 0; ioh < oh; ioh++) {
                                cur_h = sy * ioh - py + dy * ir;
                                if (cur_h < 0 || cur_h >= h)
                                    continue;
                                for (iow = 0; iow < ow; iow++) {
                                    cur_w = sx * iow - px + dx * is;
                                    if (cur_w < 0 || cur_w >= w)
                                        continue;
                                    i_idx = in * c * h * w + ig * c_per_group * h * w + ic * h * w + cur_h * w + cur_w;
                                    o_idx = in * k * oh * ow + ig * k_per_group * oh * ow + ik * oh * ow + ioh * ow + iow;
                                    value += static_cast<double>(input_x[i_idx]) * static_cast<double>(output_grad[o_idx]);
                                }
                            }
                        }
                        filter_grad[f_idx] = static_cast<float>(value);
                    }
                }
            }
        }
    }

}







static void Check_Result(const float *data_miopen,const float *data_cpu, size_t data_lens, bool is_wrw = false){
    double tolerance = 1.5e-6;
    if(is_wrw)
        tolerance = 1.5e-6 * 10;
    double square_difference = 0.0;
    double mag1 = -9999;
    double mag2 = -9999;
    for(size_t i = 0; i<data_lens;i++){
        square_difference +=(data_miopen[i]-data_cpu[i])*(data_miopen[i]-data_cpu[i]);
        if(std::fabs(data_miopen[i])>mag1)mag1=std::fabs(data_miopen[i]);
        if(std::fabs(data_cpu[i]>mag2))mag2=std::fabs(data_cpu[i]);
    }
    double mag = std::max({std::fabs(mag1),std::fabs(mag2),std::numeric_limits<double>::min()});
    double rms = std::sqrt(square_difference)/(std::sqrt(data_lens)*mag);
    if(!std::isnan(rms) && rms>tolerance)
        printf("Result Failed!!! (error %f)\n",rms);
    else
        printf("Result PASS!!! \n");
}