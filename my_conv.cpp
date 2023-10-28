#include<stdio.h>
#include "miopen.hpp"
#include "tensor.hpp"
#include "naive_algo.hpp"


class EventTimer{
    public:
        EventTimer(){}
        ~EventTimer(){
            reset();
        }
        void reset(){
            if(event_created){
                hipEventDestroy(start_ev);
                hipEventDestroy(stop_ev);
                event_created=false;
            }
        }

        void start(){
            reset();
            CHECK_HIP(hipEventCreate(&start_ev));
            CHECK_HIP(hipEventCreate(&stop_ev));
            event_created=true;
            CHECK_HIP(hipDeviceSynchronize());
            CHECK_HIP(hipEventRecord( start_ev, queue));
        }

        double elapsed(){
            float elapsed_time = 0.0;
            CHECK_HIP(hipEventRecord(stop_ev,queue));
            CHECK_HIP(hipEventSynchronize(stop_ev));
            CHECK_HIP(hipEventElapsedTime(&elapsed_time, start_ev, stop_ev));
            return (double)elapsed_time;
        }

    private:
        hipStream_t queue = NULL;
        hipEvent_t start_ev,stop_ev;
        bool event_created=false;
};



int main(int argc, char *argv[]) {
    if(argc != 12){
        printf("usage: ./my_conv [N] [C] [H] [W] [k] [f_h] [f_w] [s] [p] [d] [g]\n");
        exit(0);
    }
    // paraments
    int N = atoi(argv[1]);
    int C = atoi(argv[2]);
    int H = atoi(argv[3]);
    int W = atoi(argv[4]);
    int k = atoi(argv[5]);
    int h = atoi(argv[6]);
    int w = atoi(argv[7]);
    int s = atoi(argv[8]);
    int p = atoi(argv[9]);
    int d = atoi(argv[10]);
    int g = atoi(argv[11]);
    assert((g >= 1) && (C % g == 0) && (k % g == 0));
    
    int pads[2]           = {p, p};
    int conv_strides[2]   = {s, s};
    int conv_dilations[2] = {d, d};
    size_t group_count = g;
    int c,out_c,out_w, out_h;
    c = C/g;
    int iter = 100;


    device_init();
    miopenEnableProfiling(mio::handle(), true);    // 用来统计time——enabled等信息
    // NCHW
    Tensor input(N, C, H,W);        
    Tensor weights(k, c, h, w);    
    Tensor dinput(N, C, H,W); 
    Tensor dweights(k, c, h, w);
    size_t input_lens = N*C*H*W;
    size_t weight_lens = k*c*h*w;

    // XX_c data on host to cpu verify, XX_h from device2host of miopen API
    float *input_c=(float *)malloc(input_lens*sizeof(float));
    float *weights_c=(float *)malloc(weight_lens*sizeof(float));
    float *dinput_c=(float *)malloc(input_lens*sizeof(float));
    float *dinput_h=(float *)malloc(input_lens*sizeof(float));
    float *dweight_c=(float *)malloc(weight_lens*sizeof(float));
    float *dweight_h=(float *)malloc(weight_lens*sizeof(float));

    // initialize data
    miopenConvolutionDescriptor_t conv_desc;
    miopenStatus_t rc;

    rand_float(input_c, input_lens, 0, 1);
    // for(int i=0;i<100;i=i+5)
    //     printf("%f ",input_c[i]);
    // printf("\n");
    rand_float(weights_c, weight_lens, 0, 1);
    input.initData_fromCPU(input_c,input_lens);
    // std::vector<float> a = input.toHost();
    //   for(int i=0;i<100;i=i+5)
    //     printf("%f ",a[i]);
    weights.initData_fromCPU(weights_c,weight_lens);
    
    
    // Create the convolution descriptor
    miopenCreateConvolutionDescriptor(&conv_desc);
    miopenConvolutionMode_t mode_t = miopenConvolution;
    // miopenInitConvolutionDescriptor(conv_desc, mode_t, 0, 0, 1, 1, 1,1);
    miopenInitConvolutionNdDescriptor(conv_desc, 2, pads,conv_strides,conv_dilations,mode_t);
    miopenSetConvolutionGroupCount(conv_desc,group_count);  

    // Get the convolution output dimensions
    miopenGetConvolutionForwardOutputDim(conv_desc, input.desc, weights.desc, &N,&out_c, &out_h, &out_w);

    Tensor output = Tensor(N, out_c, out_h, out_w);
    Tensor doutput = Tensor(N, out_c, out_h, out_w);

    size_t output_lens = N*out_c*out_h*out_w;
    float *output_c=(float *)malloc(output_lens*sizeof(float));
    float *output_h=(float *)malloc(output_lens*sizeof(float));
    float *doutput_c=(float *)malloc(output_lens*sizeof(float));
    float *doutput_h=(float *)malloc(output_lens*sizeof(float));
    rand_float(doutput_c,output_lens,0,1);
    doutput.initData_fromCPU(doutput_c,output_lens);

    EventTimer timer; 

    printf("Stage1: fwd \n");
    // Find-db model
    /*
    size_t find_fwd_ws = 0;
    int ret_algo_count;
    int request_algo_count = 2;
    std::vector<miopenConvAlgoPerf_t> perf_results(request_algo_count);
    void *buf;
    miopenConvolutionForwardGetWorkSpaceSize(mio::handle(),weights.desc,input.desc,conv_desc,output.desc,&find_fwd_ws);
    if(find_fwd_ws)
        hipMalloc((void**)(&buf),1*find_fwd_ws);
    miopenFindConvolutionForwardAlgorithm(mio::handle(),input.desc,input.data,weights.desc,weights.data,conv_desc,
                                            output.desc,output.data,request_algo_count,&ret_algo_count,perf_results.data(),find_fwd_ws?buf:nullptr,find_fwd_ws,1);
    float alpha = 1;
    float beta = 0;
    miopenConvolutionForward(mio::handle(),&alpha,input.desc,input.data,weights.desc,weights.data,conv_desc,perf_results[0].fwd_algo,
                            &beta,output.desc,output.data,find_fwd_ws?buf:nullptr,perf_results[0].memory);
    if(find_fwd_ws)hipFree(buf);
    */

    // Immediate model
    std::size_t count;
    rc = miopenConvolutionForwardGetSolutionCount(mio::handle(),weights.desc,input.desc,conv_desc,output.desc,&count);
   if(rc != miopenStatusSuccess)
        printf(" Failed in getting convolution solution Count!!!");
    if(count < 1)
        printf("No convolution solution!!!");
    auto solutions = std::vector<miopenConvSolution_t>(count);
    rc = miopenConvolutionForwardGetSolution(mio::handle(),weights.desc,input.desc,conv_desc,output.desc,count,&count,solutions.data());
    if(rc != miopenStatusSuccess)
        printf(" Failed in getting convolution solution!!!");
    for (size_t i = 0; i < solutions.size(); i++) {
        std::string algorithmName = GetConvolutionAlgorithmName(solutions[i].algorithm);
        printf("Solution Algorithm: %s", algorithmName.c_str());
        printf("Solution id: %lu\n", solutions[i].solution_id);
    }
    printf("Defualt select solution id is %d\n",85);
    solutions.resize(count);
    std::size_t ws_size;
    rc = miopenConvolutionForwardGetSolutionWorkspaceSize(mio::handle(),weights.desc,input.desc,conv_desc,output.desc,85,&ws_size); // solution_id of  85 is direct_fwd algorithm, set 1 of auto select fast algo
    if(rc != miopenStatusSuccess)
        printf(" Failed in  convolution solution  workspace!!!");
    rc = miopenConvolutionForwardCompileSolution(mio::handle(),weights.desc,input.desc,conv_desc,output.desc,85);
    if(rc != miopenStatusSuccess)
        printf(" Failed in Compile convolution solution!!!");
    void *buf;  // select other algo mybe need workspase
    if(ws_size>0)
        hipMalloc((void**)(&buf),1*ws_size);
    // warmup
    rc = miopenConvolutionForwardImmediate(mio::handle(), weights.desc,weights.data,input.desc,input.data,conv_desc,output.desc,output.data,ws_size?buf:nullptr,ws_size,85);
    if(rc != miopenStatusSuccess)
        printf(" Failed in  convolution solution Immediate !!!");

    timer.start();
    for(int i=0;i<iter;i++){
        miopenConvolutionForwardImmediate(mio::handle(), weights.desc,weights.data,input.desc,input.data,conv_desc,output.desc,output.data,ws_size?buf:nullptr,ws_size,85);
    }
    double T = timer.elapsed()/iter;
    if(ws_size>0)hipFree(buf);
    
    // cpu verify
    hipMemcpy(output_h,output.data,output_lens*sizeof(float),hipMemcpyDeviceToHost);
    printf("Miopen:          ");
    for(int i=0;i<(output_lens>50?50:output_lens);i=i+5)
        printf("%f ",output_h[i]);
    printf("\n");
    printf("Naive_conv_fwd:  ");
    fwd_naive(input_c,weights_c,output_c,N,W,H,C,k,w,h,pads[0],pads[1],conv_strides[0],conv_strides[1],conv_dilations[0],conv_dilations[1],group_count);
    for(int i=0;i<(output_lens>50?50:output_lens);i=i+5)
        printf("%f ",output_c[i]);
    printf("\n");
    Check_Result(output_h,output_c,output_lens);
    
    size_t Flops_fwd = 2L*N*out_w*out_h*k*w*h*C / group_count;  
    //printf(" %zu",(N*out_w*out_h*k)*(2*(w*h*C)-1));// [ 乘、加、合并]
    // size_t flopCnt = 2L * in_n * in_c * wei_h * wei_w * out_c * out_h * out_w / groups;   fwd
    printf("Conv Forward Flops: %zu, time: %f , GFlops: %.0f \n", Flops_fwd,T,((double)Flops_fwd)/T*1e-6);
    

    printf("Stage2: bwd \n");
    rc = miopenConvolutionBackwardDataGetSolutionCount(mio::handle(),doutput.desc,weights.desc,conv_desc,dinput.desc,&count);
    if(rc != miopenStatusSuccess)
        printf(" Failed in getting convolution solution Count!!!");
    auto bwd_solutions = std::vector<miopenConvSolution_t>(count);
    rc = miopenConvolutionBackwardDataGetSolution(mio::handle(), doutput.desc,weights.desc,conv_desc, dinput.desc, count, &count, bwd_solutions.data());
    for (int i = 0; i < bwd_solutions.size(); i++) {
        std::string algorithmName = GetConvolutionAlgorithmName(bwd_solutions[i].algorithm);
        printf("Solution Algorithm: %s", algorithmName.c_str());
        printf("Solution id: %lu\n", bwd_solutions[i].solution_id);
    }
    printf("Defualt select solution id is %d\n",86);
    rc = miopenConvolutionBackwardDataGetSolutionWorkspaceSize(mio::handle(), doutput.desc,weights.desc,conv_desc, dinput.desc, 86, &ws_size);
    if(rc != miopenStatusSuccess)
        printf(" Failed in getting convolution solution workspace!!!");
    rc = miopenConvolutionBackwardDataCompileSolution(mio::handle(), doutput.desc,weights.desc,conv_desc, dinput.desc, 86);
    if(rc != miopenStatusSuccess)
        printf(" Failed in getting convolution compile solution!!!");
    timer.start();
    for (int i = 0; i < iter; i++)
    {
        miopenConvolutionBackwardDataImmediate(mio::handle(), doutput.desc,doutput.data,weights.desc,weights.data,conv_desc,dinput.desc,dinput.data, nullptr,ws_size,86);
    }
    T = timer.elapsed()/iter;
    // if(rc != miopenStatusSuccess)
    //     printf(" Failed in  convolution executing!!!")
    printf("Conv BWD Flops(approximate): %zu, time: %f , GFlops: %.0f \n", Flops_fwd,T,((double)Flops_fwd)/T*1e-6);
    
    // Find-Db model
    /*
    ret_algo_count = 0;
    request_algo_count = 2;
    size_t find_bwd_ws = 0;
    miopenConvolutionBackwardDataGetWorkSpaceSize(mio::handle(),doutput.desc,weights.desc,conv_desc,dinput.desc,&find_bwd_ws);
    miopenFindConvolutionBackwardDataAlgorithm(mio::handle(),doutput.desc,doutput.data,weights.desc,weights.data,conv_desc,dinput.desc,dinput.data,
                                                request_algo_count,&ret_algo_count,perf_results.data(),nullptr,find_bwd_ws,1);
    alpha=1;
    beta=0;
    miopenConvolutionBackwardData(mio::handle(),&alpha,doutput.desc,doutput.data,weights.desc,weights.data,conv_desc,perf_results[0].bwd_data_algo,
                                    &beta,dinput.desc,dinput.data,nullptr,perf_results[0].memory);                                           
    */

    hipMemcpy(dinput_h,dinput.data,input_lens*sizeof(float),hipMemcpyDeviceToHost);
    printf("Miopen:          ");
    for(int i=0;i<(input_lens>50?50:input_lens);i=i+5)
        printf("%f ",dinput_h[i]);
    printf("\n");
    bwd_naive(doutput_c,weights_c,dinput_c,N,W,H,C,k,w,h,pads[0],pads[1],conv_strides[0],conv_strides[1],conv_dilations[0],conv_dilations[1],group_count);
    printf("Naive_conv_bwd:  ");
    for(int i=0;i<(input_lens>50?50:input_lens);i=i+5)
        printf("%f ",dinput_c[i]);
    printf("\n");
    Check_Result(dinput_h,dinput_c,input_lens);
    
    printf("Stage3: wrw \n");
    // Immediate model
    auto wrw_solutions = std::vector<miopenConvSolution_t>(count);
    miopenConvolutionBackwardWeightsGetSolutionCount(mio::handle(), doutput.desc, input.desc, conv_desc, dweights.desc, &count);
    miopenConvolutionBackwardWeightsGetSolution(mio::handle(), doutput.desc, input.desc, conv_desc, dweights.desc, count, &count, wrw_solutions.data());
    for (int i = 0; i < wrw_solutions.size(); i++) {
        std::string algorithmName = GetConvolutionAlgorithmName(wrw_solutions[i].algorithm);
        printf("Solution Algorithm: %s", algorithmName.c_str());
        printf("Solution id: %lu\n", wrw_solutions[i].solution_id);
    }
    printf("Defualt select solution id is %d\n",87);
    miopenConvolutionBackwardWeightsGetSolutionWorkspaceSize(mio::handle(), doutput.desc, input.desc, conv_desc, dweights.desc,87, &ws_size);
    miopenConvolutionBackwardWeightsCompileSolution(mio::handle(), doutput.desc, input.desc, conv_desc, dweights.desc, 87);
    timer.start();
    for(int i=0;i<iter;i++){
        miopenConvolutionBackwardWeightsImmediate(mio::handle(), doutput.desc,doutput.data,input.desc,input.data,conv_desc,dweights.desc,dweights.data,nullptr,ws_size,87);
    }
    T = timer.elapsed()/iter;
    
    // Find-Db model 
    /* 
    ret_algo_count = 0;
    request_algo_count = 2;
    size_t find_wrw_ws = 0;
    miopenConvolutionBackwardWeightsGetWorkSpaceSize(mio::handle(),doutput.desc,input.desc,conv_desc,dweights.desc,&find_wrw_ws);
    miopenFindConvolutionBackwardWeightsAlgorithm(mio::handle(),doutput.desc,doutput.data,input.desc,input.data,conv_desc,dweights.desc,dweights.data,
                                                request_algo_count,&ret_algo_count,perf_results.data(),nullptr,find_wrw_ws,1);
    alpha=1;
    beta=0;
    miopenConvolutionBackwardWeights(mio::handle(),&alpha,doutput.desc,doutput.data,input.desc,input.data,conv_desc,perf_results[0].bwd_weights_algo,
                                    &beta,dweights.desc,dweights.data,nullptr,perf_results[0].memory);                                           

    */
    hipMemcpy(dweight_h,dweights.data,weight_lens*sizeof(float),hipMemcpyDeviceToHost);
    printf("Miopen:          ");
    for(int i=0;i<(weight_lens>50?50:weight_lens);i=i+5)
        printf("%f ",dweight_h[i]);
    printf("\n");
    wrw_naive(doutput_c,input_c,dweight_c,N,W,H,C,k,w,h,pads[0],pads[1],conv_strides[0],conv_strides[1],conv_dilations[0],conv_dilations[1],group_count);
    printf("Naive_conv_wrw:  ");
    for(int i=0;i<(weight_lens>50?50:weight_lens);i=i+5)
        printf("%f ",dweight_c[i]);
    printf("\n");
    // check_result(dweight_h,dweight_c,weight_lens,h);
    Check_Result(dweight_c,dweight_h,weight_lens,true);
    printf("Conv WRW Flops(approximate): %zu, time: %f , GFlops: %.0f \n", Flops_fwd,T,((double)Flops_fwd)/T*1e-6);
    
    // Cleanup
    timer.reset();
    miopenDestroyConvolutionDescriptor(conv_desc);
    free(input_c);
    free(output_h);
    free(output_c);
    free(doutput_c);
    free(doutput_h);
    free(weights_c);
    free(dinput_c);
    free(dinput_h);
    free(dweight_c);
    free(dweight_h);

    return 0;
}
