#include <koshi/RenderOptix.h>

#include <iostream>
#include <fstream>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <koshi/OptixHelpers.h>
#include <koshi/Aov.h>

#include <koshi/GeometryMesh.h>

KOSHI_OPEN_NAMESPACE

RenderOptix::RenderOptix()
: scene(nullptr), camera(nullptr), intersector(nullptr), aovs(std::vector<Aov>())
{
    // Set render variables to nullptr
    resources.scene = nullptr;
    resources.camera = nullptr;
    resources.intersector = nullptr;
    resources.random = nullptr;
    resources.aovs = nullptr;
    d_resources = 0;
    sbt.raygenRecord = 0;
    sbt.missRecordBase = 0;
    sbt.hitgroupRecordBase = 0;
    
    // Initialize CUDA and create OptiX context
    {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));
        CUcontext cuCtx = 0;  // zero means take the current context
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        static auto context_log_cb = [](unsigned int level, const char* tag, const char* message, void* cbdata)
        {
            std::cout << "[" << level << "][" << tag << "]: " << message << "\n";
        };
        options.logCallbackFunction       = context_log_cb;
        options.logCallbackLevel          = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
    }

    char log[2048]; // For error reporting from OptiX creation functions
    size_t sizeof_log = sizeof(log);

    // Create module
    OptixPipelineCompileOptions pipeline_compile_options = {};
    {
        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount     = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel             = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel           = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        pipeline_compile_options.usesMotionBlur        = false;
        pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipeline_compile_options.numPayloadValues      = 2;
        pipeline_compile_options.numAttributeValues    = 3;

        pipeline_compile_options.exceptionFlags        = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options.pipelineLaunchParamsVariableName = "resources";

        std::ifstream ptx_file("/home/peter/KoshiPU/plugin/ptx/optix_functions.ptx");
        const std::string ptx((std::istreambuf_iterator<char>(ptx_file)), (std::istreambuf_iterator<char>()));

        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(context, &module_compile_options, &pipeline_compile_options, ptx.c_str(), ptx.size(), log, &sizeof_log, &module));
    }

    // Create program groups, including NULL miss and hitgroups
    {
        OptixProgramGroupOptions program_group_options   = {}; // Initialize to zeros

        OptixProgramGroupDesc raygen_prog_group_desc  = {};
        raygen_prog_group_desc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_prog_group_desc.raygen.module            = module;
        raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(context, &raygen_prog_group_desc, 1 /*num program groups*/, &program_group_options, log, &sizeof_log, &raygen_prog_group) );

        OptixProgramGroupDesc miss_prog_group_desc  = {};
        miss_prog_group_desc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_prog_group_desc.miss.module            = module;
        miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(context, &miss_prog_group_desc, 1 /*num program groups*/, &program_group_options, log, &sizeof_log, &miss_prog_group) );

        OptixProgramGroupDesc hitgroup_prog_group_desc = {};
        hitgroup_prog_group_desc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_prog_group_desc.hitgroup.moduleCH            = module;
        hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK_LOG( optixProgramGroupCreate(context, &hitgroup_prog_group_desc, 1 /*num program groups*/, &program_group_options, log, &sizeof_log, &hitgroup_prog_group) );
    }

    // Link pipeline
    {
        const uint32_t    max_trace_depth  = 1;
        OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth          = max_trace_depth;
        pipeline_link_options.debugLevel             = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        OPTIX_CHECK_LOG(optixPipelineCreate(context, &pipeline_compile_options, &pipeline_link_options, program_groups, sizeof(program_groups) / sizeof(program_groups[0]), log, &sizeof_log, &pipeline));

        OptixStackSizes stack_sizes = {};
        for( auto& prog_group : program_groups )
        {
            OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
        }

        uint32_t direct_callable_stack_size_from_traversal;
        uint32_t direct_callable_stack_size_from_state;
        uint32_t continuation_stack_size;
        OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0 /*maxCCDepth*/, 0 /*maxDCDEpth*/, &direct_callable_stack_size_from_traversal, &direct_callable_stack_size_from_state, &continuation_stack_size));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal, direct_callable_stack_size_from_state, continuation_stack_size, 1/*maxTraversableDepth*/));
    }
}

void RenderOptix::setScene(Scene * _scene)
{
    scene = _scene;
}

void RenderOptix::setCamera(Camera * _camera)
{
    // if(camera->getResolution() != _camera->getResolution())
    // {
        // TODO: Loop through AOVS and change resolution if it's different;
    // }
    camera = _camera;
}

Aov * RenderOptix::getAov(const std::string& name)
{
    for(uint i = 0; i < aovs.size(); i++)
        if(aovs[i].name == name)
            return &(aovs[i]);
    return nullptr;
}

Aov * RenderOptix::addAov(const std::string& name, const uint& channels)
{
    if(!camera) return nullptr; // TODO: Should error here
    for(uint i = 0; i < aovs.size(); i++)
        if(aovs[i].name == name /* && aov[i]->channels == channels */)
            return &aovs[i];
    aovs.emplace_back(name, camera->getResolution(), channels);
    return &aovs[aovs.size()-1];
}

void RenderOptix::reset()
{
    // TODO: Replace sample with sample aov
    sample = 0;

    // Free SBT.
    CUDA_FREE(sbt.raygenRecord);
    CUDA_FREE(sbt.missRecordBase);
    CUDA_FREE(sbt.hitgroupRecordBase);

    // Free device resources.
    CUDA_FREE(resources.camera);
    CUDA_FREE(resources.intersector);
    CUDA_FREE(resources.aovs);
    CUDA_FREE(resources.scene);
    CUDA_FREE(resources.random);
    CUDA_FREE(d_resources);

    // TODO: We shouldn't remake all aovs and camera on reset. if we are just updating an objects translation ect. hydra needs some way of checking aovs against it's list.
    aovs.clear();
    camera = nullptr;
}

void RenderOptix::start()
{
    if(!camera || !scene) 
        return; // TODO: Should error here

    if(!intersector)
        intersector = new IntersectorOptix(scene, context);

    // TODO: For device objects. Malloc Once and the Memcpy to update.

    // Copy Camera to Device.
    CUDA_CHECK(cudaMalloc(&resources.camera, sizeof(Camera)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(resources.camera), camera, sizeof(Camera), cudaMemcpyHostToDevice));

    // Copy Intersector to Device.
    CUDA_CHECK(cudaMalloc(&resources.intersector, sizeof(Intersector)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(resources.intersector), intersector, sizeof(Intersector), cudaMemcpyHostToDevice));

    // Copy Aovs to Device.
    resources.aovs_size = aovs.size();
    CUDA_CHECK(cudaMalloc(&resources.aovs, sizeof(Aov) * aovs.size()));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(resources.aovs), aovs.data(), sizeof(Aov) * aovs.size(), cudaMemcpyHostToDevice));

    // Copy Scene to Device.
    device_scene.init(scene);
    CUDA_CHECK(cudaMalloc(&resources.scene, sizeof(DeviceScene)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(resources.scene), &device_scene, sizeof(DeviceScene), cudaMemcpyHostToDevice));

    // Copy Random to Device
    random.init(camera->getResolution());
    CUDA_CHECK(cudaMalloc(&resources.random, sizeof(Random)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(resources.random), &random, sizeof(Random), cudaMemcpyHostToDevice));

    // Copy Resrouces to device
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_resources), sizeof(Resources)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_resources), &resources, sizeof(Resources), cudaMemcpyHostToDevice));

    // Set up shader binding table
    CUdeviceptr  raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size ));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t      miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
    RayGenSbtRecord ms_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t      hitgroup_record_size = device_scene.num_geometries * sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc( reinterpret_cast<void**>(&hitgroup_record), hitgroup_record_size));
    std::vector<HitGroupSbtRecord> hg_sbt(device_scene.num_geometries);
    for(uint i = 0; i < hg_sbt.size(); i++)
    {
        hg_sbt[i].data.geometry_id = i;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt[i]));
    }
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(hitgroup_record), hg_sbt.data(), hitgroup_record_size, cudaMemcpyHostToDevice));

    sbt = {};
    sbt.raygenRecord                = raygen_record;
    sbt.missRecordBase              = miss_record;
    sbt.missRecordStrideInBytes     = sizeof( MissSbtRecord );
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupSbtRecord );
    sbt.hitgroupRecordCount         = 2;
}

void RenderOptix::pass()
{
    random.setSeeds(sample, 0);

    // Perform the actual intersection.
    OPTIX_CHECK(optixLaunch(pipeline, 0, d_resources, sizeof(Resources), &sbt, camera->getResolution().x, camera->getResolution().y, /*depth=*/1));
    CUDA_SYNC_CHECK();

    for(uint i = 0; i < aovs.size(); i++)
        aovs[i].transferDeviceBuffer();

    sample++;
}

RenderOptix::~RenderOptix()
{
    // Cleanup
    std::cout << "Deleteing Render" << "\n";
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));

    // Free SBT.
    CUDA_FREE(sbt.raygenRecord);
    CUDA_FREE(sbt.missRecordBase);
    CUDA_FREE(sbt.hitgroupRecordBase);

    // Free device resources.
    CUDA_FREE(resources.camera);
    CUDA_FREE(resources.intersector);
    CUDA_FREE(resources.aovs);
    CUDA_FREE(resources.scene);
    CUDA_FREE(resources.random);
    CUDA_FREE(d_resources);
}

KOSHI_CLOSE_NAMESPACE