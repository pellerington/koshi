#include "renderPass.h"

#include <iostream>

#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/renderBuffer.h>

#include <koshi/math/Vec2.h>

PXR_NAMESPACE_OPEN_SCOPE

HdKoshiRenderPass::HdKoshiRenderPass(Koshi::Scene * scene, HdRenderIndex * index, const HdRprimCollection& collection)
: HdRenderPass(index, collection), scene(scene)
{
    render.setScene(scene);
    renderThread.SetRenderCallback(std::bind(&HdKoshiRenderPass::CopyPass, this));
    renderThread.StartThread();
    std::cout << "Creating renderPass" << std::endl;
}

HdKoshiRenderPass::~HdKoshiRenderPass()
{
    renderThread.StopThread();
    std::cout << "Destroying renderPass" << std::endl;
}

void
HdKoshiRenderPass::_Execute(const HdRenderPassStateSharedPtr& renderPassState, const TfTokenVector& renderTags)
{
    bool restart = false;

    // restart = scene->isDirty() || restart;

    // TODO: Use the proper camera delegate...
    GfVec4f vp = renderPassState->GetViewport();
    const Koshi::Vec2u resolution(vp[2], vp[3]);
    GfMatrix4d view = renderPassState->GetWorldToViewMatrix();
    GfMatrix4d proj = renderPassState->GetProjectionMatrix();

    restart = (resolution != previous_resolution) || restart;
    restart = (view != previous_view) || restart;
    restart = (proj != previous_proj) || restart;

    // CHECK AOVS...

    if(!restart) return;

    std::cout << "Restarting Render..." << std::endl;

    previous_resolution = resolution;
    previous_view = view;
    previous_proj = proj;

    renderThread.StopRender();
    render.reset();

    // scene->updateDirtiness();
    
    // // ADD A INTERSECTOR(?) AND A LIST OF GEOMTRY (AND CAMERA?) TO SCENE SO THEY WILL WORK WITH DIRTINESS
    
    // scene->preRender(); // ???

    camera = Koshi::Camera(resolution, Koshi::Transform::fromColumnFirstData(view.data()), Koshi::Transform::fromColumnFirstData(proj.data()));
    render.setCamera(&camera);
    render.addAov("color", 4);

    aovBindings = renderPassState->GetAovBindings();

    const HdRenderPassAovBinding * hd_color_aov;
    for(auto aov = aovBindings.begin(); aov != aovBindings.end(); ++aov)
        if(aov->aovName == "color")
            hd_color_aov = &(*aov);
    hd_color_aov->renderBuffer->Allocate(GfVec3i(resolution.x, resolution.y, 1), HdFormatFloat32Vec4, /*multiSampled=*/true);

    // Start the background render thread.
    render.start();
    renderThread.StartRender();

    std::cout << "=> Execute RenderPass" << std::endl;
}

void
HdKoshiRenderPass::CopyPass()
{
    while(!renderThread.IsPauseRequested() && !renderThread.IsStopRequested())
    {
        render.pass();

        Koshi::Aov * color_aov = render.getAov("color");
        const HdRenderPassAovBinding * hd_color_aov;
        for(auto aov = aovBindings.begin(); aov != aovBindings.end(); ++aov)
            if(aov->aovName == "color")
                hd_color_aov = &(*aov);
        color_aov->copyBuffer(hd_color_aov->renderBuffer->Map(), Koshi::Format::FLOAT32, render.sample);
        hd_color_aov->renderBuffer->Unmap();
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
