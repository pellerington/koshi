#include "renderPass.h"

#include <iostream>

#include <pxr/imaging/hd/renderPassState.h>
#include "renderBuffer.h"

#include <koshi/math/Vec2.h>

PXR_NAMESPACE_OPEN_SCOPE

HdKoshiRenderPass::HdKoshiRenderPass(Koshi::Scene * scene, HdRenderThread * render_thread, HdRenderIndex * index, const HdRprimCollection& collection)
: HdRenderPass(index, collection), scene(scene), render_thread(render_thread)
{
    render.setScene(scene);
    render_thread->SetRenderCallback(std::bind(&HdKoshiRenderPass::Render, this));
    render_thread->StartThread();
    std::cout << "Creating renderPass" << std::endl;
}

HdKoshiRenderPass::~HdKoshiRenderPass()
{
    render_thread->StopThread();
    std::cout << "Destroying renderPass" << std::endl;
}

void
HdKoshiRenderPass::_Execute(const HdRenderPassStateSharedPtr& renderPassState, const TfTokenVector& renderTags)
{
    bool restart = false;

    // restart = scene->isDirty() || restart;

    const CameraUtilFraming &framing = renderPassState->GetFraming();
    const Koshi::Vec2u resolution(framing.dataWindow.GetWidth(), framing.dataWindow.GetHeight());

    GfMatrix4d view = renderPassState->GetWorldToViewMatrix();
    GfMatrix4d proj = renderPassState->GetProjectionMatrix();
    restart = (resolution != previous_resolution) || restart;
    restart = (view != previous_view) || restart;
    restart = (proj != previous_proj) || restart;

    HdRenderPassAovBindingVector new_aov_bindings = renderPassState->GetAovBindings();
    restart = (aov_bindings != new_aov_bindings) || restart;

    if(!restart) return;

    render_thread->StopRender();
    render.reset();

    previous_resolution = resolution;
    previous_view = view;
    previous_proj = proj;

    aov_bindings = new_aov_bindings;

    // scene->updateDirtiness();
    
    // // ADD A INTERSECTOR(?) AND A LIST OF GEOMTRY (AND CAMERA?) TO SCENE SO THEY WILL WORK WITH DIRTINESS
    
    // scene->preRender(); // ???

    camera = Koshi::Camera(resolution, Koshi::Transform::fromColumnFirstData(view.data()), Koshi::Transform::fromColumnFirstData(proj.data()));
    render.setCamera(&camera);

    for(auto aov = aov_bindings.begin(); aov != aov_bindings.end(); ++aov)
    {
        render.addAov(aov->aovName, 4);
        aov->renderBuffer->Allocate(GfVec3i(resolution.x, resolution.y, 1), HdFormatFloat32Vec4, true);
    }

    // Start the background render thread.
    render.start();
    render_thread->StartRender();
}

void
HdKoshiRenderPass::Render()
{
    const uint max_passes = 1024;

    while(!render_thread->IsStopRequested() /* && !restartRequested? && AND RENDER NOT CONVERGED... */ && render.getPasses() < max_passes)
    {
        while(render_thread->IsPauseRequested())
            std::this_thread::sleep_for(std::chrono::milliseconds(64));

        render.pass();

        for(auto aov = aov_bindings.begin(); aov != aov_bindings.end(); ++aov)
        {
            Koshi::Aov * koshi_aov = render.getAov(aov->aovName);
            koshi_aov->copy(aov->renderBuffer->Map(), render.getPasses()/* TODO: Replace this with the samples aov */);
            ((HdKoshiRenderBuffer*)aov->renderBuffer)->SetConverged();
            aov->renderBuffer->Unmap();
        }

        for(auto aov = aov_bindings.begin(); aov != aov_bindings.end(); ++aov)
            ((HdKoshiRenderBuffer*)aov->renderBuffer)->SetConverged();
    }

    /* SET AOVs as CONVERGED */
}

PXR_NAMESPACE_CLOSE_SCOPE