#include "renderPass.h"

#include <iostream>

#include <pxr/imaging/hd/renderPassState.h>
#include <pxr/imaging/hd/renderBuffer.h>

#include <koshi/Vec2.h>

PXR_NAMESPACE_OPEN_SCOPE

HdKoshiRenderPass::HdKoshiRenderPass(Koshi::Scene * scene, HdRenderIndex * index, const HdRprimCollection& collection)
: HdRenderPass(index, collection)
{
    render.setScene(scene);
    std::cout << "Creating renderPass" << std::endl;
}

HdKoshiRenderPass::~HdKoshiRenderPass()
{
    std::cout << "Destroying renderPass" << std::endl;
}

void
HdKoshiRenderPass::_Execute(const HdRenderPassStateSharedPtr& renderPassState, const TfTokenVector& renderTags)
{
    render.reset();

    GfVec4f vp = renderPassState->GetViewport();
    const Koshi::Vec2u resolution(vp[2], vp[3]);
    GfMatrix4d view = renderPassState->GetWorldToViewMatrix();
    GfMatrix4d proj = renderPassState->GetProjectionMatrix();

    Koshi::Camera camera(resolution, Koshi::Transform::fromData(view.data(), false), Koshi::Transform::fromData(proj.data(), false));
    render.setCamera(&camera);
    render.addAov("color", 4);

    render.start();

    Koshi::Aov * color_aov = render.getAov("color");
    const HdRenderPassAovBinding * hd_color_aov;
    HdRenderPassAovBindingVector aovBindings = renderPassState->GetAovBindings();
    for(auto aov = aovBindings.begin(); aov != aovBindings.end(); ++aov)
        if(aov->aovName == "color")
            hd_color_aov = &(*aov);
    color_aov->copyBuffer(hd_color_aov->renderBuffer->Map(), Koshi::Format::UINT8);
    hd_color_aov->renderBuffer->Unmap();

    std::cout << "=> Execute RenderPass" << std::endl;
}

PXR_NAMESPACE_CLOSE_SCOPE
