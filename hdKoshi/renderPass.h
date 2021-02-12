#pragma once

#include "pxr/pxr.h"
#include "pxr/imaging/hd/renderPass.h"
#include "pxr/imaging/hd/renderThread.h"

#include <koshi/RenderOptix.h>

PXR_NAMESPACE_OPEN_SCOPE

/// \class HdTinyRenderPass
///
/// HdRenderPass represents a single render iteration, rendering a view of the
/// scene (the HdRprimCollection) for a specific viewer (the camera/viewport
/// parameters in HdRenderPassState) to the current draw target.
///
class HdKoshiRenderPass final : public HdRenderPass 
{
public:
    /// Renderpass constructor.
    ///   \param index The render index containing scene data to render.
    ///   \param collection The initial rprim collection for this renderpass.
    HdKoshiRenderPass(Koshi::Scene * scene, HdRenderIndex *index, HdRprimCollection const &collection);

    /// Renderpass destructor.
    virtual ~HdKoshiRenderPass();

protected:

    /// Draw the scene with the bound renderpass state.
    ///   \param renderPassState Input parameters (including viewer parameters)
    ///                          for this renderpass.
    ///   \param renderTags Which rendertags should be drawn this pass.
    void _Execute(HdRenderPassStateSharedPtr const& renderPassState, TfTokenVector const &renderTags) override;

    void CopyPass();

    Koshi::Scene * scene;
    Koshi::RenderOptix render;
    Koshi::Camera camera;

    HdRenderThread renderThread; 
    HdRenderPassAovBindingVector aovBindings;

    Koshi::Vec2u previous_resolution;
    GfMatrix4d previous_view;
    GfMatrix4d previous_proj;
};

PXR_NAMESPACE_CLOSE_SCOPE
