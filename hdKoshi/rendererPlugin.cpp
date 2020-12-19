#include "rendererPlugin.h"
#include "renderDelegate.h"

#include "pxr/imaging/hd/rendererPluginRegistry.h"

PXR_NAMESPACE_OPEN_SCOPE

// Register the plugin with the renderer plugin system.
TF_REGISTRY_FUNCTION(TfType)
{
    HdRendererPluginRegistry::Define<HdKoshiRendererPlugin>();
}

HdRenderDelegate * HdKoshiRendererPlugin::CreateRenderDelegate()
{
    return new HdKoshiRenderDelegate();
}

HdRenderDelegate * HdKoshiRendererPlugin::CreateRenderDelegate(const HdRenderSettingsMap& settingsMap)
{
    return new HdKoshiRenderDelegate(settingsMap);
}

void HdKoshiRendererPlugin::DeleteRenderDelegate(HdRenderDelegate * renderDelegate)
{
    // TODO: Should be casted first?
    delete renderDelegate;
}

bool HdKoshiRendererPlugin::IsSupported() const
{
    // We assume if the plugin loads correctly it is supported.
    return true;
}

PXR_NAMESPACE_CLOSE_SCOPE
