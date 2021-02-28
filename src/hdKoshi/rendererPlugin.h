#pragma once

#include "pxr/pxr.h"
#include "pxr/imaging/hd/rendererPlugin.h"

PXR_NAMESPACE_OPEN_SCOPE

class HdKoshiRendererPlugin final : public HdRendererPlugin 
{
public:
    HdKoshiRendererPlugin() = default;
    virtual ~HdKoshiRendererPlugin() = default;

    virtual HdRenderDelegate * CreateRenderDelegate() override;
    virtual HdRenderDelegate * CreateRenderDelegate(const HdRenderSettingsMap& settingsMap) override;

    virtual void DeleteRenderDelegate(HdRenderDelegate * renderDelegate) override;

    virtual bool IsSupported() const override;

private:
    // This class does not support copying.
    HdKoshiRendererPlugin(const HdKoshiRendererPlugin&) = delete;
    HdKoshiRendererPlugin &operator =(const HdKoshiRendererPlugin&) = delete;
};

PXR_NAMESPACE_CLOSE_SCOPE