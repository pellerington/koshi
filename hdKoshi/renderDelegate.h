#pragma once

#include "pxr/pxr.h"
#include "pxr/imaging/hd/renderDelegate.h"
#include "pxr/imaging/hd/resourceRegistry.h"
#include "pxr/base/tf/staticTokens.h"

#include "renderParam.h"

#include <koshi/Scene.h>

PXR_NAMESPACE_OPEN_SCOPE

class HdKoshiRenderDelegate final : public HdRenderDelegate 
{
public:
    /// Render delegate constructor. 
    HdKoshiRenderDelegate();
    /// Render delegate constructor. 
    HdKoshiRenderDelegate(const HdRenderSettingsMap& settingsMap);
    /// Render delegate destructor.
    virtual ~HdKoshiRenderDelegate();

    /// Supported types
    const TfTokenVector& GetSupportedRprimTypes() const override;
    const TfTokenVector& GetSupportedSprimTypes() const override;
    const TfTokenVector& GetSupportedBprimTypes() const override;

    // Basic value to return from the RD
    HdResourceRegistrySharedPtr GetResourceRegistry() const override;

    // Prims
    HdRenderPassSharedPtr CreateRenderPass(HdRenderIndex *index, HdRprimCollection const& collection) override;

    HdInstancer * CreateInstancer(HdSceneDelegate *delegate, SdfPath const& id, SdfPath const& instancerId) override;
    void DestroyInstancer(HdInstancer *instancer) override;

    HdRprim * CreateRprim(TfToken const& typeId, SdfPath const& rprimId, SdfPath const& instancerId) override;
    void DestroyRprim(HdRprim * rPrim) override;

    HdSprim * CreateSprim(TfToken const& typeId, SdfPath const& sprimId) override;
    HdSprim * CreateFallbackSprim(TfToken const& typeId) override;
    void DestroySprim(HdSprim * sprim) override;

    HdBprim * CreateBprim(TfToken const& typeId, SdfPath const& bprimId) override;
    HdBprim *  CreateFallbackBprim(TfToken const& typeId) override;
    void DestroyBprim(HdBprim * bprim) override;

    HdAovDescriptor GetDefaultAovDescriptor(const TfToken& name) const;

    void CommitResources(HdChangeTracker * tracker) override;

    HdRenderParam * GetRenderParam() const override;

private:
    static const TfTokenVector SUPPORTED_RPRIM_TYPES;
    static const TfTokenVector SUPPORTED_SPRIM_TYPES;
    static const TfTokenVector SUPPORTED_BPRIM_TYPES;

    Koshi::Scene scene;
    std::shared_ptr<HdKoshiRenderParam> param;

    void _Initialize();

    HdResourceRegistrySharedPtr _resourceRegistry;

    // This class does not support copying.
    HdKoshiRenderDelegate(const HdKoshiRenderDelegate &) = delete;
    HdKoshiRenderDelegate &operator =(const HdKoshiRenderDelegate &) = delete;
};


PXR_NAMESPACE_CLOSE_SCOPE