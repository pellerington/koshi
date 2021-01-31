//
// Copyright 2020 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
#include "mesh.h"

#include <iostream>
#include "pxr/imaging/hd/mesh.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/imaging/hd/meshUtil.h"
#include "pxr/imaging/hd/smoothNormals.h"
#include "pxr/imaging/hd/vertexAdjacency.h"
#include "renderParam.h"

PXR_NAMESPACE_OPEN_SCOPE

HdKoshiMesh::HdKoshiMesh(Koshi::Scene * scene, SdfPath const& id, SdfPath const& instancerId)
: HdMesh(id, instancerId), scene(scene), geometry(nullptr)
{
}

HdKoshiMesh::~HdKoshiMesh()
{
    scene->removeGeometry(GetId().GetString());
}

HdDirtyBits HdKoshiMesh::GetInitialDirtyBitsMask() const
{
    int mask = HdChangeTracker::Clean
        | HdChangeTracker::InitRepr
        | HdChangeTracker::DirtyPoints
        | HdChangeTracker::DirtyTopology
        | HdChangeTracker::DirtyTransform
        | HdChangeTracker::DirtyVisibility
        | HdChangeTracker::DirtyCullStyle
        | HdChangeTracker::DirtyDoubleSided
        | HdChangeTracker::DirtyDisplayStyle
        | HdChangeTracker::DirtySubdivTags
        | HdChangeTracker::DirtyPrimvar
        | HdChangeTracker::DirtyNormals
        | HdChangeTracker::DirtyInstancer;
    return (HdDirtyBits)mask;
}

HdDirtyBits HdKoshiMesh::_PropagateDirtyBits(HdDirtyBits bits) const
{
    return bits;
}

void HdKoshiMesh::_InitRepr(TfToken const &reprToken, HdDirtyBits *dirtyBits)
{

}

void HdKoshiMesh::Sync(HdSceneDelegate * sceneDelegate, HdRenderParam * renderParam, HdDirtyBits * dirtyBits, const TfToken& reprToken)
{
    // std::cout << "* (multithreaded) Sync Tiny Mesh id=" << GetId() << std::endl;

    const SdfPath& id = GetId();

    // If we don't have a geometry yet create our geometry.
    if(!geometry)
    {
        if(GetPrimvar(sceneDelegate, HdTokens->points).IsEmpty())
            return; // TODO: We should error here.

        geometry = std::make_shared<Koshi::GeometryMesh>();

        GfMatrix4f transform = GfMatrix4f(sceneDelegate->GetTransform(id));
        geometry->setTransform(Koshi::Transform::fromColumnFirstData(transform.data()));

        _MeshReprConfig::DescArray descs = _GetReprDesc(reprToken);
        const HdMeshReprDesc& desc = descs[0];

        HdDisplayStyle const displayStyle = sceneDelegate->GetDisplayStyle(id);
        topology = HdMeshTopology(GetMeshTopology(sceneDelegate), displayStyle.refineLevel);
        HdMeshUtil meshUtil(&topology, GetId());
        meshUtil.ComputeTriangleIndices(&triangulatedIndices, &trianglePrimitiveParams);

        // TODO: We should name vertices the same as HdTokens->points and have a way for the intersector to acccess the right one.
        VtValue value = GetPrimvar(sceneDelegate, HdTokens->points);
        points = value.Get<VtVec3fArray>();
        geometry->setAttribute("vertices", Koshi::Format::FLOAT32, Koshi::GeometryMeshAttribute::VERTICES, points.size(), 3, points.cdata(), triangulatedIndices.size(), 3, (uint32_t*)triangulatedIndices.cdata());

        // Set our constant primvars
        HdPrimvarDescriptorVector constant_primvars = GetPrimvarDescriptors(sceneDelegate, HdInterpolation::HdInterpolationConstant);
        for (const HdPrimvarDescriptor& pv: constant_primvars) 
        {
            // Ignore some primvars
			if (pv.name == HdTokens->points || pv.name.GetString().substr(0, 2) == "__")
				continue;

            primvars.push_back(GetPrimvar(sceneDelegate, pv.name));
            VtValue& value = primvars[primvars.size()-1];

            if(value.IsHolding<VtVec3fArray>())
            {
                // TODO: Should we be storing this array instead of the TfValue?
                const auto& array = value.Get<VtVec3fArray>();
                geometry->setAttribute(pv.name, Koshi::Format::FLOAT32, Koshi::GeometryMeshAttribute::CONSTANT, array.size(), 3, array.cdata(), 0, 0, nullptr);
            }
            if(value.IsHolding<VtVec2fArray>())
            {
                // TODO: Should we be storing this array instead of the TfValue?
                const auto& array = value.Get<VtVec2fArray>();
                geometry->setAttribute(pv.name, Koshi::Format::FLOAT32, Koshi::GeometryMeshAttribute::CONSTANT, array.size(), 2, array.cdata(), 0, 0, nullptr);
            }
	        else if (value.IsHolding<VtFloatArray>())
            {
                // TODO: Should we be storing this array instead of the TfValue?
                const auto& array = value.Get<VtFloatArray>();
                geometry->setAttribute(pv.name, Koshi::Format::FLOAT32, Koshi::GeometryMeshAttribute::CONSTANT, array.size(), 1, array.cdata(), 0, 0, nullptr);
            }
        }

        // Smooth normals required, but not supplied.
        if(!desc.flatShadingEnabled && !geometry->getAttribute(HdTokens->normals.data()))
        {
            Hd_VertexAdjacency adjacency;
            adjacency.BuildAdjacencyTable(&topology);
            normals = Hd_SmoothNormals::ComputeSmoothNormals(&adjacency, points.size(), points.cdata());

            geometry->setAttribute("normals", Koshi::Format::FLOAT32, Koshi::GeometryMeshAttribute::VERTICES, normals.size(), 3, normals.cdata(), triangulatedIndices.size(), 3, (uint32_t*)triangulatedIndices.cdata());
            // TODO: set geomtry normals attribute ID here.
        }

        // 	if (!_primvars.HasNormals() && _smoothNormals)
        // 	{
        // 		/*
        // 			If the topology is dirty, update the adjacency table, a processed
        // 			form of the topology that helps calculate smooth normals quickly.
        // 		*/
        // 		if (dirty_topology)
        // 		{
        // 			_adjacency.BuildAdjacencyTable(&_topology);
        // 		}
        // 		/*
        // 			If the points are dirty, or the topology above changed, update the
        // 			smooth normals.
        // 		*/
        // 		if( dirty_topology || dirty_points )
        // 		{
        // 			const VtVec3fArray &points = _primvars.GetPoints();
        // 			VtVec3fArray normals = Hd_SmoothNormals::ComputeSmoothNormals(
        // 				&_adjacency, points.size(), points.cdata());

        // 			nsi.SetAttribute(_base.Shape(), (
        // 				*NSI::Argument("N")
        // 					.SetType(NSITypeNormal)
        // 					->SetCount(normals.size())
        // 					->SetValuePointer(normals.cdata()),
        // 				*NSI::Argument("N.indices")
        // 					.SetType(NSITypeInteger)
        // 					->SetCount(_faceVertexIndices.size())
        // 					->SetValuePointer(_faceVertexIndices.cdata())));
        // 		}




        scene->addGeometry(GetId().GetString(), geometry.get());
        // scene->addGeometry(GetId().GetAsString(), geometry.get());
    }
}

PXR_NAMESPACE_CLOSE_SCOPE
