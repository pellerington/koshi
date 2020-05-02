#pragma once

#include <string>
#include <fstream>
#include <json.hpp>

#include <Math/Types.h>
#include <base/Scene.h>
#include <base/Settings.h>
#include <base/Object.h>
#include <base/ObjectGroup.h>

class AttributeAccessor;

struct Type 
{
    Type(const std::string& name) : name(name) {}
    std::string name;
    std::vector<std::string> reserved_attributes;
    // TODO: Add default attributes somehow. For integrators and intersectors.
    typedef Object * (CreateObjectCallback)(AttributeAccessor& accessor, Object * parent);
    CreateObjectCallback * create_object_cb = nullptr;
};

class Types
{
public:
    void add(const Type& type) { types.insert(std::make_pair(type.name, type)); }
    Type * get(const std::string& name)
    { 
        auto type = types.find(name);
        return (type != types.end()) ? &type->second : nullptr;
    }
private:
    std::unordered_map<std::string, Type> types;
};

class SceneFile
{
public:

    SceneFile();
    void Import(const std::string& filename, Scene& scene, /*Render*/Settings& settings);
    Object * CreateObject(const std::string& name, nlohmann::json& json_object, nlohmann::json& scene_file, Scene& scene, Object * parent = nullptr);

private:
    Types types;

    static Object * create_object(AttributeAccessor& accessor, Object * parent);
    static  Object * create_object_group(AttributeAccessor& accessor, Object * parent);
};

class AttributeAccessor
{
public:
    AttributeAccessor(const std::string& object_name, nlohmann::json& json, nlohmann::json& scene_file_json, Scene& scene, SceneFile * scene_file)
    : object_name(object_name), json(json), scene_file_json(scene_file_json), scene(scene), scene_file(scene_file) {}

    inline Object * get_object(const std::string& name, Object * parent = nullptr) {
        Object * attr_object = nullptr;
        auto attr = json[name];
        if(attr.is_string())
        {
            const std::string attr_name = attr;
            // Try and find the object.
            attr_object = scene.get_object(attr_name);
            if(!attr_object) attr_object = scene_file->CreateObject(attr_name, scene_file_json[attr_name], scene_file_json, scene);
        }
        else if(attr.is_object())
        {
            attr_object = scene_file->CreateObject(object_name + "." + name, attr, scene_file_json, scene, parent);
        }
        return attr_object;
    }

    inline ObjectGroup * get_objects(const std::string& name, Object * parent = nullptr) {
        std::vector<std::string> objects = get_strings(name);
        ObjectGroup * object_group = new ObjectGroup;
        for(uint i = 0; i < objects.size(); i++)
        {
            // Try and find the object.
            Object * object = scene.get_object(objects[i]);
            if(!object) 
                object = scene_file->CreateObject(objects[i], scene_file_json[objects[i]], scene_file_json, scene);
            if(object) object_group->push(object);
        }
        scene.add_object(name, object_group);
        return object_group;
    }

    inline void add_object(const std::string& name, Object * object) {
        scene.add_object(object_name + "." + name, object);
    }

    inline bool get_bool(const std::string& name, const bool& unknown = false) {
        return (json[name].is_boolean()) ? (bool)json[name] : unknown;
    }

    inline float get_float(const std::string& name, const float& unknown = 0.f) {
        return json[name].is_number() ? (float)json[name] : unknown;
    }

    inline uint get_uint(const std::string& name, const uint& unknown = 0) {
        return json[name].is_number() ? (uint)json[name] : unknown;
    }

    inline std::string get_string(const std::string& name, const std::string& unknown = "") {
        return json[name].is_string() ? (std::string)json[name] : unknown;
    }

    inline Vec3f get_vec3f(const std::string& name, const float& unknown = 0.f) {
        if(!json[name].is_array()) return Vec3f(unknown);
        Vec3f v;
        for(int i = 0; i < 3; i++)
            v[i] = json[name][i].is_number() ? (float)json[name][i] : unknown;
        return v;
    }

    inline Vec2i get_vec2i(const std::string& name, const int& unknown = 0) {
        if(!json[name].is_array()) return Vec2i(unknown);
        Vec2i v;
        for(int i = 0; i < 2; i++)
            v[i] = json[name][i].is_number() ? (int)json[name][i] : unknown;
        return v;
    }

    inline Vec2u get_vec2u(const std::string& name, const uint& unknown = 0) {
        if(!json[name].is_array()) return Vec2u(unknown);
        Vec2u v;
        for(int i = 0; i < 2; i++)
            v[i] = json[name][i].is_number() ? (int)json[name][i] : unknown;
        return v;
    }

    inline std::vector<std::string> get_strings(const std::string& name) {
        std::vector<std::string> strings;
        if(!json[name].is_array()) return strings;
        for(auto it = json[name].begin(); it != json[name].end(); ++it)
            if(it->is_string())
                strings.push_back(*it);
        return strings;
    }

private:
    std::string object_name;
    nlohmann::json& json;
    nlohmann::json& scene_file_json;
    Scene& scene;
    SceneFile * scene_file;
};

    //     // if(scene_file["volumes"].is_array())
    //     // {
    //     //     for (auto it = scene_file["volumes"].begin(); it != scene_file["volumes"].end(); ++it)
    //     //     {
    //     //         if((*it)["type"].is_string() && (*it)["name"].is_string())
    //     //         {
    //     //             if((*it)["type"] == "volume")
    //     //             {
    //     //                 const float density = get_float(*it, "density");
    //     //                 const Vec3f transparency = get_vec3f(*it, "transparency");
    //     //                 const Vec3f density_gain = Vec3f::clamp(VEC3F_ONES-transparency, 0.f, 1.f)*density;
    //     //                 Texture * density_texture = ((*it)["density_texture"].is_string()) ? textures[(*it)["density_texture"]] : nullptr;

    //     //                 const float g = get_float(*it, "anistropy");
    //     //                 const Vec3f scattering = get_vec3f(*it, "scattering");
    //     //                 const Vec3f emission = get_vec3f(*it, "emission");

    //     //                 std::shared_ptr<Volume> volume = new Volume(density_gain, density_texture, scattering, g, emission));
    //     //                 volumes[(*it)["name"]] = volume;
    //     //             }
    //     //         }
    //     //     }
    //     // }