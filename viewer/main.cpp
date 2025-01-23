#include <iostream>
#include <MyRender/Scene.h>
#include <MyRender/MainLoop.h>
#include <MyRender/NavigationCamera.h>
#include <MyRender/RenderMesh.h>
#include <MyRender/utils/PrimitivesFactory.h>
#include <MyRender/Window.h>
#include <MyRender/shaders/Shader.h>
#include <MyRender/shaders/ShaderProgramLoader.h>
#include <SdfLib/utils/Mesh.h>
#include <SdfLib/OctreeSdf.h>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <imgui.h>
#include <ImGuizmo.h>
#include <args.hxx>

class UiManager : public myrender::System
{
public:
    void addSlider(myrender::Shader& shader, const std::string& propName,
                   const std::string& showName,
                   float startValue,
                   float min, float max)
    {
        UiSliderProperty prop { propName, showName, &shader, min, max, startValue };
        mUiSliderProperties.push_back(prop);
    }

    void drawGui() override
    {
        ImGui::Spacing();
        ImGui::Separator();
        ImGui::Text((systemName == "") ? "RenderMesh" : systemName.c_str());

        for(UiSliderProperty& sprop : mUiSliderProperties)
        {
            bool change = ImGui::SliderFloat(sprop.showName.c_str(), &sprop.value, sprop.min, sprop.max);
            sprop.shader->setUniform(sprop.propName, sprop.value);
        }
    }

private:
    struct UiSliderProperty
    {
        std::string propName;
        std::string showName;
        myrender::Shader* shader;
        float min;
        float max;
        float value;
    };

    std::vector<UiSliderProperty> mUiSliderProperties;
};


class RenderSdf : public myrender::System
{
public:
    RenderSdf(const std::string& computeShaderName) : mComputeShaderName(computeShaderName) {}
    void start() override
    {
        mShader = std::make_unique<myrender::Shader>(myrender::Shader::loadShader(mComputeShaderName));

        // CreateTexture
        {
            glGenTextures(1, &mRenderTexture);
            glBindTexture(GL_TEXTURE_2D, mRenderTexture);
            // set the texture wrapping/filtering options (on the currently bound texture object)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

            mRenderTextureSize = myrender::Window::getCurrentWindow().getWindowSize();
            std::vector<uint32_t> colorImage(mRenderTextureSize.x * mRenderTextureSize.y);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mRenderTextureSize.x, mRenderTextureSize.y, 0, GL_RGBA, GL_FLOAT, NULL);

            glBindImageTexture(0, mRenderTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        }

        std::shared_ptr<myrender::Mesh> planeMesh = myrender::PrimitivesFactory::getPlane();
        planeMesh->applyTransform(glm::scale(glm::mat4(1.0f), glm::vec3(2.0f)));
        
        mRenderMesh = std::make_shared<myrender::RenderMesh>();
        mRenderMesh->start();
        mRenderMesh->setMeshData(*planeMesh);
        mRenderMesh->setShader(myrender::Shader::loadShader("ScreenPlane"));
    }

    void draw(myrender::Camera* camera) override
    {
        const glm::ivec2 currentScreenSize = myrender::Window::getCurrentWindow().getWindowSize();
        if( currentScreenSize.x != mRenderTextureSize.x ||
            currentScreenSize.y != mRenderTextureSize.y)
        {
            mRenderTextureSize = currentScreenSize;
            glBindTexture(GL_TEXTURE_2D, mRenderTexture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, mRenderTextureSize.x, mRenderTextureSize.y, 0, GL_RGBA, GL_FLOAT, NULL);
            glBindImageTexture(0, mRenderTexture, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F);
        }

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, mRenderTexture);

        mShader->bind(camera, &mTransform);

        const glm::vec2 nearAndFarPlane = glm::vec2(camera->getZNear(), camera->getZFar());
        float screenHalfSize = 0.5f * glm::tan(glm::radians(camera->getFov())) * nearAndFarPlane.x;
        float screenHalfSizeAspectRatio = screenHalfSize * camera->getRatio();
        glm::vec2 nearPlaneHalfSize = glm::vec2(screenHalfSizeAspectRatio, screenHalfSize);

        mShader->setUniform("pixelToView", 2.0f * nearPlaneHalfSize / glm::vec2(mRenderTextureSize));
        mShader->setUniform("nearPlaneHalfSize", nearPlaneHalfSize);
        mShader->setUniform("nearAndFarPlane", nearAndFarPlane);
        
        // Dispatch
        glDispatchCompute(mRenderTextureSize.x/16, mRenderTextureSize.y/16, 1);
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

        mRenderMesh->draw(camera);
    }

    myrender::Shader* getShader() { return mShader.get(); }

    const glm::mat4x4& getTransform() const { return mTransform; }
    void setTransform(glm::mat4x4 transfrom) { mTransform = transfrom; }

private:
    std::string mComputeShaderName;
    std::unique_ptr<myrender::Shader> mShader;
    uint32_t mRenderTexture;
    glm::ivec2 mRenderTextureSize;
    glm::mat4 mTransform;
    std::shared_ptr<myrender::RenderMesh> mRenderMesh;
};

class BaseScene : public myrender::Scene
{
public:
    BaseScene(const std::string& meanFieldPath, 
              std::optional<std::reference_wrapper<std::string>> mVarFieldPath = {}) 
    {
        // Load mean Field
        auto msdf = sdflib::SdfFunction::loadFromFile(meanFieldPath);
        if(msdf == nullptr)
        {
            return;
        }
        if(msdf->getFormat() != sdflib::SdfFunction::SdfFormat::TRILINEAR_OCTREE)
        {
            std::cout << "Error: Octree '" << meanFieldPath << "' is not a trilinear octree" << std::endl;
            return;
        }
        
        mMeanOctree = std::unique_ptr<sdflib::OctreeSdf>(static_cast<sdflib::OctreeSdf*>(msdf.get()));
        msdf.release();

        // Load Var field
        if(mVarFieldPath.has_value())
        {
            const std::string& vPath = mVarFieldPath.value();
            auto vsdf = sdflib::SdfFunction::loadFromFile(vPath);
            if(vsdf == nullptr)
            {
                return;
            }
            if(vsdf->getFormat() != sdflib::SdfFunction::SdfFormat::TRILINEAR_OCTREE)
            {
                std::cout << "Error: Octree '" << vPath << "' is not a trilinear octree" << std::endl;
                return;
            }

            mVarField = std::unique_ptr<sdflib::OctreeSdf>(static_cast<sdflib::OctreeSdf*>(vsdf.get()));
            vsdf.release();
        }
    }

protected:
    std::unique_ptr<sdflib::OctreeSdf> mMeanOctree;
    std::unique_ptr<sdflib::OctreeSdf> mVarField;
};

class MyScene2 : public BaseScene
{
public:
    MyScene2(const std::string& meanFieldPath, 
            std::optional<std::reference_wrapper<std::string>> mVarFieldPath = {}) : 
                BaseScene(meanFieldPath, mVarFieldPath)
    { }

    void start() override
    {
        namespace mr = myrender;
        { // Create camera
            auto nCamera = createSystem<mr::NavigationCamera>();
            nCamera->setDiableMouseOnRotation(false);
            nCamera->setPosition(glm::vec3(0.0, 0.0, 1.2));
            setMainCamera(nCamera);
        }

        mr::Window::getCurrentWindow().setBackgroudColor(glm::vec4(0.9f, 0.9f, 0.9f, 1.0f));
        
        auto rsdf = createSystem<RenderSdf>("OctreeRender");

        rsdf->setTransform(glm::translate(glm::mat4(1.0f), glm::vec3(-0.5f, -0.5f, -0.5f)));
        
        // const float invSize = 1.0f / mMeanOctree->getGridBoundingBox().getSize().x;
        rsdf->getShader()->setUniform("distanceScale", 1.0f / 32.0f);
        rsdf->getShader()->setUniform("minBorderValue", mMeanOctree->getOctreeMinBorderValue());
        rsdf->getShader()->setUniform("startGridSize", glm::vec3(mMeanOctree->getStartGridSize()));
        rsdf->getShader()->setBufferData("sdfOctree", mMeanOctree->getOctreeData());
    }
private:

};

class MyScene : public BaseScene
{
public:
    MyScene(const std::string& meanFieldPath, 
            std::optional<std::reference_wrapper<std::string>> mVarFieldPath = {}) : 
                BaseScene(meanFieldPath, mVarFieldPath), 
                mCurrentProp(Property::SDF)
    { }

    void start() override
    {
        init();
    }

    void init()
    {
        clearScene();

        namespace mr = myrender;
        { // Create camera
            auto nCamera = createSystem<mr::NavigationCamera>();
            nCamera->setDiableMouseOnRotation(false);
            nCamera->setPosition(glm::vec3(0.0, 0.0, 1.2));
            setMainCamera(nCamera);
        }

        mr::Window::getCurrentWindow().setBackgroudColor(glm::vec4(0.9f, 0.9f, 0.9f, 1.0f));

        mSelectArea = mMeanOctree->getGridBoundingBox();

        auto uiMan = createSystem<UiManager>();

        { // Create box
            auto myCube = mr::PrimitivesFactory::getCube();
            myCube->computeNormals();
            mBoxModel = createSystem<mr::RenderMesh>();
            mBoxModel->setMeshData(*myCube);
            mBoxModel->setShader(mr::Shader::loadShader("ModelPlaneCut"));
        }

        { // Create plane
            auto myPlane = mr::PrimitivesFactory::getPlane();
            mPlaneModel = createSystem<mr::RenderMesh>();
            mPlaneModel->setMeshData(*myPlane);

            if(mCurrentProp == SDF)
            {
                mPlaneModel->setShader(mr::Shader::loadShader("SdfOctreePlane"));
            }
            else
            {
                mPlaneModel->setShader(mr::Shader::loadShader("VarOctreePlane"));
            }

            mGizmoStartMatrix = glm::mat4x4(1.0f);
            mGizmoMatrix = mGizmoStartMatrix;
            mPlaneModel->setTransform(mGizmoMatrix);

            auto& s = mPlaneModel->getShader();
            const float invSize = 1.0f / mMeanOctree->getGridBoundingBox().getSize().x;
            s.setUniform("distanceScale", invSize);
            s.setUniform("worldToStartGridMatrix", glm::translate(glm::mat4(1.0f), -glm::vec3(-0.5f, -0.5f, -0.5f)));
            s.setUniform("startGridSize", glm::vec3(mMeanOctree->getStartGridSize()));
            s.setUniform("octreeValueRange", mMeanOctree->getOctreeValueRange());
            s.setUniform("printGrid", (uint32_t)true);
            s.setUniform("printIsolines", (uint32_t)true);
            s.setBufferData("sdfOctree", mMeanOctree->getOctreeData());

            if(mCurrentProp == SDF)
            {
                uiMan->addSlider(mPlaneModel->getShader(), "linesSpace", "Isolines space", 0.1, 0.002, 0.5);
                uiMan->addSlider(mPlaneModel->getShader(), "gridThickness", "Grid thickness", 0.01, 0.001, 0.02);
                uiMan->addSlider(mPlaneModel->getShader(), "linesThickness", "Isolines thickness", 2.5, 0.5, 6.0);
                uiMan->addSlider(mPlaneModel->getShader(), "surfaceThickness", "Surface thickness", 2.5, 0.5, 6.0);
            }
            else
            {
                // Compute max and min value
                float minVarValue = INFINITY;
                float maxVarValue = -INFINITY;

                const auto& octree = mVarField->getOctreeData();
                std::function<void(uint32_t)> processNode;
                processNode = [&](uint32_t nIdx)
                {
                    sdflib::IOctreeSdf::OctreeNode node = octree[nIdx];
                    if(!node.isLeaf())
                    {
                        for(uint32_t i=0; i < 8; i++) processNode(node.getChildrenIndex() + i);
                    }
                    else
                    {
                        const std::array<float, 8>* values = reinterpret_cast<const std::array<float, 8>*>(&octree[node.getChildrenIndex()]);
                        for(float v : *values)
                        {
                            minVarValue = glm::min(minVarValue, v);
                            maxVarValue = glm::max(maxVarValue, v);
                        }
                    }
                };

                glm::ivec3 gridSize = mVarField->getStartGridSize();
                uint32_t numNodes = gridSize.x * gridSize.y * gridSize.z;
                for(uint32_t n=0; n < numNodes; n++) processNode(n);

                std::cout << minVarValue << " // " << maxVarValue << std::endl;

                s.setUniform("varOctreeMinValue", minVarValue);
                s.setUniform("varOctreeMaxValue", maxVarValue);
                s.setBufferData("varOctree", mVarField->getOctreeData());
                s.setUniform("mode", static_cast<int>(mCurrentProp) - 1);
            }
        }
    }

    void update(float deltaTime) override
    {
        namespace mr = myrender;
        Scene::update(deltaTime);

        // Short keys to change cube cuts
		if(mr::Window::getCurrentWindow().isKeyPressed(GLFW_KEY_1))
		{
			mGizmoMatrix = mGizmoStartMatrix;
		} 
		else if(mr::Window::getCurrentWindow().isKeyPressed(GLFW_KEY_2))
		{
			mGizmoMatrix = glm::rotate(glm::mat4x4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f)) * mGizmoStartMatrix;
		} 
		else if(mr::Window::getCurrentWindow().isKeyPressed(GLFW_KEY_3))
		{
			mGizmoMatrix = glm::rotate(glm::mat4x4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * mGizmoStartMatrix;
		}
		else if(mr::Window::getCurrentWindow().isKeyPressed(GLFW_KEY_4))
		{
			mGizmoMatrix = glm::rotate(glm::mat4x4(1.0f), glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f)) * mGizmoStartMatrix;
		}
		else if(mr::Window::getCurrentWindow().isKeyPressed(GLFW_KEY_5))
		{
			mGizmoMatrix = glm::rotate(glm::mat4x4(1.0f), glm::radians(-90.0f), glm::vec3(-1.0f, 0.0f, 0.0f)) * mGizmoStartMatrix;
		}
		else if(mr::Window::getCurrentWindow().isKeyPressed(GLFW_KEY_6))
		{
			mGizmoMatrix = glm::rotate(glm::mat4x4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, -1.0f, 0.0f)) * mGizmoStartMatrix;
		}

        ImGuiIO& io = ImGui::GetIO();
    	ImGuizmo::SetRect(0, 0, io.DisplaySize.x, io.DisplaySize.y);

        if(!mr::Window::getCurrentWindow().isKeyPressed(GLFW_KEY_LEFT_CONTROL))
		{
			if(mr::Window::getCurrentWindow().isKeyPressed(GLFW_KEY_LEFT_ALT))
			{
				ImGuizmo::Manipulate(glm::value_ptr(getMainCamera()->getViewMatrix()), 
								glm::value_ptr(getMainCamera()->getProjectionMatrix()),
								ImGuizmo::OPERATION::ROTATE, ImGuizmo::MODE::LOCAL, glm::value_ptr(mGizmoMatrix));
			}
			else
			{
				ImGuizmo::Manipulate(glm::value_ptr(getMainCamera()->getViewMatrix()), 
								glm::value_ptr(getMainCamera()->getProjectionMatrix()),
								ImGuizmo::OPERATION::TRANSLATE_Z, ImGuizmo::MODE::LOCAL, glm::value_ptr(mGizmoMatrix));
			}
		}

        mPlaneModel->setTransform(mGizmoMatrix);
        const glm::vec3 planeNormal = glm::normalize(glm::vec3(mGizmoMatrix * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));
		const glm::vec3 planePoint = glm::vec3(mGizmoMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
        mBoxModel->getShader().setUniform("cutPlane", 
            glm::vec4(planeNormal.x, planeNormal.y, planeNormal.z, -glm::dot(planeNormal, planePoint)));
        mPlaneModel->getShader().setUniform("planeNormal", planeNormal);

        if(ImGui::BeginCombo("Property", propertiesStr[static_cast<uint32_t>(mCurrentProp)]))
        {
            const Property oldProp = mCurrentProp;
            for (int n = 0; n < IM_ARRAYSIZE(propertiesStr); n++)
            {
                Property prop = static_cast<Property>(n);
                if(prop != Property::SDF && mVarField == nullptr) continue;

                bool isSelected = mCurrentProp == prop;
                if (ImGui::Selectable(propertiesStr[n], isSelected))
                    mCurrentProp = static_cast<Property>(n);
                if(isSelected)
                    ImGui::SetItemDefaultFocus();
            }

            ImGui::EndCombo();

            if(mCurrentProp != oldProp)
            {
                init();
            }
        }
    }

private:
    enum Property
    {
        SDF,
        VARIANCE,
        PSURFACE,
        PVOLUME
    };

    const char* propertiesStr[4] = {
        "SDF",
        "Variance",
        "PSurface",
        "PVolume"
    };
    
    Property mCurrentProp;

    sdflib::BoundingBox mSelectArea;

    std::shared_ptr<myrender::RenderMesh> mBoxModel;
    std::shared_ptr<myrender::RenderMesh> mPlaneModel;

    glm::mat4x4 mGizmoStartMatrix;
    glm::mat4x4 mGizmoMatrix;
};

int main(int argc, char** argv)
{
    args::ArgumentParser parser("");
    args::HelpFlag help(parser, "help", "Display help menu", {'h', "help"});
    args::Positional<std::string> meanFieldPath(parser, "mean_field", "Mean Field");
    args::ValueFlag<std::string> varFieldPath(parser, "var_field", "Var Field", {"var_field"});

    try
    {
        parser.ParseCLI(argc, argv);
    }
    catch(args::Help)
    {
        std::cerr << parser;
        return 0;
    }

    namespace mr = myrender;
    mr::ShaderProgramLoader::getInstance()->addSearchPath("./build/_deps/myrender_lib-src/shaders");
    mr::ShaderProgramLoader::getInstance()->addSearchPath("./viewer/shaders");
    mr::MainLoop ml;
    using MScene = MyScene;
    // using MScene = MyScene;
    if(varFieldPath)
    {
        MScene scene(args::get(meanFieldPath), args::get(varFieldPath));
        ml.start(scene);
    }
    else
    {
        MScene scene(args::get(meanFieldPath), std::nullopt);
        ml.start(scene);
    }
}