#include "episode_manifest.h"
#include <chrono>
#include <filesystem>
#include <functional>
#include <iostream>

static void Check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
static void Reject(const std::function<void()> &run) {
    bool rejected = false;
    try {
        run();
    } catch (const std::exception &) {
        rejected = true;
    }
    Check(rejected, "Invalid fixture was accepted");
}
int main() {
    auto folder = std::filesystem::temp_directory_path() /
                  ("furry-manifest-" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(folder);
    auto write = [&](const std::string &value) {
        std::ofstream(folder / "manifest.json") << value;
    };
    auto load = [&] { return episodereplay::LoadManifest(folder.string()); };
    try {
        for (const auto *scene : {"lab", "stand", "cup", "push_t"}) {
            write(std::string("{\"scene\":\"") + scene + "\",\"nq\":50,\"nv\":48,\"nu\":43}");
            auto manifest = load();
            Check(manifest.scene == scene, "Wrong scene selected");
            manifest.ValidateDimensions(50, 48, 43);
            std::string model =
                std::string(scene) == "lab" ? "scene.xml" : "scene-" + std::string(scene) + ".xml";
            Check(manifest.ModelPath("assets") == "assets/" + model,
                  "Wrong packaged model selected");
            Reject([&] { manifest.ValidateDimensions(43, 48, 43); });
            Reject([&] { manifest.ValidateDimensions(50, 42, 43); });
            Reject([&] { manifest.ValidateDimensions(50, 48, 29); });
        }
        write("{\"sha256\":{\"scene\":\"cup\"},\"limitations\":[\"scene "
              "cup\",{\"nq\":999}],\"nq\":50}");
        auto legacy = load();
        Check(legacy.scene == "lab", "Nested scene overrode legacy fallback");
        legacy.ValidateDimensions(50, 48, 43);
        Check(!legacy.catchUp&&!legacy.firstPerson,"Legacy episode acquired catch-up");
        Check(!legacy.catchUpV2&&!legacy.neutralWrists&&legacy.travelGainPercent==100,
              "Legacy episode acquired new retarget settings");
        write("{\"catch_up_v2\":true,\"neutral_wrists\":true,\"travel_gain_percent\":90}");
        auto updated=load();
        Check(updated.catchUpV2&&updated.neutralWrists&&updated.travelGainPercent==90,
              "New episode retarget settings lost");
        for(const auto* bad:{"{\"catch_up_v2\":1}","{\"neutral_wrists\":null}",
                            "{\"travel_gain_percent\":49}","{\"travel_gain_percent\":121}",
                            "{\"travel_gain_percent\":90.5}","{\"neutral_wrists\":false,\"neutral_wrists\":true}"}){
            write(bad);Reject([&]{load();});
        }
        write("{\"catch_up\":true,\"first_person\":false}");
        Check(load().catchUp&&!load().firstPerson,"Episode control settings lost");
        for(const auto* bad:{"{\"catch_up\":1}","{\"first_person\":null}","{\"catch_up\":false,\"catch_up\":true}"}){write(bad);Reject([&]{load();});}
        Check(legacy.PolicyPath("assets",folder.string())=="assets/policy.onnx","Legacy policy fallback changed");
        std::ofstream(folder/"policy.onnx",std::ios::binary)<<"abc";
        const std::string digest="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        Check(questpolicy::FileSha256((folder/"policy.onnx").string())==digest,"SHA256 abc vector failed");
        write("{\"policy_file\":\"policy-candidate.onnx\",\"policy_sha256\":\""+digest+"\"}");
        Check(load().PolicyPath("missing-assets",folder.string())==(folder/"policy.onnx").string(),"Archived policy was not used");
        std::ofstream(folder/"policy.onnx",std::ios::binary)<<"modified";
        Reject([&]{load().PolicyPath("assets",folder.string());});
        for(const auto* invalid:{"{\"policy_file\":\"policy.onnx\"}",
            "{\"policy_file\":\"../policy.onnx\",\"policy_sha256\":\"bad\"}",
            "{\"policy_sha256\":\"bad\"}"}){write(invalid);Reject([&]{load();});}
        std::ofstream(folder/"empty",std::ios::binary);
        Check(questpolicy::FileSha256((folder/"empty").string())=="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","SHA256 empty vector failed");
        write("{\"sc\\u0065ne\":\"c\\u0075p\",\"nested\":[true,false,null,-1.2e+3,{\"unicode\":"
              "\"\\uD83D\\uDC3B\"}]}");
        Check(load().scene == "cup", "Valid JSON escape failed");
        for (const auto *invalid :
             {"{} trailing", "{\"scene\":\"../cup\"}", "{\"scene\":false}", "{\"scene\":null}",
              "{\"scene\":\"other\"}", "{\"scene\":\"cup\",\"scene\":\"stand\"}", "{\"nq\":-1}",
              "{\"nv\":1.5}", "{\"nu\":\"43\"}", "{\"nq\":999999999999999}", "{\"nq\":01}",
              "{\"nested\":[}", "{\"scene\":\"cup}"}) {
            write(invalid);
            Reject([&] { load(); });
        }
        std::filesystem::remove(folder / "manifest.json");
        Reject([&] { load(); });
        {
            std::ofstream events(folder / "events.csv");
            events << "receive_ns,sequence,event\n10,1,record_button_start\n20,2,menu_open\n30,3,"
                      "menu_close\n40,4,user_pause\n50,5,session_resume\n60,6,focus_pause\n70,7,"
                      "focus_resume\n";
        }
        Check(episodereplay::PauseTimes(folder.string()) == std::vector<int64_t>({20, 40, 60}),
              "Pause events were not preserved");
        {
            std::ofstream events(folder / "events.csv");
            events << "receive_ns,sequence,event\n20,2,menu_open\n10,3,user_pause\n";
        }
        Reject([&] { episodereplay::PauseTimes(folder.string()); });
        std::filesystem::remove_all(folder);
        std::cout << "Episode scene/dimension/JSON/pause fixtures passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::filesystem::remove_all(folder);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
