#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

struct CaptureStage {const char* id;const char* title;const char* hint;};
inline constexpr CaptureStage kTrainStages[]={
    {"forward_slow","Вперёд: медленно","Короткие шаги, руки свободно по бокам"},
    {"forward_variable","Вперёд: меняйте темп","Обычный мах рук, меняйте темп без бега"},
    {"carry_two_hands","Переноска: двумя руками","Идите, будто несёте коробку перед собой"},
    {"carry_one_hand","Переноска: одной рукой","Одна рука занята; на полпути смените её"},
    {"backward","Короткие шаги назад","Проверяйте свободное место вокруг"},
    {"sidestep_left","Приставные шаги влево","Возвращайтесь, если мало места"},
    {"sidestep_right","Приставные шаги вправо","Возвращайтесь, если мало места"},
    {"turn_left","Ходьба с поворотами влево","Поворачивайте небольшими шагами"},
    {"turn_right","Ходьба с поворотами вправо","Поворачивайте небольшими шагами"},
    {"carry_turns","Повороты с предметом","Руки перед собой, как с коробкой"},
    {"start_stop_normal","Старт и остановка","Пройдите 3-5 шагов и постойте 2-3 с"},
    {"start_stop_short","Короткие перемещения","Шагните 1-2 раза, затем остановитесь"},
    {"carry_start_stop","Остановки с предметом","Идите и останавливайтесь с руками впереди"},
    {"stand_reach_place","Стоя: взять и поставить","Стопы на месте, тянитесь и верните руки"},
    {"stand_head","Стойте: голова и корпус","Стопы на месте, смотрите по сторонам"}
};
inline constexpr CaptureStage kTestStages[]={
    {"test_walk_carry","Тест: ходьба и переноска","Чередуйте свободные руки и переноску"},
    {"test_side_back","Тест: боком и назад","Чередуйте направления по своему выбору"},
    {"test_turns","Тест: повороты","Меняйте маршрут и положение рук"},
    {"test_start_stop","Тест: старт и остановка","Чередуйте свободные руки и переноску"},
    {"test_stand","Тест: движения стоя","Тянитесь за предметом, смотрите на него"}
};

// Pure clock/state machine: simulation state cannot end a human capture.
// Only adjacent eligible ticks count; stalls, pauses and stale input do not.
struct GuidedCapture {
    int mode=0,stage=0; // 0 manual, 1 train, 2 held-out test
    bool running=false,paused=false,calibrated=false,completed=false;
    double countdown=5,lastNow=0;
    bool previousEligible=false;
    std::array<double,15> accepted{};
    int Count()const{return mode==1?15:mode==2?5:0;}
    const CaptureStage& Current()const{return mode==2?kTestStages[std::min(stage,4)]:kTrainStages[std::min(stage,14)];}
    const char* Split()const{return mode==2?"test":"train";}
    double Total()const{double t=0;for(double s:accepted)t+=s;return t;}
    double Remaining()const{return std::max(0.,60-accepted[std::min(stage,14)]);}
    void Cycle(){if(running)return;mode=(mode+1)%3;completed=false;accepted.fill(0);stage=0;}
    void Start(double now){running=true;paused=false;calibrated=false;completed=false;stage=0;countdown=5;accepted.fill(0);lastNow=now;previousEligible=false;}
    void Calibrate(){calibrated=true;countdown=5;previousEligible=false;}
    void Invalidate(){calibrated=false;previousEligible=false;}
    void Pause(){paused=!paused;previousEligible=false;}
    void Stop(){running=false;previousEligible=false;}
    void Tick(double now,bool valid){
        double dt=now-lastNow;lastNow=now;
        bool eligible=running&&!completed&&!paused&&calibrated&&valid;
        bool count=eligible&&previousEligible&&dt>0&&dt<=.1;previousEligible=eligible;
        if(!count)return;
        if(countdown>0){countdown=std::max(0.,countdown-dt);return;}
        accepted[stage]=std::min(60.,accepted[stage]+dt);
        if(accepted[stage]>=60){
            if(stage+1==Count()){completed=true;return;}
            stage++;countdown=5;previousEligible=false;
        }
    }
    int Status(bool focused,bool fresh)const{
        if(completed)return 6;
        if(!running)return 0;
        if(!calibrated)return 1;
        if(paused||!focused)return 2;
        if(!fresh)return 3;
        return countdown>0?4:5;
    }
    static const char* StatusName(int status){
        const char* names[]={"ready","calibrate","paused","tracking_lost","prepare","capture","complete"};
        return names[std::clamp(status,0,6)];
    }
    void WritePlan(const std::string& path)const{
        std::ofstream f;f.exceptions(std::ios::badbit|std::ios::failbit);f.open(path+"/capture_plan.json");
        f<<"{\"schema_version\":1,\"protocol\":\"quest_virtual_legs_v2\",\"split\":\""<<Split()
         <<"\",\"capture_kind\":\"human_skeleton_only\",\"target_seconds\":"<<Count()*60
         <<",\"preparation_seconds_per_stage\":5,\"lower_body\":\"Meta estimated legs\",\"stages\":[";
        for(int i=0;i<Count();i++){
            const auto& s=mode==2?kTestStages[i]:kTrainStages[i];
            f<<(i?",":"")<<"{\"id\":\""<<s.id<<"\",\"title\":\""<<s.title<<"\",\"hint\":\""<<s.hint<<"\",\"seconds\":60}";
        }
        f<<"]}\n";f.close();
    }
    void WriteSummary(const std::string& path)const{
        std::ofstream f;f.exceptions(std::ios::badbit|std::ios::failbit);f.open(path+"/capture_summary.json");
        f<<std::setprecision(17)<<"{\"schema_version\":1,\"split\":\""<<Split()<<"\",\"completed\":"<<(completed?"true":"false")
         <<",\"accepted_seconds\":"<<Total()<<",\"target_seconds\":"<<Count()*60<<",\"stage_seconds\":[";
        for(int i=0;i<Count();i++)f<<(i?",":"")<<accepted[i];
        f<<"]}\n";f.close();
    }
};
