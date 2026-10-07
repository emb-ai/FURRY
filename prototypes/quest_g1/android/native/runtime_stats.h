#pragma once
#include <array>
#include "camera_fusion.h"
#include <cstdint>
struct RuntimeStats {
    CameraOverlay cameraOverlay;
    int physicsWorkers=0;
    int cameraState=0,cameraLegs=0;bool cameraConnected=false,cameraClockSynced=false;
    double cameraClockOffsetMs=0,cameraClockRttMs=-1;
    double cameraAgeMs=-1,cameraFitMm=-1,cameraWeight=0;
    double cameraPositionError=-1,cameraOrientationError=-1,visualScale=1;
    double published=0, cycleMs=0, physicsMs=0, gmrMs=0, inferenceMs=0;
    double simTime=0, realTimeFactor=0, inputAgeMs=0, bodyGapMs=0, confidence=0;
    std::array<int,2> footContacts{};
    double legErrorDegrees=0;
    double height=0, tilt=0, depthMm=0, residual=0, positionError=0;
    std::array<double,2> targetXY{},actualXY{},commandXY{},velocityXY{};
    int contacts=0, overlaps=0, pairs=0, solverIterations=0, constraints=0;
    int gmrIterations=0, status=0, reason=0, warnings=0;
    uint64_t bodyVersion=0, droppedInputs=0, overruns=0;
};
