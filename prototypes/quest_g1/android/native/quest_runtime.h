#pragma once
#include <cstdint>
#include "retarget.h"
struct android_app;
void G1Initialize(android_app* app);
void G1Shutdown(bool interrupted=false);
void G1SetActive(bool active);
void G1Grip(int hand, float value);
void G1Reset();
void G1PrepareFrame();
void G1Render(const float* viewProjection);
void G1CaptureFrame(int width,int height);
void G1SubmitTracking(const TrackingFrame& input);
void G1ReferenceSpaceChange(int64_t changeTime);
void G1Calibrate();
void G1ToggleTracking();
void G1ToggleRecording();
