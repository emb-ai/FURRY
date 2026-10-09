"""Add session-menu input to the already adapted, pinned hello_xr source.

This runs after prepare_android.py's original OpenXR transforms. A changed or
partially transformed source is an error, so a build cannot silently ship the
controller bindings that disagree with the runtime.
"""


def _replace_once(text: str, before: str, after: str, label: str) -> str:
    count = text.count(before)
    if count != 1:
        raise ValueError(f"Menu input patch: {label}: expected 1 occurrence, found {count}")
    return text.replace(before, after, 1)


def adapt_menu_input(text: str) -> str:
    """Return adapted C++ with Menu, A/B/X/Y, left-stick view and two controller aim rays."""
    def patch(before: str, after: str, label: str) -> None:
        nonlocal text
        text = _replace_once(text, before, after, label)

    patch("ButtonLatch recordLatch, captureLatch;", "ButtonLatch recordLatch;", "record latch")
    patch(
        "XrAction calibrateAction{XR_NULL_HANDLE}, pauseAction{XR_NULL_HANDLE}, resetAction{XR_NULL_HANDLE}, "
        "recordAction{XR_NULL_HANDLE}, viewAction{XR_NULL_HANDLE}, captureModeAction{XR_NULL_HANDLE};",
        "XrAction calibrateAction{XR_NULL_HANDLE}, pauseAction{XR_NULL_HANDLE}, resetAction{XR_NULL_HANDLE}, recordAction{XR_NULL_HANDLE}, viewAction{XR_NULL_HANDLE};\n"
        "        XrAction menuAimAction{XR_NULL_HANDLE}, menuTriggerAction{XR_NULL_HANDLE};\n"
        "        std::array<XrSpace, Side::COUNT> menuAimSpace{};\n"
        "        std::array<XrBool32, Side::COUNT> menuAimActive{};\n"
        "        std::array<XrActionStateFloat, Side::COUNT> menuTrigger{};",
        "menu action fields",
    )
    for action, action_name, localized in (
        ("captureModeAction", "capture_mode", "Select guided walking capture"),
    ):
        patch(
            f'            strcpy_s(actionInfo.actionName, "{action_name}");\n'
            f'            strcpy_s(actionInfo.localizedActionName, "{localized}");\n'
            f"            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.{action}));",
            "",
            f"remove {action}",
        )
    patch('"quit_session"', '"session_menu"', "menu action name")
    patch('"Reset scene"', '"Open session menu"', "menu action label")
    patch(
        "CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.viewAction));",
        """CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.viewAction));
            // Aim and trigger have independent left/right subaction paths.
            actionInfo.countSubactionPaths = uint32_t(m_input.handSubactionPath.size());
            actionInfo.subactionPaths = m_input.handSubactionPath.data();
            actionInfo.actionType = XR_ACTION_TYPE_POSE_INPUT;
            strcpy_s(actionInfo.actionName, "menu_aim");
            strcpy_s(actionInfo.localizedActionName, "Menu pointer");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.menuAimAction));
            actionInfo.actionType = XR_ACTION_TYPE_FLOAT_INPUT;
            strcpy_s(actionInfo.actionName, "menu_trigger");
            strcpy_s(actionInfo.localizedActionName, "Select menu item");
            CHECK_XRCMD(xrCreateAction(m_input.actionSet, &actionInfo, &m_input.menuTriggerAction));""",
        "create aim and trigger actions",
    )
    patch(
        """            XrPath aButton, bButton, xButton, yButton;
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/right/input/a/click",&aButton));
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/right/input/b/click",&bButton));
            bindings.push_back({m_input.calibrateAction,aButton});
            bindings.push_back({m_input.pauseAction,bButton});
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/x/click",&xButton));
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/y/click",&yButton));
            bindings.push_back({m_input.resetAction,xButton});
            bindings.push_back({m_input.recordAction,yButton});
            XrPath viewClick;
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/thumbstick/click",&viewClick));
            bindings.push_back({m_input.viewAction,viewClick});
            XrPath captureClick;
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/right/input/thumbstick/click",&captureClick));
            bindings.push_back({m_input.captureModeAction,captureClick});""",
        """            XrPath aButton, bButton, xButton, yButton;
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/right/input/a/click",&aButton));
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/right/input/b/click",&bButton));
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/y/click",&yButton));
            bindings.push_back({m_input.calibrateAction,aButton});
            bindings.push_back({m_input.pauseAction,bButton});
            bindings.push_back({m_input.recordAction,yButton});
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/x/click",&xButton));
            bindings.push_back({m_input.resetAction,xButton});
            XrPath viewClick;
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/thumbstick/click",&viewClick));
            bindings.push_back({m_input.viewAction,viewClick});
            std::array<XrPath, Side::COUNT> menuAimPath{};
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/left/input/aim/pose",&menuAimPath[Side::LEFT]));
            CHECK_XRCMD(xrStringToPath(m_instance,"/user/hand/right/input/aim/pose",&menuAimPath[Side::RIGHT]));
            bindings.push_back({m_input.menuAimAction,menuAimPath[Side::LEFT]});
            bindings.push_back({m_input.menuAimAction,menuAimPath[Side::RIGHT]});
            bindings.push_back({m_input.menuTriggerAction,triggerValuePath[Side::LEFT]});
            bindings.push_back({m_input.menuTriggerAction,triggerValuePath[Side::RIGHT]});""",
        "Oculus menu bindings",
    )
    patch(
        """for(int button=0;button<6;button++){
            XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
            const XrAction buttons[]={m_input.calibrateAction,m_input.pauseAction,m_input.resetAction,m_input.recordAction,m_input.viewAction,m_input.captureModeAction};
            info.action=buttons[button];
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            CHECK_XRCMD(xrGetActionStateBoolean(m_session,&info,&state));
            bool press=button==3?recordLatch.Update(state.isActive,state.currentState):button==5?captureLatch.Update(state.isActive,state.currentState):(state.isActive && state.changedSinceLastSync && state.currentState);
            if(press){
                if(button==0)G1Calibrate();else if(button==1)G1ToggleTracking();
                else if(button==2)G1Reset();else if(button==3)G1ToggleRecording();else if(button==4)G1ToggleView();else G1CycleCaptureMode();
            }
        }""",
        """for(int button=0;button<5;button++){
            XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
            const XrAction buttons[]={m_input.calibrateAction,m_input.pauseAction,m_input.resetAction,m_input.recordAction,m_input.viewAction};
            info.action=buttons[button];
            XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
            CHECK_XRCMD(xrGetActionStateBoolean(m_session,&info,&state));
            bool focused=IsSessionFocused();
            bool press=button==3?recordLatch.Update(state.isActive && focused,state.currentState)
                :(state.isActive && state.changedSinceLastSync && state.currentState);
            if(press && focused){
                if(button==0)G1Calibrate();else if(button==1)G1ToggleTracking();
                else if(button==2)G1Reset();else if(button==3)G1ToggleRecording();else G1ToggleView();
            }
        }""",
        "A B X Y and view dispatch",
    )
    patch("            G1Reset();", "            G1ToggleMenu();", "hardware menu dispatch")
    patch(
        "if ((quitValue.isActive == XR_TRUE) && (quitValue.changedSinceLastSync == XR_TRUE) && (quitValue.currentState == XR_TRUE)) {",
        "if (IsSessionFocused() && (quitValue.isActive == XR_TRUE) && (quitValue.changedSinceLastSync == XR_TRUE) && (quitValue.currentState == XR_TRUE)) {",
        "focused hardware menu",
    )
    patch(
        "        m_input.handActive = {XR_FALSE, XR_FALSE};",
        """        m_input.handActive = {XR_FALSE, XR_FALSE};
        m_input.menuAimActive = {XR_FALSE, XR_FALSE};
        for(auto& trigger : m_input.menuTrigger) trigger = XrActionStateFloat{XR_TYPE_ACTION_STATE_FLOAT};""",
        "clear menu input before sync",
    )
    patch(
        "            m_input.handActive[hand] = poseState.isActive;",
        """            m_input.handActive[hand] = poseState.isActive;
            getInfo.action = m_input.menuAimAction;
            XrActionStatePose aimState{XR_TYPE_ACTION_STATE_POSE};
            CHECK_XRCMD(xrGetActionStatePose(m_session, &getInfo, &aimState));
            m_input.menuAimActive[hand] = aimState.isActive && IsSessionFocused();
            getInfo.action = m_input.menuTriggerAction;
            CHECK_XRCMD(xrGetActionStateFloat(m_session, &getInfo, &m_input.menuTrigger[hand]));
            if(!IsSessionFocused()){
                m_input.menuTrigger[hand].isActive = XR_FALSE;
                m_input.menuTrigger[hand].currentState = 0.f;
            }""",
        "poll aim and trigger state",
    )
    patch(
        "        CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.handSpace[Side::RIGHT]));",
        """        CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.handSpace[Side::RIGHT]));
        actionSpaceInfo.action = m_input.menuAimAction;
        actionSpaceInfo.subactionPath = m_input.handSubactionPath[Side::LEFT];
        CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.menuAimSpace[Side::LEFT]));
        actionSpaceInfo.subactionPath = m_input.handSubactionPath[Side::RIGHT];
        CHECK_XRCMD(xrCreateActionSpace(m_session, &actionSpaceInfo, &m_input.menuAimSpace[Side::RIGHT]));""",
        "create both aim spaces",
    )
    patch(
        "                xrDestroySpace(m_input.handSpace[hand]);",
        """                xrDestroySpace(m_input.handSpace[hand]);
                if(m_input.menuAimSpace[hand] != XR_NULL_HANDLE)
                    xrDestroySpace(m_input.menuAimSpace[hand]);""",
        "destroy aim spaces before action set",
    )
    patch(
        "        G1SubmitTracking(tracking);\n        G1PrepareFrame();",
        """        G1SubmitTracking(tracking);
        for(int hand=0;hand<2;hand++){
            XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
            location.pose.orientation.w = 1.f;
            XrResult r=xrLocateSpace(m_input.menuAimSpace[hand],m_appSpace,predictedDisplayTime,&location);
            const auto required=XR_SPACE_LOCATION_POSITION_VALID_BIT|XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
            bool valid=XR_SUCCEEDED(r) && (location.locationFlags&required)==required
                && m_input.menuAimActive[hand] && IsSessionFocused();
            const auto& trigger=m_input.menuTrigger[hand];
            G1SubmitMenuRay(hand,trackedPose(location.pose),valid,
                trigger.isActive && IsSessionFocused(),trigger.currentState);
        }
        G1PrepareFrame();""",
        "locate rays before menu frame",
    )
    patch(
        "TrackingFrame missing;missing.sequence=++trackingSequence;missing.xr_time_ns=predictedDisplayTime;G1SubmitTracking(missing);",
        """TrackingFrame missing;missing.sequence=++trackingSequence;missing.xr_time_ns=predictedDisplayTime;G1SubmitTracking(missing);
            for(int hand=0;hand<2;hand++)G1SubmitMenuRay(hand,TrackedPose{},false,false,0.f);""",
        "invalidate rays when views are missing",
    )
    patch(
        "            layers.push_back(passthrough.Layer());",
        "            if(G1PassthroughVisible())layers.push_back(passthrough.Layer());",
        "passthrough visibility",
    )
    for removed in ("captureModeAction", "captureLatch", "/user/hand/right/input/thumbstick/click"):
        if removed in text:
            raise ValueError(f"Menu input patch: obsolete binding remains: {removed}")
    return text
