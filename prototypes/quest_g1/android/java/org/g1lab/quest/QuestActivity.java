package org.g1lab.quest;

import android.app.NativeActivity;
import android.content.Intent;
import android.media.projection.MediaProjectionManager;
import android.os.Bundle;

/** JNI entry points execute UI operations on the main thread. */
public final class QuestActivity extends NativeActivity {
    private static final int CAPTURE_REQUEST = 4701;
    private boolean requesting;

    public void toggleOperatorVideo() {
        runOnUiThread(() -> {
            if (requesting) return;
            if (OperatorVideoService.state == 3) { stopOperatorVideo(); return; }
            if (OperatorVideoService.state != 0) return;
            try {
                requesting = true;
                OperatorVideoService.state = 1;
                OperatorVideoService.message = "Подтвердите запись в системном окне";
                MediaProjectionManager manager = getSystemService(MediaProjectionManager.class);
                startActivityForResult(manager.createScreenCaptureIntent(), CAPTURE_REQUEST);
            } catch (RuntimeException error) {
                requesting = false;
                OperatorVideoService.fail("Не удалось запросить запись", error);
            }
        });
    }
    public int getOperatorVideoState() { return OperatorVideoService.state; }
    public String getOperatorVideoMessage() { return OperatorVideoService.message; }
    public void stopOperatorVideo() {
        runOnUiThread(() -> stopService(new Intent(this, OperatorVideoService.class)));
    }
    @Override protected void onActivityResult(int request, int result, Intent data) {
        super.onActivityResult(request, result, data);
        if (request != CAPTURE_REQUEST) return;
        requesting = false;
        if (result != RESULT_OK || data == null) {
            OperatorVideoService.state = 0;
            OperatorVideoService.message = "Запись видео отменена";
            return;
        }
        OperatorVideoService.state = 2;
        OperatorVideoService.message = "Запуск видео...";
        try {
            startForegroundService(new Intent(this, OperatorVideoService.class)
                .putExtra("resultCode", result).putExtra("resultData", data));
        } catch (RuntimeException error) {
            OperatorVideoService.fail("Не удалось запустить запись", error);
        }
    }
    @Override protected void onDestroy() {
        stopService(new Intent(this, OperatorVideoService.class));
        if (requesting) { OperatorVideoService.state = 0; requesting = false; }
        super.onDestroy();
    }
}
